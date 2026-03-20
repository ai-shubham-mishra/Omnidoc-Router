"""
Redis client for session caching.
Provides write-through cache with automatic TTL refresh.
"""
import os
import json
import gzip
import logging
from typing import Optional, Dict, Any
from redis import asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", "259200"))  # 3 days


class RedisClient:
    """Async Redis client with connection pooling and error handling."""
    
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.enabled = True
    
    async def connect(self):
        """Establish Redis connection."""
        try:
            self.redis = await aioredis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=False,  # We handle binary data (gzip)
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
            # Test connection
            await self.redis.ping()
            logger.info(f"✅ Redis connected: {REDIS_URL}")
            self.enabled = True
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable: {e}. Falling back to MongoDB-only mode.")
            self.enabled = False
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session from cache with TTL refresh (sliding window).
        Returns None if not found or Redis unavailable.
        """
        if not self.enabled or not self.redis:
            return None
        
        try:
            key = f"session:{session_id}"
            cached = await self.redis.get(key)
            
            if cached:
                # Sliding window: refresh TTL on every access
                await self.redis.expire(key, SESSION_TTL)
                
                # Decompress and deserialize
                decompressed = gzip.decompress(cached)
                session = json.loads(decompressed.decode('utf-8'))
                logger.debug(f"✅ Cache HIT: {session_id[:8]}... (TTL refreshed)")
                return session
            
            logger.debug(f"❌ Cache MISS: {session_id[:8]}...")
            return None
        
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None
    
    async def set_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """
        Store session in cache with TTL.
        Returns True if successful, False otherwise.
        """
        if not self.enabled or not self.redis:
            return False
        
        try:
            key = f"session:{session_id}"
            
            # Serialize and compress
            serialized = json.dumps(session_data, default=str).encode('utf-8')
            compressed = gzip.compress(serialized)
            
            # Store with TTL
            await self.redis.setex(key, SESSION_TTL, compressed)
            logger.debug(f"✅ Cache SET: {session_id[:8]}... ({len(compressed)} bytes, TTL: {SESSION_TTL}s)")
            return True
        
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session from cache."""
        if not self.enabled or not self.redis:
            return False
        
        try:
            key = f"session:{session_id}"
            deleted = await self.redis.delete(key)
            if deleted:
                logger.debug(f"✅ Cache DEL: {session_id[:8]}...")
            return bool(deleted)
        
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics."""
        if not self.enabled or not self.redis:
            return {"status": "disabled"}
        
        try:
            info = await self.redis.info("memory")
            return {
                "status": "connected",
                "used_memory_mb": round(info["used_memory"] / 1024 / 1024, 2),
                "used_memory_human": info["used_memory_human"],
                "connected_clients": (await self.redis.info("clients"))["connected_clients"],
            }
        except Exception as e:
            logger.error(f"Redis STATS error: {e}")
            return {"status": "error", "error": str(e)}


# Singleton instance
redis_client = RedisClient()

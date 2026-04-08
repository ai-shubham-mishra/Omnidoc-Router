"""
Pinecone Vector Database Client for Omnidoc Router.
Provides semantic search and long-term context persistence.

Key Features:
- Embed files and messages using Pinecone's llama-text-embed-v2 model
- Store vectors with rich metadata (session_id, run_id, user_id, org_id, type)
- Hybrid search: semantic similarity + metadata filtering
- Persistent context beyond Redis TTL (3 days)

Architecture:
- Uses Pinecone's hosted inference API (no external embedding service needed)
- Namespace pattern: org_{org_id} or session_{session_id}
- Vector metadata: session_id, run_id, user_id, org_id, type, content_preview, filename, etc.
"""
import os
import logging
import hashlib
import tiktoken
from typing import List, Dict, Any, Optional
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

from components.KeyVaultClient import get_secret

load_dotenv()
logger = logging.getLogger(__name__)

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "llama-text-embed-v2-index")
PINECONE_MODEL = "llama-text-embed-v2"
EMBEDDING_DIMENSION = 1024  # llama-text-embed-v2 dimension
MAX_TOKENS_PER_CHUNK = 2048  # llama-text-embed-v2 max input


class PineconeClient:
    """
    Pinecone client for semantic search and context persistence.
    Uses Pinecone's hosted inference API for embeddings.
    """
    
    def __init__(self):
        self.pc = None
        self.index = None
        self.enabled = False
        self.tokenizer = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Pinecone client with API key from Key Vault."""
        try:
            # Fetch API key from Azure Key Vault
            api_key = get_secret("VECTOR_SEARCH_API_KEY")
            
            if not api_key:
                logger.warning("⚠️ VECTOR_SEARCH_API_KEY not found in Key Vault - Pinecone disabled")
                return
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=api_key)
            
            # Connect to existing index
            if PINECONE_INDEX_NAME not in [idx.name for idx in self.pc.list_indexes()]:
                logger.error(f"❌ Pinecone index '{PINECONE_INDEX_NAME}' not found")
                return
            
            self.index = self.pc.Index(PINECONE_INDEX_NAME)
            
            # Initialize tokenizer for chunking
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Tokenizer init failed, using fallback: {e}")
                self.tokenizer = None
            
            self.enabled = True
            logger.info(f"✅ Pinecone initialized: {PINECONE_INDEX_NAME} (model: {PINECONE_MODEL})")
        
        except Exception as e:
            logger.error(f"❌ Pinecone initialization failed: {e}")
            self.enabled = False
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Fallback: rough estimate (1 token ≈ 4 chars)
        return len(text) // 4
    
    def _chunk_text(self, text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> List[str]:
        """
        Split text into chunks that fit within token limit.
        Uses sliding window with overlap for better context.
        """
        if self._count_tokens(text) <= max_tokens:
            return [text]
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            word_tokens = self._count_tokens(word)
            if current_tokens + word_tokens > max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                # Overlap: keep last 10% of previous chunk
                overlap_size = max(1, len(current_chunk) // 10)
                current_chunk = current_chunk[-overlap_size:] + [word]
                current_tokens = sum(self._count_tokens(w) for w in current_chunk)
            else:
                current_chunk.append(word)
                current_tokens += word_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _generate_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate deterministic vector ID from content + metadata."""
        hash_input = f"{metadata.get('session_id', '')}_{metadata.get('type', '')}_{content[:100]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def embed_and_upsert_message(
        self,
        session_id: str,
        message: str,
        role: str,
        user_id: str,
        org_id: str,
        run_id: Optional[str] = None,
        workflow_name: Optional[str] = None,
    ) -> bool:
        """
        Embed and store a chat message in Pinecone.
        
        Args:
            session_id: Chat session ID
            message: Message content
            role: 'user' or 'assistant'
            user_id: User ID
            org_id: Organization ID
            run_id: Workflow run ID (if message is part of workflow)
            workflow_name: Name of workflow (if applicable)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Chunk message if too long
            chunks = self._chunk_text(message)
            
            vectors = []
            for i, chunk in enumerate(chunks):
                # Generate embedding using Pinecone's inference API
                embedding = self.pc.inference.embed(
                    model=PINECONE_MODEL,
                    inputs=[chunk],
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                
                # Build metadata
                metadata = {
                    "type": "message",
                    "session_id": session_id,
                    "user_id": user_id,
                    "org_id": org_id,
                    "role": role,
                    "content": chunk,
                    "timestamp": datetime.utcnow().isoformat(),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                
                if run_id:
                    metadata["run_id"] = run_id
                if workflow_name:
                    metadata["workflow_name"] = workflow_name
                
                # Generate vector ID
                vector_id = self._generate_id(chunk, metadata)
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding[0].values,
                    "metadata": metadata
                })
            
            # Upsert to Pinecone (namespace by org for isolation)
            namespace = f"org_{org_id}"
            self.index.upsert(vectors=vectors, namespace=namespace)
            
            logger.debug(f"✅ Embedded message: {len(chunks)} chunk(s) → Pinecone")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to embed message: {e}")
            return False
    
    def embed_and_upsert_file(
        self,
        session_id: str,
        file_id: str,
        filename: str,
        content: str,
        user_id: str,
        org_id: str,
        run_id: Optional[str] = None,
        mime_type: Optional[str] = None,
        classification: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Embed and store file content in Pinecone.        
        Args:
            session_id: Chat session ID
            file_id: Unique file ID
            filename: Original filename
            content: Extracted text content from file
            user_id: User ID
            org_id: Organization ID
            run_id: Workflow run ID (if file uploaded during workflow)
            mime_type: File MIME type
            classification: File classification dict (from FileIntelligence)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not content:
            return False
        
        try:
            # Chunk file content
            chunks = self._chunk_text(content)
            
            vectors = []
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.pc.inference.embed(
                    model=PINECONE_MODEL,
                    inputs=[chunk],
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                
                # Build metadata
                metadata = {
                    "type": "file",
                    "session_id": session_id,
                    "user_id": user_id,
                    "org_id": org_id,
                    "file_id": file_id,
                    "filename": filename,
                    "content_preview": chunk[:200],
                    "timestamp": datetime.utcnow().isoformat(),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                
                if run_id:
                    metadata["run_id"] = run_id
                if mime_type:
                    metadata["mime_type"] = mime_type
                if classification:
                    metadata["classification"] = str(classification.get("category", ""))
                    metadata["confidence"] = classification.get("confidence_score", "")
                
                # Generate vector ID
                vector_id = f"file_{file_id}_chunk_{i}"
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding[0].values,
                    "metadata": metadata
                })
            
            # Upsert to Pinecone
            namespace = f"org_{org_id}"
            self.index.upsert(vectors=vectors, namespace=namespace)
            
            logger.info(f"✅ Embedded file '{filename}': {len(chunks)} chunk(s) → Pinecone")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to embed file '{filename}': {e}")
            return False
    
    def search_context(
        self,
        query: str,
        org_id: str,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        top_k: int = 5,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant context using semantic similarity.
        
        Args:
            query: Search query (user message or workflow context)
            org_id: Organization ID (for namespace isolation)
            session_id: Optional session filter
            run_id: Optional run filter
            top_k: Number of results to return
            filter_type: Optional filter ('message' or 'file')
        
        Returns:
            List of matches with metadata and scores
        """
        if not self.enabled:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.pc.inference.embed(
                model=PINECONE_MODEL,
                inputs=[query],
                parameters={"input_type": "query", "truncate": "END"}
            )
            
            # Build filter
            filter_dict = {}
            if session_id:
                filter_dict["session_id"] = session_id
            if run_id:
                filter_dict["run_id"] = run_id
            if filter_type:
                filter_dict["type"] = filter_type
            
            # Search in Pinecone
            namespace = f"org_{org_id}"
            results = self.index.query(
                vector=query_embedding[0].values,
                top_k=top_k,
                namespace=namespace,
                filter=filter_dict if filter_dict else None,
                include_metadata=True,
            )
            
            matches = []
            for match in results.matches:
                matches.append({
                    "score": match.score,
                    "metadata": match.metadata,
                    "id": match.id,
                })
            
            logger.debug(f"🔍 Pinecone search: {len(matches)} results for query")
            return matches
        
        except Exception as e:
            logger.error(f"❌ Pinecone search failed: {e}")
            return []
    
    def delete_session_vectors(self, session_id: str, org_id: str) -> bool:
        """Delete all vectors for a session."""
        if not self.enabled:
            return False
        
        try:
            namespace = f"org_{org_id}"
            self.index.delete(
                filter={"session_id": session_id},
                namespace=namespace
            )
            logger.info(f"✅ Deleted vectors for session: {session_id[:8]}...")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to delete session vectors: {e}")
            return False


# Global singleton instance
pinecone_client = PineconeClient()

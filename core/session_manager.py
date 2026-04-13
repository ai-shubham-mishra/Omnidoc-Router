"""
Session manager for the LLM Router.
Handles chat session CRUD operations with Redis write-through cache.
Sessions stored in MongoDB (persistent) and Redis (fast access, 3-day TTL).

Pinecone Integration:
- Embeds chat messages for semantic search and long-term context
- Enables retrieval of relevant conversation history beyond Redis TTL
"""
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from pymongo import MongoClient
from dotenv import load_dotenv

from utils.redis_client import redis_client
from utils.config import SESSION_TTL_SECONDS
from components.PineconeClient import pinecone_client

load_dotenv()
logger = logging.getLogger(__name__)

from utils.db_config import (
    AZ_COSMOS_DB_URL, AZ_COSMOS_DB_NAME,
    CHAT_SESSIONS_COLLECTION,
)


class SessionManager:
    """Manages chat session state with Redis caching and Cosmos DB persistence."""

    def __init__(self):
        self.client = MongoClient(AZ_COSMOS_DB_URL)
        self.db = self.client[AZ_COSMOS_DB_NAME]
        self.collection = self.db[CHAT_SESSIONS_COLLECTION]
        self.redis = redis_client
        self._ensure_indexes()

    def _ensure_indexes(self):
        """Create indexes on first use."""
        try:
            self.collection.create_index("user_id")
            self.collection.create_index("org_id")
            self.collection.create_index("last_active")
        except Exception as e:
            logger.warning(f"Index creation skipped (may already exist): {e}")

    def create_session(
        self,
        session_id: str,
        user_id: str,
        org_id: str,
        jwt_token: str,
    ) -> Dict[str, Any]:
        """Create a new chat session with multi-workflow support."""
        now = datetime.utcnow()
        session = {
            "_id": session_id,
            "user_id": user_id,
            "org_id": org_id,
            "jwt_token": jwt_token,
            "status": "active",  # Never changes, session stays active
            "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_active": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "conversation_history": [],
            
            # Current workflow being processed (reset after completion)
            "current_workflow": {
                "workflow_id": None,
                "workflow_name": None,
                "workflow_endpoint": None,
                "status": "idle",  # idle | identifying | collecting | ready_to_execute | executing | awaiting_confirmation
                "required_inputs": [],
                "collected_inputs": {},
                "run_id": None,
                "confirmation_data": None,
            },
            
            # History of completed workflows in this session
            "workflow_history": [],
            
            # Uploaded files (persistent context, not auto-execute)
            "uploaded_files": [],
            
            # File IDs for file-based workflow communication
            "file_ids": [],
        }
        
        # Write to MongoDB (source of truth)
        self.collection.insert_one(session)
        logger.info(f"✅ Session created: {session_id[:8]}... (user: {user_id})")
        
        # Try to cache in Redis (non-blocking, graceful degradation)
        try:
            if self.redis.enabled:
                import asyncio
                asyncio.create_task(self.redis.set_session(session_id, session))
        except Exception as e:
            logger.debug(f"Redis cache create failed (non-critical): {e}")
        
        return session

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session with Redis cache (read-through pattern).
        Automatically refreshes TTL on access (sliding window).
        """
        # Try Redis first (fast path)
        cached = await self.redis.get_session(session_id)
        if cached:
            return cached
        
        # Cache miss: fetch from MongoDB (slow path)
        session = self.collection.find_one({"_id": session_id})
        if session:
            # Populate cache for next time
            await self.redis.set_session(session_id, session)
            logger.debug(f"📀 Loaded from MongoDB → Redis: {session_id[:8]}...")
        
        return session

    def get_session_sync(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous get session (MongoDB only, for non-async contexts)."""
        return self.collection.find_one({"_id": session_id})

    async def add_message(self, session_id: str, role: str, content: str):
        """Add message to conversation history and update cache."""
        now = datetime.utcnow()
        session = await self.get_session(session_id)
        turn = len(session.get("conversation_history", [])) + 1 if session else 1

        # Update MongoDB
        self.collection.update_one(
            {"_id": session_id},
            {
                "$push": {
                    "conversation_history": {
                        "turn": turn,
                        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "role": role,
                        "content": content,
                    }
                },
                "$set": {"last_active": now.strftime("%Y-%m-%dT%H:%M:%SZ")},
            },
        )
        
        # Update Redis cache (graceful degradation if Redis unavailable)
        try:
            updated_session = self.collection.find_one({"_id": session_id})
            if updated_session:
                await self.redis.set_session(session_id, updated_session)
        except Exception as e:
            logger.debug(f"Redis cache update failed (non-critical): {e}")
        
        # Embed message in Pinecone for semantic search
        if session:
            user_id = session.get("user_id", "")
            org_id = session.get("org_id", "")
            current_wf = session.get("current_workflow", {})
            run_id = current_wf.get("run_id")
            workflow_name = current_wf.get("workflow_name")
            
            pinecone_client.embed_and_upsert_message(
                session_id=session_id,
                message=content,
                role=role,
                user_id=user_id,
                org_id=org_id,
                run_id=run_id,
                workflow_name=workflow_name,
            )

    async def set_workflow_context(
        self,
        session_id: str,
        workflow: Dict[str, Any],
        required_inputs: List[Dict[str, Any]],
    ):
        """Set current workflow being processed."""
        wf_summary = {
            "workflow_id": workflow.get("workflowId", ""),
            "workflow_name": workflow.get("workflowName", ""),
            "workflow_endpoint": workflow.get("workflowEndpoint", ""),
            "workflow_schema": workflow.get("workflowSchema", {}),
            "workflow_api_calls": workflow.get("workflowApiCalls", {}),
            "status": "collecting",
            "required_inputs": required_inputs,
            "collected_inputs": {},
            "run_id": None,
            "confirmation_data": None,
        }

        # Update MongoDB
        self.collection.update_one(
            {"_id": session_id},
            {
                "$set": {
                    "current_workflow": wf_summary,
                     "last_active": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            },
        )
        
        # Update Redis cache
        updated_session = self.collection.find_one({"_id": session_id})
        if updated_session:
            await self.redis.set_session(session_id, updated_session)

    async def mark_input_collected(self, session_id: str, field_name: str, value: Any):
        """Mark a specific input as collected and store its value."""
        session = await self.get_session(session_id)
        if not session:
            return

        current_wf = session.get("current_workflow", {})
        inputs = current_wf.get("required_inputs", [])
        
        for inp in inputs:
            if inp["field"] == field_name:
                inp["collected"] = True
                inp["value"] = value

        collected = current_wf.get("collected_inputs", {})
        collected[field_name] = value

        # Update MongoDB
        self.collection.update_one(
            {"_id": session_id},
            {
                "$set": {
                    "current_workflow.required_inputs": inputs,
                    "current_workflow.collected_inputs": collected,
                    "last_active": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            },
        )
        
        # Update Redis cache
        updated_session = self.collection.find_one({"_id": session_id})
        if updated_session:
            await self.redis.set_session(session_id, updated_session)

    async def add_files(self, session_id: str, files_info: List[Dict[str, Any]]):
        """
        Add uploaded files to session context (NOT auto-execute).
        Files become available for future workflow executions.
        """
        # Update MongoDB
        self.collection.update_one(
            {"_id": session_id},
            {
                "$push": {"uploaded_files": {"$each": files_info}},
                "$set": {"last_active": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
            },
        )
        
        # Update Redis cache
        updated_session = self.collection.find_one({"_id": session_id})
        if updated_session:
            await self.redis.set_session(session_id, updated_session)
        
        logger.info(f"📎 {len(files_info)} file(s) added to context: {session_id[:8]}...")

    async def add_file_ids(self, session_id: str, file_ids: List[str]):
        """
        Add file IDs to session context for file-based workflow communication.
        File IDs are lightweight references to files stored in blob/DB.
        """
        if not file_ids:
            return
        
        # Update MongoDB
        self.collection.update_one(
            {"_id": session_id},
            {
                "$addToSet": {"file_ids": {"$each": file_ids}},
                "$set": {"last_active": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
            },
        )
        
        # Update Redis cache
        updated_session = self.collection.find_one({"_id": session_id})
        if updated_session:
            await self.redis.set_session(session_id, updated_session)
        
        logger.info(f"🔗 {len(file_ids)} file_id(s) added to session: {session_id[:8]}...")

    async def update_files(self, session_id: str, updated_files: List[Dict[str, Any]]):
        """
        Replace all uploaded_files with updated list (e.g., after marking files as used).
        """
        self.collection.update_one(
            {"_id": session_id},
            {
                "$set": {
                    "uploaded_files": updated_files,
                    "last_active": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            },
        )
        
        # Update Redis cache
        updated_session = self.collection.find_one({"_id": session_id})
        if updated_session:
            await self.redis.set_session(session_id, updated_session)

    async def set_run_id(self, session_id: str, run_id: str):
        """Set runId for current workflow execution."""
        # Update MongoDB
        self.collection.update_one(
            {"_id": session_id},
            {
                "$set": {
                    "current_workflow.run_id": run_id,
                    "last_active": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            },
        )
        
        # Update Redis cache
        updated_session = self.collection.find_one({"_id": session_id})
        if updated_session:
            await self.redis.set_session(session_id, updated_session)

    async def update_workflow_status(
        self,
        session_id: str,
        status: str,
        confirmation_data: Optional[Dict] = None,
        file_ids_in_use: Optional[List[str]] = None,
    ):
        """Update current workflow status and track file_ids being used."""
        update_doc: Dict[str, Any] = {
            "current_workflow.status": status,
            "last_active": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        
        if confirmation_data is not None:
            update_doc["current_workflow.confirmation_data"] = confirmation_data
        
        if file_ids_in_use is not None:
            update_doc["current_workflow.file_ids_in_use"] = file_ids_in_use

        # Update MongoDB
        self.collection.update_one({"_id": session_id}, {"$set": update_doc})
        
        # Update Redis cache
        updated_session = self.collection.find_one({"_id": session_id})
        if updated_session:
            await self.redis.set_session(session_id, updated_session)

    async def complete_workflow(
        self,
        session_id: str,
        result: Any,
        status: str = "completed",
        result_summary: Optional[str] = None,
    ):
        """
        Mark current workflow as complete and move to history.
        Resets current_workflow to idle state for next workflow.
        
        Args:
            session_id: Session ID
            result: Raw workflow result
            status: Completion status (completed/failed/cancelled)
            result_summary: Optional human-readable summary of key result data
        """
        session = await self.get_session(session_id)
        if not session:
            return
        
        current_wf = session.get("current_workflow", {})
        
        # Add to workflow history with file_ids and inputs for rerun capability
        history_entry = {
            "workflow_id": current_wf.get("workflow_id"),
            "workflow_name": current_wf.get("workflow_name"),
            "workflow_endpoint": current_wf.get("workflow_endpoint"),
            "run_id": current_wf.get("run_id"),
            "status": status,
            "result": result,
            "completed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "file_ids_used": current_wf.get("file_ids_in_use", []),
            "collected_inputs": current_wf.get("collected_inputs", {}),
        }
        
        if result_summary:
            history_entry["result_summary"] = result_summary
        
        # Reset current_workflow to idle
        idle_workflow = {
            "workflow_id": None,
            "workflow_name": None,
            "workflow_endpoint": None,
            "status": "idle",
            "required_inputs": [],
            "collected_inputs": {},
            "run_id": None,
            "confirmation_data": None,
        }
        
        # Update MongoDB
        self.collection.update_one(
            {"_id": session_id},
            {
                "$push": {"workflow_history": history_entry},
                "$set": {
                    "current_workflow": idle_workflow,
                    "last_active": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            },
        )
        
        # Update Redis cache
        updated_session = self.collection.find_one({"_id": session_id})
        if updated_session:
            await self.redis.set_session(session_id, updated_session)
        
        logger.info(f"✅ Workflow completed: {current_wf.get('workflow_name')} → History")

    async def update_jwt_token(self, session_id: str, jwt_token: str):
        """Update JWT token in session (handles token refresh)."""
        # Update MongoDB
        self.collection.update_one(
            {"_id": session_id},
            {
                "$set": {
                    "jwt_token": jwt_token,
                    "last_active": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            },
        )
        
        # Update Redis cache
        updated_session = self.collection.find_one({"_id": session_id})
        if updated_session:
            await self.redis.set_session(session_id, updated_session)

    async def delete_session(self, session_id: str, user_id: str) -> bool:
        """Delete a session from both MongoDB and Redis."""
        # Verify ownership
        session = await self.get_session(session_id)
        if not session or session.get("user_id") != user_id:
            return False
        
        # Delete from MongoDB
        self.collection.delete_one({"_id": session_id})
        
        # Delete from Redis
        await self.redis.delete_session(session_id)
        
        logger.info(f"🗑️ Session deleted: {session_id[:8]}...")
        return True

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get full session info (for GET /session endpoint)."""
        session = self.collection.find_one({"_id": session_id})
        if not session:
            return None
        
        # Format for API response
        return {
            "session_id": session["_id"],
            "user_id": session["user_id"],
            "org_id": session["org_id"],
            "status": session["status"],
            "created_at": session["created_at"],
            "last_active": session["last_active"],
            "conversation_history": session["conversation_history"],
            "current_workflow": session["current_workflow"],
            "workflow_history": session.get("workflow_history", []),
            "uploaded_files": [
                {
                    "file_id": f.get("file_id"),
                    "original_name": f.get("original_name"),
                    "uploaded_at": f.get("uploaded_at"),
                    "mime_type": f.get("mime_type"),
                }
                for f in session.get("uploaded_files", [])
            ],
        }

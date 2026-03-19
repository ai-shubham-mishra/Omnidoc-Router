"""
Session manager for the LLM Router.
Handles chat session CRUD operations in MongoDB.
Sessions auto-expire after 24 hours via TTL index.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
CHAT_SESSIONS_COLLECTION = "chat_sessions"


class SessionManager:
    """Manages chat session state in MongoDB."""

    def __init__(self):
        self.client = MongoClient(MONGO_DB_URL)
        self.db = self.client[MONGO_DB_NAME]
        self.collection = self.db[CHAT_SESSIONS_COLLECTION]
        self._ensure_indexes()

    def _ensure_indexes(self):
        """Create indexes on first use."""
        try:
            self.collection.create_index("ttl", expireAfterSeconds=0)
            self.collection.create_index("user_id")
            self.collection.create_index("org_id")
        except Exception as e:
            logger.warning(f"Index creation skipped (may already exist): {e}")

    def create_session(
        self,
        session_id: str,
        user_id: str,
        org_id: str,
        jwt_token: str,
    ) -> Dict[str, Any]:
        """Create a new chat session."""
        now = datetime.utcnow()
        session = {
            "_id": session_id,
            "user_id": user_id,
            "org_id": org_id,
            "jwt_token": jwt_token,
            "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ttl": now + timedelta(hours=24),
            "conversation_history": [],
            "workflow_context": {},
            "execution_context": {
                "runId": None,
                "workflowId": None,
                "status": "new",
                "current_step": None,
                "confirmation_data": None,
                "final_result": None,
            },
            "files_uploaded": [],
        }
        self.collection.insert_one(session)
        return session

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session by ID."""
        return self.collection.find_one({"_id": session_id})

    def add_message(self, session_id: str, role: str, content: str):
        """Add message to conversation history."""
        now = datetime.utcnow()
        session = self.collection.find_one({"_id": session_id}, {"conversation_history": 1})
        turn = len(session.get("conversation_history", [])) + 1 if session else 1

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
                "$set": {"updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")},
            },
        )

    def set_workflow_context(
        self,
        session_id: str,
        workflow: Dict[str, Any],
        required_inputs: List[Dict[str, Any]],
    ):
        """Set identified workflow and required inputs."""
        wf_summary = {
            "workflowId": workflow.get("workflowId", ""),
            "workflowName": workflow.get("workflowName", ""),
            "workflowEndpoint": workflow.get("workflowEndpoint", ""),
            "workflowSchema": workflow.get("workflowSchema", {}),
            "workflowApiCalls": workflow.get("workflowApiCalls", {}),
        }

        self.collection.update_one(
            {"_id": session_id},
            {
                "$set": {
                    "workflow_context.identified_workflow": wf_summary,
                    "workflow_context.required_inputs": required_inputs,
                    "workflow_context.collected_inputs": {},
                    "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            },
        )

    def mark_input_collected(self, session_id: str, field_name: str, value: Any):
        """Mark a specific input as collected and store its value."""
        session = self.get_session(session_id)
        if not session:
            return

        inputs = session.get("workflow_context", {}).get("required_inputs", [])
        for inp in inputs:
            if inp["field"] == field_name:
                inp["collected"] = True
                inp["value"] = value

        collected = session.get("workflow_context", {}).get("collected_inputs", {})
        collected[field_name] = value

        self.collection.update_one(
            {"_id": session_id},
            {
                "$set": {
                    "workflow_context.required_inputs": inputs,
                    "workflow_context.collected_inputs": collected,
                    "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            },
        )

    def add_files(self, session_id: str, files_info: List[Dict[str, Any]], field_name: str):
        """Add uploaded file records to session and mark the input collected."""
        self.collection.update_one(
            {"_id": session_id},
            {
                "$push": {"files_uploaded": {"$each": files_info}},
                "$set": {"updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
            },
        )
        self.mark_input_collected(session_id, field_name, [f["stored_path"] for f in files_info])

    def set_run_id(self, session_id: str, run_id: str):
        """Set runId for workflow execution."""
        self.collection.update_one(
            {"_id": session_id},
            {
                "$set": {
                    "execution_context.runId": run_id,
                    "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            },
        )

    def update_execution_status(
        self,
        session_id: str,
        status: str,
        workflowId: Optional[str] = None,
        step_number: Optional[int] = None,
        confirmation_data: Optional[Dict] = None,
        final_result: Optional[Any] = None,
    ):
        """Update workflow execution status."""
        update_doc: Dict[str, Any] = {
            "execution_context.status": status,
            "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if workflowId is not None:
            update_doc["execution_context.workflowId"] = workflowId
        if step_number is not None:
            update_doc["execution_context.current_step"] = step_number
        if confirmation_data is not None:
            update_doc["execution_context.confirmation_data"] = confirmation_data
        if final_result is not None:
            update_doc["execution_context.final_result"] = final_result

        self.collection.update_one({"_id": session_id}, {"$set": update_doc})

    def update_jwt_token(self, session_id: str, jwt_token: str):
        """Update JWT token in session (handles token refresh)."""
        self.collection.update_one(
            {"_id": session_id},
            {
                "$set": {
                    "jwt_token": jwt_token,
                    "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            },
        )

    def delete_session(self, session_id: str):
        """Delete a session."""
        self.collection.delete_one({"_id": session_id})

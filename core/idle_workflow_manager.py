"""
Idle Workflow Manager - Handles workflows awaiting files or mid-execution pauses.

Manages workflows in partial/waiting state with:
- Partial input collection tracking
- Missing input specifications with file signatures
- Workflow completeness checking
- Automatic cleanup via TTL
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pymongo import MongoClient

from dotenv import load_dotenv
from utils.db_config import AZ_COSMOS_DB_URL, AZ_COSMOS_DB_NAME

load_dotenv()
logger = logging.getLogger(__name__)


class IdleWorkflowManager:
    """Manages workflows in idle/waiting state with persistent status-based lifecycle."""
    
    COLLECTION_NAME = "idle_workflows"
    MAX_IDLE_WORKFLOWS_PER_SESSION = 10
    DEFAULT_TTL_DAYS = 7
    ACTIVE_TTL_DAYS = 30  # Active workflows get longer TTL
    
    def __init__(self):
        self.client = MongoClient(AZ_COSMOS_DB_URL)
        self.db = self.client[AZ_COSMOS_DB_NAME]
        self.collection = self.db[self.COLLECTION_NAME]
        self._ensure_indexes()
    
    def _ensure_indexes(self):
        """Create necessary indexes for efficient querying."""
        try:
            self.collection.create_index("session_id")
            self.collection.create_index([("session_id", 1), ("status", 1)])
            self.collection.create_index([("user_id", 1), ("status", 1)])
            self.collection.create_index([("workflow_id", 1), ("session_id", 1)])
            self.collection.create_index("expires_at")  # TTL index
            logger.info("✅ Idle workflow indexes created")
        except Exception as e:
            logger.warning(f"Index creation skipped (may already exist): {e}")
    
    async def create_idle_workflow(
        self,
        workflow_id: str,
        workflow_name: str,
        workflow_endpoint: str,
        user_id: str,
        org_id: str,
        session_id: str,
        collected_inputs: Dict[str, Any],
        missing_inputs: List[Dict[str, Any]],
        idle_source: str = "partial_collection",
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new idle workflow entry.
        
        Args:
            workflow_id: Registry workflow ID
            workflow_name: Human-readable name
            workflow_endpoint: API endpoint
            user_id, org_id, session_id: Context
            collected_inputs: What we have so far
            missing_inputs: What we're waiting for (with file_signatures)
            idle_source: "partial_collection" | "mid_execution_discovery"
            execution_context: For mid-execution resumes (run_id, state, etc.)
        
        Returns:
            instance_id: Unique identifier for this idle workflow instance
        """
        # Check if this workflow already exists in idle state (from previous switch)
        existing = self.collection.find_one({
            "workflow_id": workflow_id,
            "session_id": session_id,
            "status": {"$in": ["idle_awaiting_files", "idle_mid_execution", "idle_awaiting_confirmation"]}
        })
        
        if existing:
            # Update existing instead of creating new
            instance_id = existing["instance_id"]
            logger.info(f"🔄 Updating existing idle workflow: {instance_id[:8]}... ({workflow_name})")
            
            self.collection.update_one(
                {"instance_id": instance_id},
                {
                    "$set": {
                        "collected_inputs": collected_inputs,
                        "missing_inputs": missing_inputs,
                        "execution_context": execution_context or {},
                        "status": "idle_mid_execution" if idle_source == "mid_execution_discovery" else "idle_awaiting_files",
                        "idle_source": idle_source,
                        "last_active": datetime.utcnow().isoformat(),
                    }
                }
            )
            return instance_id
        
        # Check session limit for new workflows
        session_count = self.collection.count_documents({
            "session_id": session_id,
            "status": {"$in": ["idle_awaiting_files", "idle_mid_execution", "idle_awaiting_confirmation"]}
        })
        
        if session_count >= self.MAX_IDLE_WORKFLOWS_PER_SESSION:
            raise ValueError(
                f"Session has reached maximum idle workflows limit ({self.MAX_IDLE_WORKFLOWS_PER_SESSION}). "
                "Complete or cancel existing workflows first."
            )
        
        import uuid
        instance_id = str(uuid.uuid4())
        
        now = datetime.utcnow()
        expires_at = now + timedelta(days=self.DEFAULT_TTL_DAYS)
        
        idle_workflow = {
            "_id": instance_id,
            "instance_id": instance_id,
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "workflow_endpoint": workflow_endpoint,
            "user_id": user_id,
            "org_id": org_id,
            "session_id": session_id,
            
            "status": "idle_mid_execution" if idle_source == "mid_execution_discovery" else "idle_awaiting_files",
            "idle_source": idle_source,
            
            "collected_inputs": collected_inputs,
            "missing_inputs": missing_inputs,
            
            "execution_context": execution_context or {},
            
            "created_at": now.isoformat(),
            "last_active": now.isoformat(),
            "expires_at": expires_at.isoformat(),
        }
        
        self.collection.insert_one(idle_workflow)
        logger.info(f"✅ Idle workflow created: {instance_id[:8]}... ({workflow_name})")
        
        return instance_id
    
    async def get_idle_workflows(
        self,
        session_id: str,
        include_expired: bool = False,
        include_active: bool = False,
        include_completed: bool = False
    ) -> List[Dict[str, Any]]:
        """Get idle workflows for a session, filtering by status.
        
        Args:
            session_id: Session identifier
            include_expired: Include expired workflows
            include_active: Include workflows with status='active'
            include_completed: Include workflows with status='completed'
        
        Returns:
            List of idle workflow documents
        """
        query = {"session_id": session_id}
        
        # Filter by status - exclude active and completed by default
        status_filter = ["idle_awaiting_files", "idle_mid_execution", "idle_awaiting_confirmation"]
        if include_active:
            status_filter.append("active")
        if include_completed:
            status_filter.append("completed")
        
        query["status"] = {"$in": status_filter}
        
        if not include_expired:
            query["expires_at"] = {"$gt": datetime.utcnow().isoformat()}
        
        workflows = list(self.collection.find(query))
        
        # Remove MongoDB _id for cleaner response
        for wf in workflows:
            wf.pop("_id", None)
        
        return workflows
    
    async def get_idle_workflow(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get specific idle workflow by instance ID."""
        workflow = self.collection.find_one({"instance_id": instance_id})
        if workflow:
            workflow.pop("_id", None)
        return workflow
    
    async def find_idle_workflow_by_workflow_id(
        self, 
        session_id: str, 
        workflow_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find idle workflow for a session by workflow_id.
        Used during workflow switching to check if target workflow exists in idle state.
        
        Args:
            session_id: Session identifier
            workflow_id: Workflow registry ID
        
        Returns:
            Idle workflow document if found, None otherwise
        """
        workflow = self.collection.find_one({
            "session_id": session_id,
            "workflow_id": workflow_id,
            "status": {"$in": ["idle_awaiting_files", "idle_awaiting_confirmation", "idle_mid_execution"]}
        })
        if workflow:
            workflow.pop("_id", None)
            logger.info(f"📍 Found existing idle workflow: {workflow['workflow_name']} (instance: {workflow['instance_id'][:8]}...)")
        return workflow
    
    async def add_file_to_idle_workflow(
        self,
        instance_id: str,
        input_field: str,
        file_id: str
    ) -> bool:
        """
        Add a classified file to an idle workflow's collected inputs.
        Updates missing_inputs to reflect collection.
        
        Returns:
            True if successful, False otherwise
        """
        workflow = await self.get_idle_workflow(instance_id)
        if not workflow:
            logger.error(f"Idle workflow {instance_id} not found")
            return False
        
        # Update collected_inputs
        collected = workflow.get("collected_inputs", {})
        
        if input_field in collected:
            # File-type input - ensure it's a list
            if not isinstance(collected[input_field], list):
                collected[input_field] = [collected[input_field]]
            collected[input_field].append(file_id)
        else:
            collected[input_field] = [file_id]
        
        # Update missing_inputs - increment already_collected or remove if complete
        missing = workflow.get("missing_inputs", [])
        updated_missing = []
        
        for missing_input in missing:
            if missing_input["field"] == input_field:
                input_size = missing_input.get("inputSize", 1)
                already_collected = missing_input.get("already_collected", 0) + 1
                
                # Check if this input is now complete
                if isinstance(input_size, int) and already_collected >= input_size:
                    # Complete - don't add to updated_missing
                    logger.info(f"✅ Input {input_field} now complete for workflow {instance_id[:8]}...")
                    continue
                else:
                    # Update count
                    missing_input["already_collected"] = already_collected
                    updated_missing.append(missing_input)
            else:
                updated_missing.append(missing_input)
        
        # Update database
        result = self.collection.update_one(
            {"instance_id": instance_id},
            {
                "$set": {
                    "collected_inputs": collected,
                    "missing_inputs": updated_missing,
                    "last_active": datetime.utcnow().isoformat()
                }
            }
        )
        
        return result.modified_count > 0
    
    async def check_if_complete(self, instance_id: str) -> Dict[str, Any]:
        """
        Check if idle workflow has all required inputs collected.
        
        Returns:
            {
                "is_complete": bool,
                "missing_required": List[str],
                "missing_optional": List[str]
            }
        """
        workflow = await self.get_idle_workflow(instance_id)
        if not workflow:
            return {"is_complete": False, "missing_required": [], "missing_optional": []}
        
        missing_inputs = workflow.get("missing_inputs", [])
        
        missing_required = []
        missing_optional = []
        
        for missing_input in missing_inputs:
            field = missing_input["field"]
            priority = missing_input.get("priority", "required")
            
            if priority == "required":
                missing_required.append(field)
            else:
                missing_optional.append(field)
        
        is_complete = len(missing_required) == 0
        
        return {
            "is_complete": is_complete,
            "missing_required": missing_required,
            "missing_optional": missing_optional
        }
    
    async def update_status(self, instance_id: str, new_status: str) -> bool:
        """Update workflow status (e.g., idle → ready_to_execute)."""
        result = self.collection.update_one(
            {"instance_id": instance_id},
            {
                "$set": {
                    "status": new_status,
                    "last_active": datetime.utcnow().isoformat()
                }
            }
        )
        return result.modified_count > 0
    
    async def update_missing_inputs(
        self,
        instance_id: str,
        new_missing_inputs: List[Dict[str, Any]]
    ) -> bool:
        """
        Update missing_inputs (for mid-execution discovery).
        Used when workflow discovers it needs more files during execution.
        """
        result = self.collection.update_one(
            {"instance_id": instance_id},
            {
                "$set": {
                    "missing_inputs": new_missing_inputs,
                    "last_active": datetime.utcnow().isoformat()
                }
            }
        )
        return result.modified_count > 0
    
    async def activate_workflow(self, instance_id: str) -> bool:
        """Mark workflow as active (currently being worked on).
        
        Active workflows get extended TTL and are excluded from idle queries.
        This prevents accidental deletion while workflow is in progress.
        """
        now = datetime.utcnow()
        new_expires_at = now + timedelta(days=self.ACTIVE_TTL_DAYS)
        
        result = self.collection.update_one(
            {"instance_id": instance_id},
            {
                "$set": {
                    "status": "active",
                    "last_active": now.isoformat(),
                    "expires_at": new_expires_at.isoformat(),
                    "activated_at": now.isoformat()
                }
            }
        )
        
        if result.modified_count > 0:
            logger.info(f"▶️ Workflow activated: {instance_id[:8]}... (TTL extended to {self.ACTIVE_TTL_DAYS} days)")
        return result.modified_count > 0
    
    async def deactivate_workflow(
        self,
        instance_id: str,
        new_status: str = "idle_awaiting_files"
    ) -> bool:
        """Mark workflow as idle (paused/switched away from).
        
        Args:
            instance_id: Workflow instance identifier
            new_status: Status to set (idle_awaiting_files, idle_awaiting_confirmation, etc.)
        
        Returns:
            True if successfully deactivated
        """
        result = self.collection.update_one(
            {"instance_id": instance_id},
            {
                "$set": {
                    "status": new_status,
                    "last_active": datetime.utcnow().isoformat()
                },
                "$unset": {"activated_at": ""}
            }
        )
        
        if result.modified_count > 0:
            logger.info(f"⏸️ Workflow deactivated: {instance_id[:8]}... (status: {new_status})")
        return result.modified_count > 0
    
    async def delete_idle_workflow(self, instance_id: str) -> bool:
        """Permanently delete an idle workflow.
        
        Only use this for:
        - Workflow completion (success/failure)
        - Explicit user cancellation
        - TTL expiration cleanup
        
        Do NOT use for workflow switching - use deactivate_workflow() instead.
        """
        result = self.collection.delete_one({"instance_id": instance_id})
        if result.deleted_count > 0:
            logger.info(f"🗑️ Idle workflow deleted: {instance_id[:8]}...")
        return result.deleted_count > 0
    
    async def delete_active_workflows(self, session_id: str) -> int:
        """Delete all active workflows for a session.
        
        Called when a workflow completes - cleans up the active workflow.
        
        Returns:
            Count of deleted workflows
        """
        result = self.collection.delete_many({
            "session_id": session_id,
            "status": "active"
        })
        
        if result.deleted_count > 0:
            logger.info(f"🗑️ Deleted {result.deleted_count} active workflow(s) for session {session_id[:8]}...")
        return result.deleted_count
    
    async def extend_ttl(self, instance_id: str, days: int = None) -> bool:
        """Extend TTL for a workflow (useful for long-running idle workflows)."""
        if days is None:
            days = self.DEFAULT_TTL_DAYS
        
        new_expires_at = datetime.utcnow() + timedelta(days=days)
        
        result = self.collection.update_one(
            {"instance_id": instance_id},
            {
                "$set": {
                    "expires_at": new_expires_at.isoformat(),
                    "last_active": datetime.utcnow().isoformat()
                }
            }
        )
        return result.modified_count > 0
    
    async def cleanup_expired(self) -> int:
        """
        Remove expired idle workflows (TTL cleanup).
        Returns count of deleted workflows.
        """
        result = self.collection.delete_many({
            "expires_at": {"$lt": datetime.utcnow().isoformat()}
        })
        
        if result.deleted_count > 0:
            logger.info(f"🧹 Cleaned up {result.deleted_count} expired idle workflows")
        
        return result.deleted_count
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get lightweight summary of idle workflows for a session.
        Used for session state and router responses.
        """
        workflows = await self.get_idle_workflows(session_id)
        
        return [
            {
                "instance_id": wf["instance_id"],
                "workflow_name": wf["workflow_name"],
                "status": wf["status"],
                "missing_count": len(wf.get("missing_inputs", [])),
                "last_active": wf["last_active"],
                "idle_source": wf.get("idle_source", "partial_collection")
            }
            for wf in workflows
        ]

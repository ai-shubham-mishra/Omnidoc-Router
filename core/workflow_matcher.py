"""
Workflow Matcher for the LLM Router.
Matches user intent to registered workflows using the existing
MongoDB workflow registry. Zero changes to the registry schema.
"""
import os
import logging
from typing import Optional, Dict, Any, List

from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

from utils.db_config import (
    AZ_COSMOS_DB_URL, AZ_COSMOS_DB_NAME,
    REGISTERED_WORKFLOWS_COLLECTION,
)


class WorkflowMatcher:
    """Match user messages to workflows using the existing workflow registry."""

    def __init__(self):
        self.client = MongoClient(AZ_COSMOS_DB_URL)
        self.db = self.client[AZ_COSMOS_DB_NAME]
        self.collection = self.db[REGISTERED_WORKFLOWS_COLLECTION]

    def get_all_workflows(self, org_id: str) -> List[Dict[str, Any]]:
        """Get all workflows accessible to user's organization."""
        workflows = list(
            self.collection.find(
                {
                    "$or": [
                        {"workflowType": "public"},
                        {"idOrg": org_id},              # Legacy: string match
                        {"idOrg": {"$in": [org_id]}},   # New: array contains org_id
                        {"idClient": org_id},            # Legacy: string match
                        {"idClient": {"$in": [org_id]}}, # New: array contains org_id
                    ]
                }
            )
        )
        for wf in workflows:
            wf.pop("_id", None)
        return workflows

    def get_workflow_summaries(self, org_id: str) -> List[Dict[str, Any]]:
        """Get lightweight workflow info for LLM context."""
        workflows = self.get_all_workflows(org_id)
        return [
            {
                "id": wf.get("workflowId", ""),
                "name": wf.get("workflowName", ""),
                "description": wf.get("workflowDescription", ""),
                "tags": wf.get("workflowTags", []),
                "service_type": wf.get("serviceType", ""),
            }
            for wf in workflows
        ]

    def match_by_keywords(
        self,
        message: str,
        org_id: str,
        threshold: float = 2.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Fast keyword-based matching using existing workflowTags,
        workflowName, and workflowDescription.
        """
        message_lower = message.lower()
        workflows = self.get_all_workflows(org_id)

        best_match = None
        best_score = 0.0

        for wf in workflows:
            score = 0.0

            for tag in wf.get("workflowTags", []):
                if tag.lower() in message_lower:
                    score += 3.0

            name_words = wf.get("workflowName", "").lower().split()
            for word in name_words:
                if len(word) > 3 and word in message_lower:
                    score += 2.0

            desc_words = set(wf.get("workflowDescription", "").lower().split())
            msg_words = set(message_lower.split())
            matching_desc = msg_words & desc_words
            meaningful = [w for w in matching_desc if len(w) > 3]
            score += len(meaningful) * 0.5

            if score > best_score:
                best_score = score
                best_match = wf

        return best_match if best_score >= threshold else None

    def get_workflow_by_id(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow by workflowId."""
        wf = self.collection.find_one({"workflowId": workflow_id})
        if wf:
            wf.pop("_id", None)
        return wf

    def get_workflow_by_endpoint(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Get workflow by endpoint."""
        wf = self.collection.find_one({"workflowEndpoint": endpoint})
        if wf:
            wf.pop("_id", None)
        return wf

    def has_hitl(self, workflow: Dict[str, Any]) -> bool:
        """Check if workflow has human-in-the-loop (call1 exists in schema)."""
        schema = workflow.get("workflowSchema", {})
        return "call1" in schema

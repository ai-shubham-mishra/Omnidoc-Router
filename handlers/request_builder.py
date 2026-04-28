"""
Request Builder for the LLM Router.
Builds API requests (JSON or form-data) for workflow endpoints
based on the workflowSchema body_type.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class RequestBuilder:
    """Build workflow API requests (JSON or form-data) from collected inputs."""

    def build_workflow_request(
        self,
        workflow: Dict[str, Any],
        collected_inputs: Dict[str, Any],
        required_inputs: list,  # Added to get endpoint_field_name mapping
        run_id: str,
        jwt_token: str,
        session_id: str = None,
    ) -> Dict[str, Any]:
        """
        Build initial workflow request (call0) based on body_type.

        Returns:
            {
                "endpoint": "/private-po-registration",
                "content_type": "application/json" | "multipart/form-data",
                "data": {...},
                "headers": {"Authorization": "Bearer ...", "X-Session-Id": "..."},
                "has_files": bool,
                "file_fields": {field_name: {"paths": [file_paths], "endpoint_name": "files"}}
            }
        """
        schema = workflow.get("workflowSchema", {})
        call0 = schema.get("call0", {})
        body_type = call0.get("body_type", "json")
        endpoint = workflow.get("workflowEndpoint", "")

        # Build field name mapping from required_inputs
        # Maps router field names (Input0, Input1) to API parameter names (files, po_number, etc.)
        field_mapping = {}
        for inp in required_inputs:
            router_field = inp["field"]
            api_field = inp.get("endpoint_field_name", router_field)  # Default to field if not specified
            field_mapping[router_field] = api_field

        # Build set of file-type fields from schema
        file_type_fields = {inp["field"] for inp in required_inputs if inp.get("type") == "file"}

        headers = {"Authorization": jwt_token}
        
        # Add session_id to headers if provided
        if session_id:
            headers["X-Session-Id"] = session_id

        file_fields = {}
        data_fields = {}

        # Map collected inputs to API parameter names
        for router_field, value in collected_inputs.items():
            # Special handling for auto-filled values
            if value == "__auto_runid__":
                data_fields["runId"] = run_id
                continue
            if value == "__hitl_field__":
                continue

            # Get the API parameter name for this field
            api_field = field_mapping.get(router_field, router_field)

            # Skip file-type fields - they're handled by orchestrator's file_id conversion
            if router_field in file_type_fields:
                continue
            
            # Non-file input: use API parameter name directly
            data_fields[api_field] = value

        if "runId" not in data_fields and "runid" not in data_fields:
            data_fields["runId"] = run_id

        if body_type == "form" or file_fields:
            return {
                "endpoint": endpoint,
                "content_type": "multipart/form-data",
                "data": data_fields,
                "headers": headers,
                "has_files": bool(file_fields),
                "file_fields": file_fields,
            }
        else:
            headers["Content-Type"] = "application/json"
            return {
                "endpoint": endpoint,
                "content_type": "application/json",
                "data": data_fields,
                "headers": headers,
                "has_files": False,
                "file_fields": {},
            }

    def build_confirmation_request(
        self,
        workflow: Dict[str, Any],
        run_id: str,
        workflow_id: str,
        is_confirmed: bool,
        jwt_token: str,
        session_id: str = None,
        hitl_request: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build HITL confirmation request with modified hitl_request from frontend.
        Router provides runId from session, frontend provides edited hitl_request.
        """
        endpoint = workflow.get("workflowEndpoint", "")
        
        # Build payload with core fields
        payload = {
            "is_confirmed_by_user": is_confirmed,
            "runId": run_id,  # Router-generated, from session
        }
        
        # Add modified hitl_request if provided by frontend
        if hitl_request:
            payload["hitl_request"] = hitl_request
        
        logger.info(f"🔧 HITL confirmation request: runId={run_id}, has_hitl_request={bool(hitl_request)}")
        
        headers = {
            "Authorization": jwt_token,
            "Content-Type": "application/json",
        }
        
        if session_id:
            headers["X-Session-Id"] = session_id
        
        return {
            "endpoint": endpoint,
            "content_type": "application/json",
            "data": payload,
            "headers": headers,
        }

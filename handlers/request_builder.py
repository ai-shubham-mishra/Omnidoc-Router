"""
Request Builder for the LLM Router.
Builds API requests (JSON or form-data) for workflow endpoints
based on the workflowSchema body_type.
"""
import logging
from typing import Dict, Any

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
        confirmation_data: Dict[str, Any],
        jwt_token: str,
        session_id: str = None,
    ) -> Dict[str, Any]:
        """
        Build HITL confirmation request (call1) using workflow schema.
        
        Generic implementation:
        - Reads call1 schema from workflow to get field mappings
        - Extracts fields from confirmation_data['body']
        - Maps internal field names to API parameter names via endpoint_field_name
        - Always includes: is_confirmed_by_user, runId, workflowId
        
        Always JSON for confirmations.
        """
        endpoint = workflow.get("workflowEndpoint", "")
        schema = workflow.get("workflowSchema", {})
        call1 = schema.get("call1", {})
        body_data = call1.get("body_data", {})
        
        # Build field mapping from call1 schema
        # Maps internal field names (Input0, Input1) to API parameter names
        field_mapping = {}
        for field_name, field_config in body_data.items():
            api_field = field_config.get("endpoint_field_name", field_name)
            field_mapping[field_name] = api_field
        
        # Start with required fields
        payload = {
            "is_confirmed_by_user": is_confirmed,
            "runId": run_id,
            "workflowId": workflow_id,
        }
        
        # Extract fields from confirmation_data.body and map to API parameter names
        if confirmation_data:
            body = confirmation_data.get("body", {})
            
            # Map each field from body using the field_mapping
            for internal_field, value in body.items():
                # Skip if value is None or empty
                if value is None:
                    continue
                
                # Get API parameter name from mapping
                api_field = field_mapping.get(internal_field, internal_field)
                
                # Special handling: if api_field is one of the core fields, use it directly
                if api_field in ["is_confirmed_by_user", "runId", "workflowId"]:
                    # Already set above, skip
                    continue
                
                # Add to payload with API parameter name
                payload[api_field] = value
            
            # Also check if confirmation_data has direct fields (fallback for old format)
            for key in ["payloadBexio", "po_number", "user_id", "org_id"]:
                if key in confirmation_data and key not in payload:
                    payload[key] = confirmation_data[key]
        
        logger.info(f"🔧 HITL confirmation request built: {list(payload.keys())}")
        
        headers = {
            "Authorization": jwt_token,
            "Content-Type": "application/json",
        }
        
        # Add session_id to headers if provided
        if session_id:
            headers["X-Session-Id"] = session_id
        
        return {
            "endpoint": endpoint,
            "content_type": "application/json",
            "data": payload,
            "headers": headers,
        }

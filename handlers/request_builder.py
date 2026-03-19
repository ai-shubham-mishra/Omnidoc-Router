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
    ) -> Dict[str, Any]:
        """
        Build initial workflow request (call0) based on body_type.

        Returns:
            {
                "endpoint": "/private-po-registration",
                "content_type": "application/json" | "multipart/form-data",
                "data": {...},
                "headers": {"Authorization": "Bearer ..."},
                "has_files": bool,
                "file_fields": {field_name: {"paths": [file_paths], "endpoint_name": "files"}}
            }
        """
        schema = workflow.get("workflowSchema", {})
        call0 = schema.get("call0", {})
        body_type = call0.get("body_type", "json")
        endpoint = workflow.get("workflowEndpoint", "")

        # Build field name mapping
        field_mapping = {}
        for inp in required_inputs:
            field_mapping[inp["field"]] = inp.get("endpoint_field_name", inp["field"])

        headers = {"Authorization": jwt_token}

        file_fields = {}
        data_fields = {}

        for key, value in collected_inputs.items():
            if value == "__auto_runid__":
                data_fields["runId"] = run_id
                continue
            if value == "__hitl_field__":
                continue

            if isinstance(value, list) and value and isinstance(value[0], str) and ("tmp/" in value[0] or "tmp\\" in value[0]):
                endpoint_name = field_mapping.get(key, key)
                file_fields[key] = {"paths": value, "endpoint_name": endpoint_name}
            else:
                data_fields[key] = value

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
    ) -> Dict[str, Any]:
        """
        Build HIL confirmation request (call1).
        Always JSON for confirmations.
        """
        endpoint = workflow.get("workflowEndpoint", "")

        payload = {
            "is_confirmed_by_user": is_confirmed,
            "workflowId": workflow_id,
            "runId": run_id,
        }

        if confirmation_data:
            payload["confirmed_po"] = confirmation_data
            payload["confirmed_data"] = confirmation_data

        return {
            "endpoint": endpoint,
            "content_type": "application/json",
            "data": payload,
            "headers": {
                "Authorization": jwt_token,
                "Content-Type": "application/json",
            },
        }

"""
Input Collector for the LLM Router.
Parses workflow schema to determine required inputs,
auto-fills from JWT context, and extracts values from messages.
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class InputCollector:
    """Extract and validate workflow inputs from the existing workflowSchema."""

    def parse_workflow_inputs(self, workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parse required inputs from workflowSchema.call0.
        Handles both new format (body_data) and legacy format.
        
        Returns empty list for API-only workflows (MMM, etc.) that have no input requirements.
        """
        schema = workflow.get("workflowSchema", {})
        call0 = schema.get("call0", {})

        if "body_data" in call0:
            body_data = call0["body_data"]
        else:
            body_data = {k: v for k, v in call0.items() if k.startswith("Input")}

        required_inputs = []
        for input_key, input_spec in body_data.items():
            if not isinstance(input_spec, dict):
                continue
            # Use endpoint_field_name if available, otherwise fall back to input_key
            input_label = input_spec.get("inputLabel", input_key)
            endpoint_field_name = input_spec.get("endpoint_field_name", input_key)
            required_inputs.append(
                {
                    "field": input_key,
                    "label": input_label,
                    "type": input_spec.get("inputType", "str"),
                    "size": input_spec.get("inputSize", 1),
                    "endpoint_field_name": endpoint_field_name,  # Use from schema or default to field name
                    "collected": False,
                    "auto_extracted": False,
                    "value": None,
                }
            )

        return required_inputs

    def auto_fill_inputs(
        self,
        required_inputs: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Auto-fill inputs from JWT context (user_id, org_id).
        These never need to be asked from the user.
        """
        for inp in required_inputs:
            label_lower = inp.get("label", "").lower()
            field_lower = inp.get("field", "").lower()

            if any(
                kw in label_lower or kw in field_lower
                for kw in ["user_id", "userid", "user id"]
            ):
                if "user_id" in context:
                    inp["value"] = context["user_id"]
                    inp["collected"] = True
                    inp["auto_extracted"] = True

            elif any(
                kw in label_lower or kw in field_lower
                for kw in ["org_id", "orgid", "org id", "organization"]
            ):
                if "org_id" in context:
                    inp["value"] = context["org_id"]
                    inp["collected"] = True
                    inp["auto_extracted"] = True

            elif "runid" in label_lower or "run_id" in label_lower or "run id" in label_lower:
                inp["auto_extracted"] = True
                inp["collected"] = True
                inp["value"] = "__auto_runid__"

            elif any(
                kw in label_lower
                for kw in ["hitl", "confirmation", "confirmed"]
            ):
                inp["auto_extracted"] = True
                inp["collected"] = True
                inp["value"] = "__hitl_field__"

        return required_inputs

    def get_missing_inputs(self, required_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get list of inputs that still need to be collected."""
        return [
            inp
            for inp in required_inputs
            if not inp.get("collected") and not inp.get("auto_extracted")
        ]

    def get_body_type(self, workflow: Dict[str, Any]) -> str:
        """Get the body_type for the workflow's call0 (json or form)."""
        schema = workflow.get("workflowSchema", {})
        call0 = schema.get("call0", {})
        return call0.get("body_type", "json")

    def has_file_input(self, required_inputs: List[Dict[str, Any]]) -> bool:
        """Check if any required input is a file type."""
        return any(inp.get("type") == "file" for inp in required_inputs)

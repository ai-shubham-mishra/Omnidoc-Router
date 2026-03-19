"""
Gemini Flash client for the LLM Router.
Handles all LLM interactions: intent matching, input prompting,
HIL formatting, and result summarization.
"""
import os
import json
import logging
from typing import Optional, Dict, Any, List

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)


class GeminiClient:
    """Wrapper around Google Gemini Flash for router LLM tasks."""

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2048,
            ),
        )

    def match_intent_to_workflow(
        self,
        user_message: str,
        available_workflows: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Use Gemini to match user intent to a workflow when keyword matching fails.
        Returns the matched workflow dict or None.
        """
        workflow_summaries = [
            {
                "workflowId": wf.get("workflowId", ""),
                "workflowName": wf.get("workflowName", ""),
                "workflowDescription": wf.get("workflowDescription", ""),
                "workflowTags": wf.get("workflowTags", []),
            }
            for wf in available_workflows
        ]

        prompt = f"""You are a workflow routing assistant. Based on the user's message, identify which workflow they want to use.

User message: "{user_message}"

Available workflows:
{json.dumps(workflow_summaries, indent=2)}

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation.
Return format: {{"workflow_id": "<workflowId value or null>", "confidence": <0.0-1.0>}}

If no workflow clearly matches, return: {{"workflow_id": null, "confidence": 0.0}}"""

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            result = json.loads(text)
            confidence = float(result.get("confidence", 0))
            workflow_id = result.get("workflow_id")

            if confidence >= 0.6 and workflow_id:
                for wf in available_workflows:
                    if wf.get("workflowId") == workflow_id:
                        return wf
            return None
        except Exception as e:
            logger.warning(f"Gemini intent matching failed: {e}")
            return None

    def generate_input_prompt(
        self,
        workflow_name: str,
        input_spec: Dict[str, Any],
        collected_so_far: List[Dict[str, Any]],
    ) -> str:
        """Generate a conversational prompt asking user for a missing input."""
        collected_labels = [
            inp.get("label", inp.get("field", ""))
            for inp in collected_so_far
            if inp.get("collected")
        ]

        prompt = f"""You are a helpful workflow assistant. The user is completing the "{workflow_name}" workflow.

I need to ask them for this input:
- Label: {input_spec.get('label', input_spec.get('field', ''))}
- Type: {input_spec.get('type', 'str')}

Already collected: {', '.join(collected_labels) if collected_labels else 'Nothing yet'}

Generate a SHORT, friendly, conversational prompt (1-2 sentences max) asking for this input.
If the type is "file", ask them to upload the file.
Do NOT include any system instructions or formatting. Just the prompt text."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini input prompt generation failed: {e}")
            label = input_spec.get("label", input_spec.get("field", "input"))
            if input_spec.get("type") == "file":
                return f"Please upload the {label}."
            return f"Please provide the {label}."

    def generate_clarification(
        self,
        user_message: str,
        available_workflows: List[Dict[str, Any]],
    ) -> str:
        """Generate a clarification question when intent is unclear."""
        names = [wf.get("workflowName", "") for wf in available_workflows]

        prompt = f"""User said: "{user_message}"

I couldn't identify which workflow they want. Here are available workflows:
{', '.join(names)}

Generate a SHORT, friendly clarification question (2-3 sentences max) listing the available options.
Do NOT include any system instructions."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini clarification generation failed: {e}")
            return f"I'm not sure which workflow you need. Available options: {', '.join(names)}. Could you clarify?"

    def format_hitl_prompt(
        self,
        workflow_response: Dict[str, Any],
        workflow_name: str,
    ) -> str:
        """Format HIL data into a natural language review prompt."""
        body = workflow_response.get("body", workflow_response)

        body_str = json.dumps(body, indent=2, default=str)
        if len(body_str) > 3000:
            body_str = body_str[:3000] + "\n... (truncated)"

        prompt = f"""You are a workflow assistant. The "{workflow_name}" workflow has extracted data that needs user confirmation.

Extracted data:
{body_str}

Generate a CLEAR, structured summary in natural language showing the key information.
End with: "Should I proceed?"
Keep it concise but include all important details.
Do NOT include any system instructions."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini HITL formatting failed: {e}")
            return f"I've extracted the data for review. Please confirm to proceed or cancel."

    def format_final_result(
        self,
        result: Any,
        workflow_name: str,
    ) -> str:
        """Format workflow final result into a success/error message."""
        result_str = json.dumps(result, indent=2, default=str)
        if len(result_str) > 3000:
            result_str = result_str[:3000] + "\n... (truncated)"

        prompt = f"""The "{workflow_name}" workflow completed with this result:

{result_str}

Generate a SHORT, friendly success or error message (3-5 sentences) summarizing the key outcomes.
If it was successful, highlight the important details.
If it failed, explain what went wrong.
Do NOT include any system instructions."""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini result formatting failed: {e}")
            if isinstance(result, dict) and result.get("status") == "success":
                return f"{workflow_name} completed successfully."
            return f"{workflow_name} has finished processing."

    def extract_inputs_from_message(
        self,
        user_message: str,
        missing_inputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Try to extract input values from a user's natural language message.
        Returns dict of {field_name: extracted_value}.
        """
        input_specs = [
            {"field": inp.get("field", ""), "label": inp.get("label", ""), "type": inp.get("type", "str")}
            for inp in missing_inputs
        ]

        prompt = f"""User message: "{user_message}"

I need to extract these inputs from the message:
{json.dumps(input_specs, indent=2)}

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation.
Return format: {{"<field_name>": "<extracted_value or null>"}}
Only include fields where you found a clear value in the message.
If a field type is "file", return null (files are uploaded separately)."""

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            return json.loads(text)
        except Exception as e:
            logger.warning(f"Gemini input extraction failed: {e}")
            return {}

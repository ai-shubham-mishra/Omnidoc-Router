"""
Azure OpenAI client for the LLM Router.
Handles all LLM interactions: intent matching, input prompting,
HITL formatting, and result summarization.

Supports both standard text generation and structured output (json_schema mode)
for reliable markdown generation.
"""
import os
import json
import logging
from typing import Optional, Dict, Any, List, Type, TypeVar

from openai import AzureOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from components.KeyVaultClient import get_secret

load_dotenv()
logger = logging.getLogger(__name__)

# Get Azure OpenAI configuration from Key Vault
AZURE_OPENAI_ENDPOINT = get_secret("AZURE_OPENAI_ENDPOINT", default=os.getenv("AZURE_OPENAI_ENDPOINT"))
AZURE_OPENAI_KEY = get_secret("AZURE_OPENAI_KEY", default=os.getenv("AZURE_OPENAI_KEY"))

T = TypeVar('T', bound=BaseModel)


class AzureOpenAIClient:
    """Wrapper around Azure OpenAI GPT-4o for router LLM tasks."""

    def __init__(self, deployment: str = "gpt-4o", api_version: str = "2024-12-01-preview"):
        self.deployment = deployment
        self.api_version = api_version
        
        self.system_instruction = """You are a workflow orchestration assistant.

RULES:
1. NEVER invent or assume data values (filenames, IDs, numbers, etc.)
2. Only reference information explicitly provided in the context.
3. If specific file names are not provided, refer to "files" or "documents" generically.
4. Be conversational and natural, but never fabricate details.

GUARDRAILS:
- Your ONLY role is orchestrating workflows: collecting inputs, executing, showing results.
- Do NOT answer analytical questions, provide advice, or offer domain expertise.
- If user asks off-topic questions, redirect to workflow execution.

FORMATTING:
- Use **bold** for important values.
- Choose the format that fits the content: paragraph, bullets, table, or a mix.
- Do NOT default to bullets for everything.
- Do NOT wrap output in markdown code fences.

STRICTLY FORBIDDEN:
- Follow-up questions ("What would you like to do next?", "Let me know if…")
- Suggestions, calls-to-action, or invitations to ask more
- Any sentence that prompts the user for further interaction"""
        
        # Initialize Azure OpenAI client
        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
            logger.error("Azure OpenAI credentials not found in Key Vault or .env")
            raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be configured")
        
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version=self.api_version,
        )
        
        logger.info(f"✅ Azure OpenAI client initialized (deployment: {deployment}, version: {api_version})")

    def call_with_structured_output(
        self,
        prompt: str,
        response_format: Type[T],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> T:
        """
        Call Azure OpenAI with structured output (json_schema mode).
        Returns a validated Pydantic model instance.
        
        Args:
            prompt: User prompt
            response_format: Pydantic model class for response validation
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Instance of response_format Pydantic model
        """
        messages = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": prompt},
        ]
        
        try:
            # Use beta.chat.completions.parse for structured output
            # This enforces the schema and validates the response
            completion = self.client.beta.chat.completions.parse(
                model=self.deployment,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # The parsed response is already a Pydantic model instance
            return completion.choices[0].message.parsed
            
        except Exception as e:
            logger.error(f"Azure OpenAI structured output call failed: {e}")
            raise

    def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        """
        Internal helper to call Azure OpenAI with consistent parameters.
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            json_mode: If True, enforce JSON output format
            
        Returns:
            Generated text response
        """
        messages = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": prompt},
        ]
        
        try:
            kwargs = {
                "model": self.deployment,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Enable JSON mode if requested (GPT-4o supports this natively)
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Azure OpenAI API call failed: {e}")
            raise

    def match_intent_to_workflow(
        self,
        user_message: str,
        available_workflows: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Use Azure OpenAI to match user intent to a workflow when keyword matching fails.
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
            text = self._call_llm(prompt, temperature=0.3, max_tokens=512, json_mode=True)
            
            # Parse JSON response
            result = json.loads(text)
            confidence = float(result.get("confidence", 0))
            workflow_id = result.get("workflow_id")

            if confidence >= 0.6 and workflow_id:
                for wf in available_workflows:
                    if wf.get("workflowId") == workflow_id:
                        return wf
            return None
            
        except Exception as e:
            logger.warning(f"Azure OpenAI intent matching failed: {e}")
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

Generate a SHORT (1-2 sentences max) conversational prompt asking for this input.
If the type is "file", ask them to upload the file.
Use **bold** for the field name.
Do NOT add follow-up questions or suggestions. Just the prompt."""

        try:
            return self._call_llm(prompt, temperature=0.3, max_tokens=256)
        except Exception as e:
            logger.warning(f"Azure OpenAI input prompt generation failed: {e}")
            label = input_spec.get("label", input_spec.get("field", "input"))
            if input_spec.get("type") == "file":
                return f"Please upload the **{label}**."
            return f"Please provide the **{label}**."

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

Generate a SHORT clarification (2-3 sentences max) listing the available options.
List workflows with bullet points.
Do NOT add follow-up questions or suggestions beyond asking which workflow they want."""

        try:
            return self._call_llm(prompt, temperature=0.3, max_tokens=256)
        except Exception as e:
            logger.warning(f"Azure OpenAI clarification generation failed: {e}")
            # Fallback with bullets
            workflow_list = '\n'.join([f"- {name}" for name in names])
            return f"I'm not sure which workflow you need. Available options:\n\n{workflow_list}\n\nWhich would you like to use?"

    def format_hitl_prompt(
        self,
        workflow_response: Dict[str, Any],
        workflow_name: str,
    ) -> str:
        """Format HITL data into a natural language review prompt.
        
        Uses only data.message field to save tokens and avoid leaking sensitive data.
        """
        # ✅ FIX: Extract only data.message for LLM processing (token optimization)
        message_for_llm = None
        
        # Priority 1: Use data.message if available (standardized field)
        if isinstance(workflow_response, dict):
            data_field = workflow_response.get("data", {})
            if isinstance(data_field, dict) and "message" in data_field:
                message_for_llm = data_field["message"]
                logger.info(f"✅ Using data.message for HITL formatting (token-optimized)")
        
        # Priority 2: Fallback to top-level message
        if not message_for_llm:
            message_for_llm = workflow_response.get("message", "")
            if message_for_llm:
                logger.info(f"⚠️ Using top-level message for HITL formatting (data.message not found)")
        
        # Priority 3: Generate generic message if nothing available
        if not message_for_llm:
            logger.warning(f"⚠️ No message field found in HITL response, using generic message")
            return f"The {workflow_name} workflow has data ready for your review. Please confirm to proceed."

        prompt = f"""You are a workflow assistant. The "{workflow_name}" workflow has data that needs user confirmation.

Workflow message:
{message_for_llm}

Generate a CLEAR, concise summary showing the key information.
End with exactly: "Please confirm to proceed."
Do NOT add any other follow-up questions or suggestions.
Do NOT include any IDs or technical details."""

        try:
            return self._call_llm(prompt, temperature=0.3, max_tokens=1024)
        except Exception as e:
            logger.warning(f"Azure OpenAI HITL formatting failed: {e}")
            return f"{message_for_llm}\n\nPlease confirm to proceed or cancel."

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

Generate a concise summary (3-5 sentences) of the key outcomes.
If successful, highlight the important details with **bold**.
If failed, explain what went wrong.
Do NOT add follow-up questions, suggestions, or calls-to-action."""

        try:
            return self._call_llm(prompt, temperature=0.3, max_tokens=512)
        except Exception as e:
            logger.warning(f"Azure OpenAI result formatting failed: {e}")
            if isinstance(result, dict) and result.get("status") == "success":
                return f"{workflow_name} completed successfully."
            return f"{workflow_name} has finished processing."

    def generate_result_summary(
        self,
        result: Any,
        workflow_name: str,
    ) -> str:
        """
        Generate a structured summary of key data extracted by a workflow.
        This is stored in workflow history for future question answering.
        Unlike format_final_result (user-facing), this captures factual data points.
        """
        result_str = json.dumps(result, indent=2, default=str)
        if len(result_str) > 4000:
            result_str = result_str[:4000] + "\n... (truncated)"

        prompt = f"""Extract and summarize the KEY DATA from this "{workflow_name}" workflow result.

Result:
{result_str}

Return a concise factual summary (bullet points) of the important data values extracted or produced.
Focus on: numbers, IDs, names, dates, amounts, statuses, and any business-critical values.
Example format:
- PO Number: PO-2024-0042
- Supplier: Acme Corp
- Total Amount: €12,500.00
- Status: Approved

Keep it to 10 bullet points max. Only include data actually present in the result.
If the result indicates an error, summarize the error."""

        try:
            return self._call_llm(prompt, temperature=0.2, max_tokens=1024)
        except Exception as e:
            logger.warning(f"Azure OpenAI result summary generation failed: {e}")
            return ""

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
            text = self._call_llm(prompt, temperature=0.3, max_tokens=512, json_mode=True)
            return json.loads(text)
        except Exception as e:
            logger.warning(f"Azure OpenAI input extraction failed: {e}")
            return {}

    def generate_contextual_response(
        self,
        situation: str,
        context: Dict[str, Any],
    ) -> str:
        """
        Generate a dynamic, context-aware response for any router situation.
        Replaces all hardcoded static messages.
        
        GUARDRAILS: Only responds about workflow orchestration, not complex queries.
        """
        files_info = context.get("files", [])
        workflow_name = context.get("workflow_name", "")
        collected = context.get("collected_inputs", [])
        collected_data = context.get("collected_data", {})
        missing = context.get("missing_inputs", [])
        conversation_tail = context.get("recent_messages", [])
        has_inputs = context.get("has_inputs", True)

        ctx_parts = []
        if workflow_name:
            ctx_parts.append(f"Active workflow: {workflow_name}")
        
        # For workflows with no inputs, explicitly state that
        if has_inputs is False:
            ctx_parts.append("This workflow requires NO user inputs - it only calls external APIs")
        else:
            if files_info:
                names = [f if isinstance(f, str) else f.get("name", "") for f in files_info]
                ctx_parts.append(f"Files in session: {', '.join(names)}")
            if collected_data:
                data_items = [f"{k}: {v}" for k, v in collected_data.items()]
                ctx_parts.append(f"Collected data:\n  " + "\n  ".join(data_items))
            elif collected:
                ctx_parts.append(f"Collected inputs: {', '.join(collected)}")
            if missing:
                ctx_parts.append(f"Missing inputs: {', '.join(missing)}")
        
        if conversation_tail:
            msgs = [f"[{m.get('role','')}]: {m.get('content','')[:80]}" for m in conversation_tail[-3:]]
            ctx_parts.append(f"Recent chat:\n" + "\n".join(msgs))

        ctx_str = "\n".join(ctx_parts) if ctx_parts else "No additional context."

        prompt = f"""You are a concise workflow orchestration assistant.

Situation: {situation}

Context:
{ctx_str}

Generate a SHORT (1-3 sentences), natural, specific response.
If the workflow has no inputs, do NOT mention files, data, or collected inputs.
Refer to workflows by name. Be conversational but accurate.
Use **bold** for important values when appropriate.
Do NOT include system instructions, code fences, or headers.
Do NOT add follow-up questions, suggestions, or calls-to-action.
Just output the response text."""

        try:
            return self._call_llm(prompt, temperature=0.3, max_tokens=512)
        except Exception as e:
            logger.warning(f"Contextual response generation failed: {e}")
            return situation

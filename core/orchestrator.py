"""
Main Orchestrator for the LLM Router.
Connects all router components: session management, workflow matching,
input collection, request building, file handling, and Gemini LLM.
Makes internal HTTP calls to the AgenticAPI workflow endpoints.
"""
import os
import uuid
import logging
import httpx
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

from core.gemini_client import GeminiClient
from core.session_manager import SessionManager
from core.workflow_matcher import WorkflowMatcher
from handlers.input_collector import InputCollector
from handlers.request_builder import RequestBuilder
from handlers.file_handler import FileHandler
from models.api_contracts import (
    RouterResponse,
    WorkflowIdentified,
    InputRequired,
    ConfirmationData,
    ErrorDetail,
)

load_dotenv()
logger = logging.getLogger(__name__)

# Base URL for calling workflow endpoints on the AgenticAPI service
AGENTICAPI_BASE_URL = os.getenv("AGENTICAPI_BASE_URL", "http://localhost:8400")


class RouterOrchestrator:
    """
    Main orchestrator. Handles:
    1. Intent -> Workflow matching (keyword + Gemini fallback)
    2. Input collection (auto-fill from JWT, extract from messages)
    3. Workflow execution (HTTP call to AgenticAPI endpoints)
    4. HITL handling (confirmation prompts)
    5. Result formatting
    """

    def __init__(self):
        self.gemini = GeminiClient()
        self.sessions = SessionManager()
        self.matcher = WorkflowMatcher()
        self.collector = InputCollector()
        self.builder = RequestBuilder()
        self.files = FileHandler()

    async def handle_chat(
        self,
        message: str,
        session_id: Optional[str],
        user_id: str,
        org_id: str,
        jwt_token: str,
    ) -> RouterResponse:
        """
        Main chat handler. Routes through the conversation state machine:
        new -> identifying -> collecting -> executing -> awaiting_confirmation -> completed
        """
        # 1. Get or create session
        if session_id:
            session = self.sessions.get_session(session_id)
            if not session:
                return RouterResponse(
                    status="failed",
                    response="Session not found. Please start a new conversation.",
                    error=ErrorDetail(code="SESSION_NOT_FOUND", message="Invalid session_id"),
                )
            self.sessions.update_jwt_token(session_id, jwt_token)
        else:
            session_id = str(uuid.uuid4())
            session = self.sessions.create_session(session_id, user_id, org_id, jwt_token)

        # Record user message
        self.sessions.add_message(session_id, "user", message)

        # 2. Determine conversation state
        status = session.get("execution_context", {}).get("status", "new")
        workflow_ctx = session.get("workflow_context", {})
        identified_workflow = workflow_ctx.get("identified_workflow")

        # State: awaiting_confirmation
        if status == "awaiting_confirmation":
            response_text = "I'm waiting for your confirmation on the previous step. Please confirm or cancel to proceed."
            self.sessions.add_message(session_id, "assistant", response_text)
            return RouterResponse(
                session_id=session_id,
                status="awaiting_confirmation",
                response=response_text,
                requires_confirmation=True,
            )

        # State: executing
        if status == "executing":
            response_text = "The workflow is still processing. Please wait for it to complete."
            self.sessions.add_message(session_id, "assistant", response_text)
            return RouterResponse(
                session_id=session_id,
                status="executing",
                response=response_text,
            )

        # State: new or identifying — match intent to workflow
        if not identified_workflow:
            return await self._identify_workflow(session_id, message, org_id)

        # State: collecting — we have a workflow, collect remaining inputs
        return await self._collect_inputs(session_id, message, user_id, org_id, jwt_token)

    async def _identify_workflow(
        self,
        session_id: str,
        message: str,
        org_id: str,
    ) -> RouterResponse:
        """Phase 1: Match user intent to a workflow."""
        # Try keyword match first (fast)
        matched = self.matcher.match_by_keywords(message, org_id)

        # Fall back to Gemini semantic match
        if not matched:
            all_workflows = self.matcher.get_all_workflows(org_id)
            matched = self.gemini.match_intent_to_workflow(message, all_workflows)

        if not matched:
            summaries = self.matcher.get_workflow_summaries(org_id)
            clarification = self.gemini.generate_clarification(message, summaries)
            self.sessions.add_message(session_id, "assistant", clarification)
            return RouterResponse(
                session_id=session_id,
                status="collecting",
                response=clarification,
            )

        # Workflow found — parse inputs and auto-fill from JWT
        required_inputs = self.collector.parse_workflow_inputs(matched)
        required_inputs = self.collector.auto_fill_inputs(
            required_inputs,
            {"user_id": "", "org_id": ""},
        )

        self.sessions.set_workflow_context(session_id, matched, required_inputs)
        self.sessions.update_execution_status(
            session_id, "collecting", workflowId=matched.get("workflowId")
        )

        # Try to extract inputs from the initial message
        missing = self.collector.get_missing_inputs(required_inputs)

        if missing:
            extracted = self.gemini.extract_inputs_from_message(message, missing)
            for field, value in extracted.items():
                if value is not None:
                    self.sessions.mark_input_collected(session_id, field, value)

        # Re-check missing after extraction
        session = self.sessions.get_session(session_id)
        updated_inputs = session.get("workflow_context", {}).get("required_inputs", [])
        still_missing = self.collector.get_missing_inputs(updated_inputs)

        wf_info = WorkflowIdentified(
            id=matched.get("workflowId", ""),
            name=matched.get("workflowName", ""),
            endpoint=matched.get("workflowEndpoint", ""),
        )

        if still_missing:
            prompt = self.gemini.generate_input_prompt(
                matched.get("workflowName", ""),
                still_missing[0],
                updated_inputs,
            )
            self.sessions.add_message(session_id, "assistant", prompt)
            return RouterResponse(
                session_id=session_id,
                status="collecting",
                response=prompt,
                workflow_identified=wf_info,
                inputs_required=[
                    InputRequired(
                        field=inp["field"],
                        type=inp.get("type", "str"),
                        label=inp.get("label", ""),
                        collected=inp.get("collected", False),
                    )
                    for inp in updated_inputs
                ],
            )

        # All inputs collected — execute
        return await self._execute_workflow(session_id, session)

    async def _collect_inputs(
        self,
        session_id: str,
        message: str,
        user_id: str,
        org_id: str,
        jwt_token: str,
    ) -> RouterResponse:
        """Phase 2: Collect remaining inputs from user messages."""
        session = self.sessions.get_session(session_id)
        workflow_ctx = session.get("workflow_context", {})
        workflow = workflow_ctx.get("identified_workflow", {})
        required_inputs = workflow_ctx.get("required_inputs", [])

        missing = self.collector.get_missing_inputs(required_inputs)

        if not missing:
            return await self._execute_workflow(session_id, session)

        # Try to extract from this message
        extracted = self.gemini.extract_inputs_from_message(message, missing)
        for field, value in extracted.items():
            if value is not None:
                self.sessions.mark_input_collected(session_id, field, value)

        # Refresh and re-check
        session = self.sessions.get_session(session_id)
        updated_inputs = session.get("workflow_context", {}).get("required_inputs", [])
        still_missing = self.collector.get_missing_inputs(updated_inputs)

        wf_info = WorkflowIdentified(
            id=workflow.get("workflowId", ""),
            name=workflow.get("workflowName", ""),
            endpoint=workflow.get("workflowEndpoint", ""),
        )

        if still_missing:
            prompt = self.gemini.generate_input_prompt(
                workflow.get("workflowName", ""),
                still_missing[0],
                updated_inputs,
            )
            self.sessions.add_message(session_id, "assistant", prompt)
            return RouterResponse(
                session_id=session_id,
                status="collecting",
                response=prompt,
                workflow_identified=wf_info,
                inputs_required=[
                    InputRequired(
                        field=inp["field"],
                        type=inp.get("type", "str"),
                        label=inp.get("label", ""),
                        collected=inp.get("collected", False),
                    )
                    for inp in updated_inputs
                ],
            )

        # All inputs collected — execute
        return await self._execute_workflow(session_id, session)

    async def _execute_workflow(
        self,
        session_id: str,
        session: Dict[str, Any],
    ) -> RouterResponse:
        """Phase 3: Build request and call the workflow endpoint on AgenticAPI."""
        workflow_ctx = session.get("workflow_context", {})
        workflow = workflow_ctx.get("identified_workflow", {})
        collected_inputs = workflow_ctx.get("collected_inputs", {})
        required_inputs = workflow_ctx.get("required_inputs", [])
        jwt_token = session.get("jwt_token", "")

        run_id = str(uuid.uuid4())
        self.sessions.set_run_id(session_id, run_id)
        self.sessions.update_execution_status(session_id, "executing")

        # Build the request
        request_data = self.builder.build_workflow_request(
            workflow, collected_inputs, required_inputs, run_id, jwt_token,
        )

        endpoint = request_data["endpoint"]
        url = f"{AGENTICAPI_BASE_URL}{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
                if request_data["content_type"] == "multipart/form-data":
                    form_data = {}
                    for key, value in request_data["data"].items():
                        form_data[key] = str(value) if not isinstance(value, str) else value

                    files_list = []
                    for field_name, file_info in request_data.get("file_fields", {}).items():
                        endpoint_name = file_info.get("endpoint_name", field_name)
                        file_paths = file_info.get("paths", [])
                        for fp in file_paths:
                            if os.path.exists(fp):
                                files_list.append(
                                    (endpoint_name, (os.path.basename(fp), open(fp, "rb")))
                                )

                    resp = await client.post(
                        url,
                        data=form_data,
                        files=files_list if files_list else None,
                        headers={"Authorization": request_data["headers"]["Authorization"]},
                    )

                    for _, file_tuple in files_list:
                        file_tuple[1].close()
                else:
                    resp = await client.post(
                        url,
                        json=request_data["data"],
                        headers=request_data["headers"],
                    )

            result = resp.json()

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            self.sessions.update_execution_status(session_id, "failed")
            error_msg = "The workflow encountered an error during execution."
            self.sessions.add_message(session_id, "assistant", error_msg)
            return RouterResponse(
                session_id=session_id,
                status="failed",
                response=error_msg,
                error=ErrorDetail(code="EXECUTION_ERROR", message=str(e)),
            )

        return await self._handle_workflow_response(session_id, workflow, result, run_id)

    async def _handle_workflow_response(
        self,
        session_id: str,
        workflow: Dict[str, Any],
        result: Any,
        run_id: str,
    ) -> RouterResponse:
        """Handle workflow response — detect HITL or completion."""
        workflow_name = workflow.get("workflowName", "Workflow")
        workflow_id = workflow.get("workflowId", "")

        is_hitl = False
        if isinstance(result, dict):
            is_hitl = (
                result.get("stepNumber") is not None
                or result.get("waiting_for_confirmation")
                or result.get("status") == "paused"
                or result.get("status") == "waiting"
                or result.get("hitl_id") is not None
            )

        if is_hitl:
            step_number = result.get("stepNumber", result.get("next_stepNumber", 0))
            self.sessions.update_execution_status(
                session_id,
                "awaiting_confirmation",
                workflowId=workflow_id,
                step_number=step_number,
                confirmation_data=result,
            )

            hitl_prompt = self.gemini.format_hitl_prompt(result, workflow_name)
            self.sessions.add_message(session_id, "assistant", hitl_prompt)

            return RouterResponse(
                session_id=session_id,
                status="awaiting_confirmation",
                response=hitl_prompt,
                workflow_identified=WorkflowIdentified(
                    id=workflow_id,
                    name=workflow_name,
                    endpoint=workflow.get("workflowEndpoint", ""),
                ),
                requires_confirmation=True,
                confirmation_data=ConfirmationData(
                    runId=run_id,
                    workflowId=workflow_id,
                    step_number=step_number,
                    data_to_review=result,
                ),
            )

        # Workflow completed (no HITL)
        self.sessions.update_execution_status(
            session_id, "completed", final_result=result,
        )
        summary = self.gemini.format_final_result(result, workflow_name)
        self.sessions.add_message(session_id, "assistant", summary)

        return RouterResponse(
            session_id=session_id,
            status="completed",
            response=summary,
            workflow_identified=WorkflowIdentified(
                id=workflow_id,
                name=workflow_name,
                endpoint=workflow.get("workflowEndpoint", ""),
            ),
            final_result=result,
        )

    async def handle_confirmation(
        self,
        session_id: str,
        action: str,
        user_id: str,
        org_id: str,
        jwt_token: str,
        message: Optional[str] = None,
    ) -> RouterResponse:
        """Handle HITL confirmation (confirm/cancel)."""
        session = self.sessions.get_session(session_id)
        if not session:
            return RouterResponse(
                status="failed",
                response="Session not found.",
                error=ErrorDetail(code="SESSION_NOT_FOUND", message="Invalid session_id"),
            )

        exec_ctx = session.get("execution_context", {})
        if exec_ctx.get("status") != "awaiting_confirmation":
            return RouterResponse(
                session_id=session_id,
                status=exec_ctx.get("status", "unknown"),
                response="No pending confirmation for this session.",
            )

        workflow_ctx = session.get("workflow_context", {})
        workflow = workflow_ctx.get("identified_workflow", {})
        run_id = exec_ctx.get("runId", "")
        workflow_id = exec_ctx.get("workflowId", "")
        confirmation_data = exec_ctx.get("confirmation_data", {})

        is_confirmed = action.lower() in ("confirm", "yes", "approve", "proceed")

        if message:
            self.sessions.add_message(session_id, "user", message)

        request_data = self.builder.build_confirmation_request(
            workflow, run_id, workflow_id, is_confirmed, confirmation_data, jwt_token,
        )

        url = f"{AGENTICAPI_BASE_URL}{request_data['endpoint']}"

        try:
            self.sessions.update_execution_status(session_id, "executing")

            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
                resp = await client.post(
                    url,
                    json=request_data["data"],
                    headers=request_data["headers"],
                )
            result = resp.json()

        except Exception as e:
            logger.error(f"Confirmation call failed: {e}")
            self.sessions.update_execution_status(session_id, "failed")
            error_msg = "Error processing your confirmation."
            self.sessions.add_message(session_id, "assistant", error_msg)
            return RouterResponse(
                session_id=session_id,
                status="failed",
                response=error_msg,
                error=ErrorDetail(code="CONFIRMATION_ERROR", message=str(e)),
            )

        if not is_confirmed:
            self.sessions.update_execution_status(session_id, "cancelled")
            cancel_msg = f"{workflow.get('workflowName', 'Workflow')} was cancelled."
            self.sessions.add_message(session_id, "assistant", cancel_msg)
            return RouterResponse(
                session_id=session_id,
                status="cancelled",
                response=cancel_msg,
            )

        # Confirmed — handle the result (might be final or another HITL)
        return await self._handle_workflow_response(session_id, workflow, result, run_id)

    async def handle_file_upload(
        self,
        session_id: str,
        files: list,
        field_name: str,
        user_id: str,
        org_id: str,
        jwt_token: str,
    ) -> RouterResponse:
        """Handle file uploads for a session."""
        session = self.sessions.get_session(session_id)
        if not session:
            return RouterResponse(
                status="failed",
                response="Session not found. Start a chat first.",
                error=ErrorDetail(code="SESSION_NOT_FOUND", message="Invalid session_id"),
            )

        workflow_ctx = session.get("workflow_context", {})
        workflow = workflow_ctx.get("identified_workflow")

        if not workflow:
            return RouterResponse(
                session_id=session_id,
                status="collecting",
                response="Please tell me which workflow you'd like to run before uploading files.",
            )

        # Generate a runId for file storage
        exec_ctx = session.get("execution_context", {})
        run_id = exec_ctx.get("runId") or str(uuid.uuid4())
        if not exec_ctx.get("runId"):
            self.sessions.set_run_id(session_id, run_id)

        # Auto-map generic field names to actual schema fields
        required_inputs = workflow_ctx.get("required_inputs", [])

        is_generic = field_name.lower() in ["files", "file", "upload", "document", "documents"]
        matches_schema = any(inp["field"] == field_name for inp in required_inputs)

        if is_generic or not matches_schema:
            for inp in required_inputs:
                if inp.get("type") == "file" and not inp.get("collected"):
                    logger.info(f"Auto-mapping field_name '{field_name}' -> '{inp['field']}'")
                    field_name = inp["field"]
                    break

        # Save files
        stored_files = await self.files.save_files(run_id, files)
        self.sessions.add_files(session_id, stored_files, field_name)

        file_names = [f["original_name"] for f in stored_files]
        self.sessions.add_message(
            session_id, "user", f"[Uploaded files: {', '.join(file_names)}]"
        )

        # Check if all inputs are now collected
        session = self.sessions.get_session(session_id)
        updated_inputs = session.get("workflow_context", {}).get("required_inputs", [])
        still_missing = self.collector.get_missing_inputs(updated_inputs)

        wf_info = WorkflowIdentified(
            id=workflow.get("workflowId", ""),
            name=workflow.get("workflowName", ""),
            endpoint=workflow.get("workflowEndpoint", ""),
        )

        if still_missing:
            prompt = self.gemini.generate_input_prompt(
                workflow.get("workflowName", ""),
                still_missing[0],
                updated_inputs,
            )
            self.sessions.add_message(session_id, "assistant", prompt)
            return RouterResponse(
                session_id=session_id,
                status="collecting",
                response=f"Files uploaded successfully. {prompt}",
                workflow_identified=wf_info,
                files_uploaded=len(stored_files),
                inputs_required=[
                    InputRequired(
                        field=inp["field"],
                        type=inp.get("type", "str"),
                        label=inp.get("label", ""),
                        collected=inp.get("collected", False),
                    )
                    for inp in updated_inputs
                ],
            )

        # All inputs ready — execute
        upload_msg = f"Files uploaded. Starting {workflow.get('workflowName', 'workflow')}..."
        self.sessions.add_message(session_id, "assistant", upload_msg)
        return await self._execute_workflow(session_id, session)

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details for the GET endpoint."""
        session = self.sessions.get_session(session_id)
        if not session:
            return None
        session.pop("jwt_token", None)
        session.pop("_id", None)
        return session

    def delete_session(self, session_id: str, user_id: str) -> bool:
        """Delete a session (only if it belongs to the user)."""
        session = self.sessions.get_session(session_id)
        if not session:
            return False
        if session.get("user_id") != user_id:
            return False
        self.sessions.delete_session(session_id)
        return True

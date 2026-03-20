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
from core.file_intelligence import FileIntelligence
from handlers.input_collector import InputCollector
from handlers.request_builder import RequestBuilder
from handlers.file_handler import FileHandler
from utils.intent_detector import IntentDetector
from models.api_contracts import (
    RouterResponse,
    WorkflowIdentified,
    InputRequired,
    ConfirmationData,
    ErrorDetail,
    FileUploaded,
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
        self.intent = IntentDetector()
        self.file_intel = FileIntelligence()

    async def handle_message(
        self,
        message: str,
        session_id: Optional[str],
        files: List,
        user_id: str,
        org_id: str,
        jwt_token: str,
    ) -> RouterResponse:
        """
        Unified message handler for conversational workflow orchestration.
        Handles: chat messages, file uploads, intent detection, multi-workflow sessions.
        
        Key Features:
        - Files stored as context (not auto-execute)
        - Explicit execution intent required
        - Multi-workflow support
        - Session persists across workflows
        """
        #  1. Get or create session
        if session_id:
            session = await self.sessions.get_session(session_id)
            if not session:
                return RouterResponse(
                    status="failed",
                    response="Session not found. Please start a new conversation.",
                    error=ErrorDetail(code="SESSION_NOT_FOUND", message="Invalid session_id"),
                )
            await self.sessions.update_jwt_token(session_id, jwt_token)
        else:
            session_id = str(uuid.uuid4())
            session = self.sessions.create_session(session_id, user_id, org_id, jwt_token)

        # 2. Handle file uploads (add to context with classification)
        files_uploaded = []
        if files:
            stored_files = await self.files.save_files_to_session(session_id, files)
            
            # Classify each file for intelligent matching
            current_wf = session.get("current_workflow", {})
            active_workflow_name = current_wf.get("workflow_name")
            for f in stored_files:
                f["classification"] = self.file_intel.classify_file(f)
                f["status"] = "available"
                f["used_by_workflows"] = []
                if active_workflow_name:
                    f["uploaded_during_workflow"] = active_workflow_name
            
            await self.sessions.add_files(session_id, stored_files)
            files_uploaded = [
                FileUploaded(
                    file_id=f["file_id"],
                    original_name=f["original_name"],
                    mime_type=f["mime_type"],
                    size_bytes=f["size_bytes"],
                    uploaded_at=f["uploaded_at"],
                )
                for f in stored_files
            ]
            logger.info(f"📎 {len(files)} file(s) classified and added to session context")

        # 3. Add message to conversation (if provided)
        if message:
            await self.sessions.add_message(session_id, "user", message)

        # 4. Refresh session to get latest state
        session = await self.sessions.get_session(session_id)
        current_wf = session.get("current_workflow", {})
        wf_status = current_wf.get("status", "idle")

        # 4b. Immediately match newly uploaded files to active workflow inputs
        #     This runs BEFORE any intent routing so files are matched regardless
        #     of whether the message is a question, delay, or collect.
        if files_uploaded and current_wf.get("workflow_id") and wf_status in ("collecting", "ready_to_execute"):
            required_inputs = current_wf.get("required_inputs", [])
            session_files = session.get("uploaded_files", [])
            workflow_name = current_wf.get("workflow_name", "")
            matched_any = False

            for inp in required_inputs:
                if inp.get("type") == "file" and not inp.get("collected"):
                    best_file = self.file_intel.find_best_file_for_input(
                        session_files, inp, workflow_name
                    )
                    if best_file:
                        file_path = best_file.get("stored_path")
                        if file_path:
                            await self.sessions.mark_input_collected(
                                session_id, inp["field"], [file_path]
                            )
                            matched_any = True
                            logger.info(
                                f"✅ Immediate file match: '{best_file.get('original_name')}' "
                                f"→ {inp.get('label', inp['field'])} for '{workflow_name}'"
                            )

            if matched_any:
                # Refresh session after matching so all subsequent handlers see updated state
                session = await self.sessions.get_session(session_id)
                current_wf = session.get("current_workflow", {})
                wf_status = current_wf.get("status", "idle")

        # 5. State machine routing
        if wf_status == "awaiting_confirmation":
            # Check if user is confirming/canceling via text message
            if message:
                intent = self.intent.detect_execution_intent(message)
                
                if intent == "execute":
                    # User confirmed via text ("yes", "okay", "go ahead", etc.)
                    logger.info(f"✅ HITL confirmation detected via text: '{message}'")
                    return await self.handle_confirmation(
                        session_id, 
                        action="confirm", 
                        user_id=user_id, 
                        org_id=org_id, 
                        jwt_token=jwt_token,
                        message=message
                    )
                
                elif intent == "delay":
                    # User canceled via text ("cancel", "stop", "wait", etc.)
                    logger.info(f"🛑 HITL cancellation detected via text: '{message}'")
                    return await self.handle_confirmation(
                        session_id, 
                        action="cancel", 
                        user_id=user_id, 
                        org_id=org_id, 
                        jwt_token=jwt_token,
                        message=message
                    )
                
                elif intent == "question":
                    # User asking about the HITL data or session context
                    logger.info(f"❓ Question during HITL: '{message}'")
                    session_context = self.file_intel.build_session_context(session)
                    # Include confirmation data in context for HITL-specific questions
                    confirmation_data = current_wf.get("confirmation_data")
                    if confirmation_data:
                        session_context["pending_confirmation"] = {
                            "workflow": current_wf.get("workflow_name"),
                            "data_to_review": confirmation_data,
                        }
                    answer = self.file_intel.answer_session_question(message, session_context)
                    answer += "\n\nPlease confirm or cancel when you're ready."
                    await self.sessions.add_message(session_id, "assistant", answer)
                    return RouterResponse(
                        session_id=session_id,
                        status="awaiting_confirmation",
                        response=answer,
                        requires_confirmation=True,
                        files_uploaded=files_uploaded if files_uploaded else None,
                    )
            
            # No clear intent - return waiting message
            response_text = "Waiting for your confirmation. Use the confirm/cancel buttons or send 'yes'/'cancel'."
            await self.sessions.add_message(session_id, "assistant", response_text)
            return RouterResponse(
                session_id=session_id,
                status="awaiting_confirmation",
                response=response_text,
                requires_confirmation=True,
                files_uploaded=files_uploaded if files_uploaded else None,
            )

        if wf_status == "executing":
            response_text = "Workflow is currently executing. Please wait..."
            return RouterResponse(
                session_id=session_id,
                status="executing",
                response=response_text,
                files_uploaded=files_uploaded if files_uploaded else None,
            )

        # 5b. Handle conversational questions (RAG-style)
        if message:
            msg_intent = self.intent.detect_execution_intent(message)
            if msg_intent == "question":
                session_context = self.file_intel.build_session_context(session)
                answer = self.file_intel.answer_session_question(message, session_context)
                await self.sessions.add_message(session_id, "assistant", answer)
                
                # Preserve current workflow context in response
                wf_identified = None
                inputs_list = None
                if current_wf.get("workflow_id"):
                    wf_identified = WorkflowIdentified(
                        id=current_wf.get("workflow_id", ""),
                        name=current_wf.get("workflow_name", ""),
                        endpoint=current_wf.get("workflow_endpoint", ""),
                    )
                    required = current_wf.get("required_inputs", [])
                    if required:
                        inputs_list = [
                            InputRequired(
                                field=inp["field"],
                                type=inp.get("type", "str"),
                                label=inp.get("label", ""),
                                collected=inp.get("collected", False),
                            )
                            for inp in required
                        ]
                
                return RouterResponse(
                    session_id=session_id,
                    status=wf_status if wf_status != "idle" else "idle",
                    response=answer,
                    workflow_identified=wf_identified,
                    inputs_required=inputs_list,
                    files_uploaded=files_uploaded if files_uploaded else None,
                    total_session_files=len(session.get("uploaded_files", [])) or None,
                )

        # 6. Handle idle or collecting states
        if wf_status == "idle":
            # No active workflow - match new intent
            return await self._identify_workflow_v2(session_id, message, org_id, files_uploaded)

        elif wf_status in ("collecting", "ready_to_execute"):
            # Continue collecting or execute if ready
            return await self._collect_inputs_v2(
                session_id, message, user_id, org_id, jwt_token, files_uploaded
            )

        # Fallback
        response_text = "I'm not sure how to proceed. Please describe what you'd like to do."
        await self.sessions.add_message(session_id, "assistant", response_text)
        return RouterResponse(
            session_id=session_id,
            status="idle",
            response=response_text,
            files_uploaded=files_uploaded if files_uploaded else None,
        )

    async def handle_chat(
        self,
        message: str,
        session_id: Optional[str],
        user_id: str,
        org_id: str,
        jwt_token: str,
    ) -> RouterResponse:
        """
        Legacy chat handler. Redirects to unified handle_message for consistent behavior.
        Kept for backward compatibility with /api/router/chat endpoint.
        """
        return await self.handle_message(
            message=message,
            session_id=session_id,
            files=[],
            user_id=user_id,
            org_id=org_id,
            jwt_token=jwt_token,
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
        session = await self.sessions.get_session(session_id)
        if not session:
            return RouterResponse(
                status="failed",
                response="Session not found.",
                error=ErrorDetail(code="SESSION_NOT_FOUND", message="Invalid session_id"),
            )

        current_wf = session.get("current_workflow", {})
        wf_status = current_wf.get("status", "idle")
        
        if wf_status != "awaiting_confirmation":
            return RouterResponse(
                session_id=session_id,
                status=wf_status,
                response="No pending confirmation for this session.",
            )

        workflow = {
            "workflowId": current_wf.get("workflow_id"),
            "workflowName": current_wf.get("workflow_name"),
            "workflowEndpoint": current_wf.get("workflow_endpoint"),
            "workflowSchema": current_wf.get("workflow_schema"),
            "workflowApiCalls": current_wf.get("workflow_api_calls"),
        }
        run_id = current_wf.get("run_id", "")
        workflow_id = current_wf.get("workflow_id", "")
        confirmation_data = current_wf.get("confirmation_data", {})

        is_confirmed = action.lower() in ("confirm", "yes", "approve", "proceed")

        if message:
            await self.sessions.add_message(session_id, "user", message)

        request_data = self.builder.build_confirmation_request(
            workflow, run_id, workflow_id, is_confirmed, confirmation_data, jwt_token,
        )

        url = f"{AGENTICAPI_BASE_URL}{request_data['endpoint']}"

        try:
            await self.sessions.update_workflow_status(session_id, "executing")

            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
                resp = await client.post(
                    url,
                    json=request_data["data"],
                    headers=request_data["headers"],
                )
            result = resp.json()

        except Exception as e:
            logger.error(f"Confirmation call failed: {e}")
            await self.sessions.update_workflow_status(session_id, "failed")
            error_msg = "Error processing your confirmation."
            await self.sessions.add_message(session_id, "assistant", error_msg)
            return RouterResponse(
                session_id=session_id,
                status="failed",
                response=error_msg,
                error=ErrorDetail(code="CONFIRMATION_ERROR", message=str(e)),
            )

        if not is_confirmed:
            await self.sessions.complete_workflow(session_id, {"cancelled": True}, status="cancelled")
            cancel_msg = f"{workflow.get('workflowName', 'Workflow')} was cancelled."
            await self.sessions.add_message(session_id, "assistant", cancel_msg)
            return RouterResponse(
                session_id=session_id,
                status="idle",
                response=cancel_msg,
            )

        # Confirmed — handle the result (might be final or another HITL)
        return await self._handle_workflow_response_v2(session_id, workflow, result, run_id)

    async def handle_file_upload(
        self,
        session_id: str,
        files: list,
        field_name: str,
        user_id: str,
        org_id: str,
        jwt_token: str,
    ) -> RouterResponse:
        """
        Legacy file upload handler. Redirects to unified handle_message.
        Kept for backward compatibility with /api/router/upload endpoint.
        """
        return await self.handle_message(
            message="",
            session_id=session_id,
            files=files,
            user_id=user_id,
            org_id=org_id,
            jwt_token=jwt_token,
        )

    # ============== v2 Methods for Multi-Workflow Sessions ==============

    async def _identify_workflow_v2(
        self,
        session_id: str,
        message: str,
        org_id: str,
        files_uploaded: List,
    ) -> RouterResponse:
        """
        Phase 1: Match user intent to a workflow (v2 with multi-workflow support).
        """
        if not message:
            # File uploaded without message
            prompt = "Files uploaded! What would you like to do with them?"
            await self.sessions.add_message(session_id, "assistant", prompt)
            return RouterResponse(
                session_id=session_id,
                status="idle",
                response=prompt,
                files_uploaded=files_uploaded if files_uploaded else None,
            )

        # Try keyword match first
        matched = self.matcher.match_by_keywords(message, org_id)

        # Fall back to Gemini semantic match
        if not matched:
            all_workflows = self.matcher.get_all_workflows(org_id)
            matched = self.gemini.match_intent_to_workflow(message, all_workflows)

        if not matched:
            summaries = self.matcher.get_workflow_summaries(org_id)
            clarification = self.gemini.generate_clarification(message, summaries)
            await self.sessions.add_message(session_id, "assistant", clarification)
            return RouterResponse(
                session_id=session_id,
                status="idle",
                response=clarification,
                files_uploaded=files_uploaded if files_uploaded else None,
            )

        # Workflow found — parse inputs
        required_inputs = self.collector.parse_workflow_inputs(matched)
        required_inputs = self.collector.auto_fill_inputs(
            required_inputs,
            {"user_id": "", "org_id": ""},
        )

        await self.sessions.set_workflow_context(session_id, matched, required_inputs)

        # Try to extract inputs from message and session files
        session = await self.sessions.get_session(session_id)
        missing = self.collector.get_missing_inputs(required_inputs)

        if missing:
            # Extract text inputs from message
            extracted = self.gemini.extract_inputs_from_message(message, missing)
            for field, value in extracted.items():
                if value is not None:
                    await self.sessions.mark_input_collected(session_id, field, value)
            
            # Smart file matching (not blind auto-fill)
            workflow_name = matched.get("workflowName", "")
            for inp in missing:
                if inp.get("type") == "file" and not inp.get("collected"):
                    session_files = session.get("uploaded_files", [])
                    if session_files:
                        best_file = self.file_intel.find_best_file_for_input(
                            session_files, inp, workflow_name
                        )
                        if best_file:
                            file_path = best_file.get("stored_path")
                            if file_path:
                                await self.sessions.mark_input_collected(
                                    session_id, inp["field"], [file_path]
                                )
                                logger.info(
                                    f"✅ Smart-matched '{best_file.get('original_name')}' "
                                    f"to {inp['field']} for '{workflow_name}'"
                                )

        # Re-check missing inputs
        session = await self.sessions.get_session(session_id)
        current_wf = session.get("current_workflow", {})
        updated_inputs = current_wf.get("required_inputs", [])
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
            await self.sessions.add_message(session_id, "assistant", prompt)
            await self.sessions.update_workflow_status(session_id, "collecting")
            
            # Get total session files count
            session_files_count = len(session.get("uploaded_files", []))
            
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
                files_uploaded=files_uploaded if files_uploaded else None,
                total_session_files=session_files_count if session_files_count > 0 else None,
            )

        # All inputs ready - check execution intent
        all_collected = all(inp.get("collected") for inp in updated_inputs)
        should_execute = self.intent.should_auto_execute(message, all_collected)

        if should_execute:
            # User wants to execute immediately
            return await self._execute_workflow_v2(session_id, session)
        else:
            # Ask for confirmation
            inputs_summary = "\n".join([
                f"• {inp.get('label', inp['field'])}: {'✓' if inp.get('collected') else '✗'}"
                for inp in updated_inputs
            ])
            prompt = self.intent.generate_confirmation_prompt(
                matched.get("workflowName", ""),
                inputs_summary,
            )
            await self.sessions.add_message(session_id, "assistant", prompt)
            await self.sessions.update_workflow_status(session_id, "ready_to_execute")
            
            # Get total session files count
            session_files_count = len(session.get("uploaded_files", []))
            
            return RouterResponse(
                session_id=session_id,
                status="ready_to_execute",
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
                files_uploaded=files_uploaded if files_uploaded else None,
                total_session_files=session_files_count if session_files_count > 0 else None,
            )

    async def _collect_inputs_v2(
        self,
        session_id: str,
        message: str,
        user_id: str,
        org_id: str,
        jwt_token: str,
        files_uploaded: List,
    ) -> RouterResponse:
        """
        Phase 2: Collect remaining inputs or execute if ready (v2 with intent detection).
        """
        session = await self.sessions.get_session(session_id)
        current_wf = session.get("current_workflow", {})
        workflow = {
            "workflowId": current_wf.get("workflow_id"),
            "workflowName": current_wf.get("workflow_name"),
            "workflowEndpoint": current_wf.get("workflow_endpoint"),
        }
        required_inputs = current_wf.get("required_inputs", [])
        wf_status = current_wf.get("status", "collecting")

        # Check execution intent
        intent = self.intent.detect_execution_intent(message) if message else "collect"

        if wf_status == "ready_to_execute" and intent == "execute":
            # User confirmed execution
            return await self._execute_workflow_v2(session_id, session)

        elif intent == "delay":
            # User wants to wait/cancel
            response_text = "No problem! Let me know when you're ready to proceed, or if you need to make changes."
            await self.sessions.add_message(session_id, "assistant", response_text)
            return RouterResponse(
                session_id=session_id,
                status="ready_to_execute" if wf_status == "ready_to_execute" else "collecting",
                response=response_text,
                files_uploaded=files_uploaded if files_uploaded else None,
            )

        # Continue collecting inputs
        missing = self.collector.get_missing_inputs(required_inputs)

        if not missing:
            # All inputs collected - ask for confirmation
            inputs_summary = "\n".join([
                f"• {inp.get('label', inp['field'])}: ✓"
                for inp in required_inputs
            ])
            prompt = self.intent.generate_confirmation_prompt(
                workflow.get("workflowName", ""),
                inputs_summary,
            )
            await self.sessions.add_message(session_id, "assistant", prompt)
            await self.sessions.update_workflow_status(session_id, "ready_to_execute")
            
            # Get total session files count
            session_files_count = len(session.get("uploaded_files", []))
            
            return RouterResponse(
                session_id=session_id,
                status="ready_to_execute",
                response=prompt,
                workflow_identified=WorkflowIdentified(
                    id=workflow.get("workflowId", ""),
                    name=workflow.get("workflowName", ""),
                    endpoint=workflow.get("workflowEndpoint", ""),
                ),
                inputs_required=[
                    InputRequired(
                        field=inp["field"],
                        type=inp.get("type", "str"),
                        label=inp.get("label", ""),
                        collected=inp.get("collected", False),
                    )
                    for inp in required_inputs
                ],
                files_uploaded=files_uploaded if files_uploaded else None,
                total_session_files=session_files_count if session_files_count > 0 else None,
            )

        # Try to extract inputs from message
        if message:
            extracted = self.gemini.extract_inputs_from_message(message, missing)
            for field, value in extracted.items():
                if value is not None:
                    await self.sessions.mark_input_collected(session_id, field, value)

        # Smart file matching (not blind auto-fill)
        workflow_name = workflow.get("workflowName", "")
        for inp in missing:
            if inp.get("type") == "file" and not inp.get("collected"):
                session_files = session.get("uploaded_files", [])
                if session_files:
                    best_file = self.file_intel.find_best_file_for_input(
                        session_files, inp, workflow_name
                    )
                    if best_file:
                        file_path = best_file.get("stored_path")
                        if file_path:
                            await self.sessions.mark_input_collected(
                                session_id, inp["field"], [file_path]
                            )
                            logger.info(
                                f"✅ Smart-matched '{best_file.get('original_name')}' "
                                f"to {inp['field']} for '{workflow_name}'"
                            )

        # Refresh and re-check
        session = await self.sessions.get_session(session_id)
        current_wf = session.get("current_workflow", {})
        updated_inputs = current_wf.get("required_inputs", [])
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
            await self.sessions.add_message(session_id, "assistant", prompt)
            
            # Get total session files count
            session_files_count = len(session.get("uploaded_files", []))
            
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
                files_uploaded=files_uploaded if files_uploaded else None,
                total_session_files=session_files_count if session_files_count > 0 else None,
            )

        # All inputs collected - check if user is confirming in THIS message
        if intent == "execute":
            # User confirmed in the same message (e.g., "yes" + file upload)
            return await self._execute_workflow_v2(session_id, session)
        else:
            # Ask for confirmation before executing
            inputs_summary = "\n".join([
                f"• {inp.get('label', inp['field'])}: ✓"
                for inp in updated_inputs
            ])
            prompt = self.intent.generate_confirmation_prompt(
                workflow.get("workflowName", ""),
                inputs_summary,
            )
            await self.sessions.add_message(session_id, "assistant", prompt)
            await self.sessions.update_workflow_status(session_id, "ready_to_execute")
            
            # Get total session files count
            session_files_count = len(session.get("uploaded_files", []))
            
            return RouterResponse(
                session_id=session_id,
                status="ready_to_execute",
                response=prompt,
                workflow_identified=wf_info,
                inputs_required=[
                    InputRequired(
                        field=inp["field"],
                        type=inp.get("type", "str"),
                        label=inp.get("label", ""),
                        collected=True,
                    )
                    for inp in updated_inputs
                ],
                files_uploaded=files_uploaded if files_uploaded else None,
                total_session_files=session_files_count if session_files_count > 0 else None,
            )

    async def _execute_workflow_v2(
        self,
        session_id: str,
        session: Dict[str, Any],
    ) -> RouterResponse:
        """
        Phase 3: Execute workflow and handle result (v2 with multi-workflow support).
        After completion, session returns to idle state for next workflow.
        """
        current_wf = session.get("current_workflow", {})
        workflow = {
            "workflowId": current_wf.get("workflow_id"),
            "workflowName": current_wf.get("workflow_name"),
            "workflowEndpoint": current_wf.get("workflow_endpoint"),
            "workflowSchema": current_wf.get("workflow_schema"),
            "workflowApiCalls": current_wf.get("workflow_api_calls"),
        }
        collected_inputs = current_wf.get("collected_inputs", {})
        required_inputs = current_wf.get("required_inputs", [])
        jwt_token = session.get("jwt_token", "")

        # Strict validation: never execute with missing required inputs
        missing = self.collector.get_missing_inputs(required_inputs)
        if missing:
            missing_labels = [m.get("label", m.get("field")) for m in missing]
            logger.warning(
                f"⚠️ Execution blocked — missing inputs: {missing_labels}"
            )
            prompt = self.gemini.generate_input_prompt(
                workflow.get("workflowName", ""),
                missing[0],
                required_inputs,
            )
            await self.sessions.update_workflow_status(session_id, "collecting")
            await self.sessions.add_message(session_id, "assistant", prompt)
            return RouterResponse(
                session_id=session_id,
                status="collecting",
                response=prompt,
                workflow_identified=WorkflowIdentified(
                    id=workflow.get("workflowId", ""),
                    name=workflow.get("workflowName", ""),
                    endpoint=workflow.get("workflowEndpoint", ""),
                ),
                inputs_required=[
                    InputRequired(
                        field=inp["field"],
                        type=inp.get("type", "str"),
                        label=inp.get("label", ""),
                        collected=inp.get("collected", False),
                    )
                    for inp in required_inputs
                ],
            )

        run_id = str(uuid.uuid4())
        await self.sessions.set_run_id(session_id, run_id)
        await self.sessions.update_workflow_status(session_id, "executing")

        # Build request
        request_data = self.builder.build_workflow_request(
            workflow, collected_inputs, required_inputs, run_id, jwt_token,
        )

        endpoint = request_data["endpoint"]
        url = f"{AGENTICAPI_BASE_URL}{endpoint}"

        # Execute workflow
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
                if request_data["content_type"] == "multipart/form-data":
                    form_data = {}
                    for key, value in request_data["data"].items():
                        form_data[key] = str(value) if not isinstance(value, str) else value

                    # Build file upload list with original filenames
                    # Look up original_name from session.uploaded_files for each stored_path
                    session_files = session.get("uploaded_files", [])
                    file_path_to_name = {
                        f.get("stored_path"): f.get("original_name", os.path.basename(f.get("stored_path", "")))
                        for f in session_files
                    }
                    
                    files_list = []
                    for field_name, file_info in request_data.get("file_fields", {}).items():
                        endpoint_name = file_info.get("endpoint_name", field_name)
                        file_paths = file_info.get("paths", [])
                        for fp in file_paths:
                            if os.path.exists(fp):
                                # Use original filename from metadata, not stored filename
                                original_name = file_path_to_name.get(fp, os.path.basename(fp))
                                files_list.append(
                                    (endpoint_name, (original_name, open(fp, "rb")))
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
            await self.sessions.update_workflow_status(session_id, "failed")
            
            # Move to history and reset to idle
            await self.sessions.complete_workflow(session_id, {"error": str(e)}, status="failed")
            
            error_msg = "The workflow encountered an error during execution."
            await self.sessions.add_message(session_id, "assistant", error_msg)
            return RouterResponse(
                session_id=session_id,
                status="idle",  # Back to idle for next workflow
                response=error_msg,
                error=ErrorDetail(code="EXECUTION_ERROR", message=str(e)),
            )

        # Handle result
        return await self._handle_workflow_response_v2(session_id, workflow, result, run_id)

    async def _handle_workflow_response_v2(
        self,
        session_id: str,
        workflow: Dict[str, Any],
        result: Any,
        run_id: str,
    ) -> RouterResponse:
        """
        Handle workflow response (v2 with multi-workflow support).
        After completion, workflow moves to history and session returns to idle.
        """
        workflow_name = workflow.get("workflowName", "Workflow")
        workflow_id = workflow.get("workflowId", "")

        # Detect HITL
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
            # Workflow paused for confirmation
            await self.sessions.update_workflow_status(
                session_id,
                "awaiting_confirmation",
                confirmation_data=result,
            )

            hitl_prompt = self.gemini.format_hitl_prompt(result, workflow_name)
            await self.sessions.add_message(session_id, "assistant", hitl_prompt)

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
                    step_number=result.get("stepNumber", 0),
                    data_to_review=result,
                ),
            )

        # Workflow completed — mark files as used and move to history
        session = await self.sessions.get_session(session_id)
        current_wf = session.get("current_workflow", {})
        collected_inputs = current_wf.get("collected_inputs", {})
        
        # Find file paths used by this workflow
        used_file_paths = []
        for field, value in collected_inputs.items():
            if isinstance(value, list):
                used_file_paths.extend(v for v in value if isinstance(v, str) and os.path.exists(v))
            elif isinstance(value, str) and os.path.exists(value):
                used_file_paths.append(value)
        
        # Mark files as used
        if used_file_paths:
            session_files = session.get("uploaded_files", [])
            updated_files = self.file_intel.mark_files_used(
                session_files, used_file_paths, workflow_name
            )
            await self.sessions.update_files(session_id, updated_files)
        
        # Generate result summary for future context/question answering
        result_summary = self.gemini.generate_result_summary(result, workflow_name)
        
        await self.sessions.complete_workflow(
            session_id, result, status="completed", result_summary=result_summary
        )
        
        summary = self.gemini.format_final_result(result, workflow_name)
        await self.sessions.add_message(session_id, "assistant", summary)

        return RouterResponse(
            session_id=session_id,
            status="idle",  # Back to idle for next workflow!
            response=summary + "\n\nWhat would you like to do next?",
            workflow_identified=WorkflowIdentified(
                id=workflow_id,
                name=workflow_name,
                endpoint=workflow.get("workflowEndpoint", ""),
            ),
            final_result=result,
        )

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details for the GET endpoint."""
        return self.sessions.get_session_info(session_id)

    async def delete_session(self, session_id: str, user_id: str) -> bool:
        """Delete a session (only if it belongs to the user)."""
        return await self.sessions.delete_session(session_id, user_id)

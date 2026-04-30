"""
Main Orchestrator for the LLM Router.
Connects all router components: session management, workflow matching,
input collection, request building, file handling, and Gemini LLM.
Makes internal HTTP calls to the AgenticAPI workflow endpoints.

Pinecone Integration:
- Retrieves semantic context from chat history and file content
- Enhances workflow matching and input collection with long-term memory
- Enables RAG-style question answering using Pinecone vector search
"""
import os
import uuid
import logging
import httpx
from datetime import datetime
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

from core.azure_openai_client import AzureOpenAIClient
from core.session_manager import SessionManager
from core.workflow_matcher import WorkflowMatcher
from core.file_intelligence import FileIntelligence
from core.result_analyzer import ResultAnalyzer
from core.markdown_enhancer import MarkdownEnhancer
from handlers.input_collector import InputCollector
from handlers.request_builder import RequestBuilder
from handlers.file_handler import FileHandler
from utils.intent_detector import IntentDetector
from components.PineconeClient import pinecone_client
from components.KeyVaultClient import get_secret
from components.FileMiddleware import file_middleware

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
AGENTICAPI_BASE_URL = get_secret("AGENTICAPI_BASE_URL", default=os.getenv("AGENTICAPI_BASE_URL", "http://localhost:8400"))


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
        self.gemini = AzureOpenAIClient()
        self.sessions = SessionManager()
        self.matcher = WorkflowMatcher()
        self.collector = InputCollector()
        self.builder = RequestBuilder()
        self.files = FileHandler()
        self.intent = IntentDetector()
        self.file_intel = FileIntelligence()
        self.file_middleware = file_middleware
        
        # Idle workflow components
        from core.idle_workflow_manager import IdleWorkflowManager
        from core.file_classifier import FileClassifier
        self.idle_manager = IdleWorkflowManager()
        self.file_classifier = FileClassifier()
        
        # New components for rich markdown responses
        self.result_analyzer = ResultAnalyzer(self.gemini)
        self.markdown_enhancer = MarkdownEnhancer()
    
    def _enhance_response(self, structured_response, context: Optional[Dict[str, Any]] = None):
        """
        Build perfect markdown from structured LLM response.
        Uses MarkdownEnhancer to convert StructuredResponse to markdown string.
        """
        # If already a string, return as-is (no enhancement needed)
        if isinstance(structured_response, str):
            return structured_response
        
        # Try to enhance structured response
        try:
            return self.markdown_enhancer.build_from_structured(structured_response, context)
        except Exception as e:
            logger.warning(f"Markdown building failed: {e}")
            return "Response formatting failed. Please try again."
    
    def _format_value_for_display(self, value: Any, input_type: str) -> str:
        """Format collected input value for safe LLM display."""
        if input_type in ("file", "files"):
            if isinstance(value, list):
                count = len(value)
                return f"{count} file{'s' if count != 1 else ''}"
            return "1 file"
        elif value == "__auto_runid__":
            return "[auto-generated]"
        elif value == "__hitl_field__":
            return "[confirmation pending]"
        elif value is None:
            return "[not collected]"
        else:
            str_val = str(value)
            return str_val[:100] + "..." if len(str_val) > 100 else str_val
    
    async def _ensure_collected_inputs_synced(self, session_id: str) -> bool:
        """
        SAFETY NET: Detect and fix state inconsistency where collected_inputs is empty
        but required_inputs has collected items.
        
        This prevents data loss from bugs or edge cases in state management.
        
        Returns:
            True if state was fixed, False if already consistent
        """
        session = await self.sessions.get_session(session_id)
        if not session:
            return False
        
        current_wf = session.get("current_workflow", {})
        collected_inputs = current_wf.get("collected_inputs", {})
        required_inputs = current_wf.get("required_inputs", [])
        
        # Check if collected_inputs is empty but required_inputs has collected items
        has_collected_items = any(inp.get("collected") for inp in required_inputs)
        
        if not collected_inputs and has_collected_items:
            logger.warning(f"⚠️ STATE INCONSISTENCY DETECTED in session {session_id[:8]}... - rebuilding collected_inputs")
            
            # Rebuild collected_inputs from required_inputs
            rebuilt = {}
            for inp in required_inputs:
                if inp.get("collected") and "value" in inp:
                    rebuilt[inp["field"]] = inp["value"]
            
            # Update session
            await self.sessions.update_session(session_id, {
                "current_workflow.collected_inputs": rebuilt
            })
            
            logger.info(f"✅ State recovered: {len(rebuilt)} collected inputs restored")
            return True
        
        return False

    async def handle_message(
        self,
        message: str,
        session_id: Optional[str],
        files: List = None,
        file_ids: List[str] = None,
        hitl_request: Optional[Dict[str, Any]] = None,
        user_id: str = "",
        org_id: str = "",
        jwt_token: str = "",
    ) -> RouterResponse:
        """
        Unified message handler for conversational workflow orchestration.
        Handles: chat messages, file uploads, file_ids, HITL confirmation, intent detection, multi-workflow sessions.
        
        Key Features:
        - Accepts both raw files and file_ids
        - Files uploaded via middleware → file_ids stored in session
        - File_ids validated via middleware → stored in session
        - Accepts hitl_request (edited HITL data from frontend) for confirmation flow
        - Explicit execution intent required
        - Multi-workflow support
        - Session persists across workflows
        """
        # Ensure defaults
        files = files or []
        file_ids = file_ids or []
        
        # 1. Validate session_id is provided
        if not session_id:
            return RouterResponse(
                status="failed",
                response="Session ID is required. Frontend must provide X-Session-Id header.",
                error=ErrorDetail(code="MISSING_SESSION_ID", message="session_id is required"),
            )
        
        # 2. Get or create session with provided ID
        session = await self.sessions.get_session(session_id)
        if not session:
            session = self.sessions.create_session(session_id, user_id, org_id, jwt_token)
            logger.info(f"Created new session with frontend ID: {session_id[:8]}...")
        else:
            await self.sessions.update_jwt_token(session_id, jwt_token)

        # 3A. Handle raw file uploads → Upload to middleware → Get file_ids
        from datetime import datetime
        files_uploaded = []
        new_file_ids = []
        
        if files and len(files) > 0:
            uploaded = await self.file_middleware.upload_files(
                files=files,
                user_id=user_id,
                org_id=org_id,
                session_id=session_id
            )
            
            new_file_ids = [f["file_id"] for f in uploaded]
            files_uploaded = [
                FileUploaded(
                    file_id=f["file_id"],
                    original_name=f["original_name"],
                    mime_type=f["mime_type"],
                    size_bytes=f["size_bytes"],
                    uploaded_at=datetime.utcnow().isoformat(),
                )
                for f in uploaded
            ]
            logger.info(f"📤 {len(files)} file(s) uploaded → middleware → {len(new_file_ids)} file_ids")
        
        # 3B. Handle file_ids → Validate via middleware
        validated_file_ids = []
        if file_ids and len(file_ids) > 0:
            validated = self.file_middleware.validate_file_ids(
                file_ids=file_ids,
                user_id=user_id,
                org_id=org_id
            )
            validated_file_ids = validated
            logger.info(f"🔗 {len(validated)} file_ids validated for session")
        
        # 3C. Combine all file_ids and add to session
        all_file_ids = new_file_ids + validated_file_ids
        if all_file_ids:
            await self.sessions.add_file_ids(session_id, all_file_ids)
            # Force cache clear to ensure fresh read from MongoDB
            # Use the proper delete_session method which has built-in error handling
            deleted = await self.sessions.redis.delete_session(session_id)
            if deleted:
                logger.debug(f"🗑️ Cleared Redis cache for session: {session_id[:8]}...")

        # 3D. PRIORITY: Try to match files to idle workflows FIRST
        #     This handles files uploaded for workflows awaiting inputs
        if all_file_ids:
            temp_session = await self.sessions.get_session(session_id)
            idle_workflows_summary = temp_session.get("idle_workflows", [])
            
            if idle_workflows_summary:
                logger.info(f"🔍 {len(all_file_ids)} file(s) uploaded, {len(idle_workflows_summary)} idle workflows detected")
                classification_result = await self._handle_file_classification(
                    session_id=session_id,
                    new_file_ids=all_file_ids,
                    idle_workflows_summary=idle_workflows_summary,
                    conversation_context=temp_session.get("conversation_history", [])
                )
                
                # If files were matched to idle workflows, return classification result
                if classification_result and classification_result.get("matched_count", 0) > 0:
                    return classification_result["response"]

        # 3E. Add message to conversation (if provided)
        if message:
            await self.sessions.add_message(session_id, "user", message)

        # 4. Refresh session to get latest state (fetches from MongoDB if cache was cleared)
        session = await self.sessions.get_session(session_id)
        current_wf = session.get("current_workflow", {})
        wf_status = current_wf.get("status", "idle")

        # 4b. Immediately match newly uploaded files to active workflow inputs
        #     Runs BEFORE any intent routing so files are matched regardless.
        #     IMPORTANT: Only assign newly uploaded files (all_file_ids), not all session file_ids
        if current_wf.get("workflow_id") and wf_status in ("collecting", "ready_to_execute") and all_file_ids:
            await self._recheck_all_inputs(session_id, session, new_file_ids=all_file_ids)
            session = await self.sessions.get_session(session_id)
            current_wf = session.get("current_workflow", {})
            wf_status = current_wf.get("status", "idle")

        # 4b. Check if organization has any workflows registered
        if message and wf_status == "idle":
            all_workflows = self.matcher.get_all_workflows(org_id)
            if not all_workflows:
                no_wf_msg = self.gemini.generate_contextual_response(
                    "Organization has no registered workflows yet. Inform user they need workflow registration.",
                    {"org_id": org_id}
                )
                await self.sessions.add_message(session_id, "assistant", no_wf_msg)
                return RouterResponse(
                    session_id=session_id,
                    status="no_workflows",
                    response=no_wf_msg,
                    error=ErrorDetail(
                        code="NO_WORKFLOWS_REGISTERED",
                        message="No workflows available for this organization"
                    ),
                    files_uploaded=files_uploaded if files_uploaded else None,
                )
        
        # 5. State machine routing
        if wf_status == "awaiting_confirmation":
            if message:
                # ═══════════════════════════════════════════════════════════════
                # PRIORITY 1: Check if user is responding to pending workflow switch
                # This must be checked FIRST to avoid ambiguity with HITL confirmation
                # ═══════════════════════════════════════════════════════════════
                pending_switch = session.get("pending_workflow_switch")
                
                if pending_switch and pending_switch.get("from_state") == "awaiting_confirmation":
                    intent = self.intent.detect_execution_intent(message)
                    logger.info(f"🎯 Pending switch detected. User intent: {intent}")
                    
                    if intent == "execute":
                        # User confirmed the switch
                        logger.info(f"✅ Switch confirmed: {pending_switch['from_workflow_id'][:8]}... → {pending_switch['to_workflow']['workflowName']}")
                        
                        # Extract triggering message for input extraction
                        triggering_message = pending_switch.get("triggering_message")
                        
                        # Clear pending switch
                        await self.sessions.update_session(session_id, {"pending_workflow_switch": None})
                        
                        # Use universal switch method (resumes from idle if exists)
                        # ✅ Pass triggering message for input extraction
                        return await self._switch_to_workflow(
                            session_id=session_id,
                            target_workflow=pending_switch["to_workflow"],
                            user_id=user_id,
                            org_id=org_id,
                            jwt_token=jwt_token,
                            files_uploaded=files_uploaded,
                            save_current=False,  # Already saved when switch was initiated
                            current_wf=None,
                            triggering_message=triggering_message  # ✅ NEW: Pass message for input extraction
                        )
                    
                    elif intent == "delay":
                        # User cancelled the switch - restore HITL workflow from idle
                        logger.info(f"🛑 Switch cancelled. Restoring HITL workflow from idle.")
                        
                        # Find the idle workflow we just saved (most recent for this session with HITL status)
                        idle_workflows = await self.idle_manager.get_idle_workflows(session_id=session_id)
                        hitl_idle = None
                        for wf in idle_workflows:
                            if wf.get("workflow_id") == pending_switch["from_workflow_id"]:
                                hitl_idle = wf
                                break
                        
                        if hitl_idle:
                            # Restore the HITL workflow
                            logger.info(f"🔄 Restoring HITL workflow: {hitl_idle['workflow_name']}")
                            
                            # Use the resume logic to restore complete state
                            await self.sessions.update_session(session_id, {"pending_workflow_switch": None})
                            
                            return await self._resume_idle_workflow(
                                session_id=session_id,
                                instance_id=hitl_idle["instance_id"],
                                workflow_name=hitl_idle["workflow_name"],
                                user_id=user_id,
                                org_id=org_id,
                                jwt_token=jwt_token,
                                files_uploaded=files_uploaded
                            )
                        else:
                            # Fallback: just clear pending switch and continue
                            await self.sessions.update_session(session_id, {"pending_workflow_switch": None})
                            response = "Switch cancelled. Please confirm or cancel the current workflow."
                            await self.sessions.add_message(session_id, "assistant", response)
                            
                            return RouterResponse(
                                session_id=session_id,
                                status="awaiting_confirmation",
                                response=response,
                                requires_confirmation=True,
                                files_uploaded=files_uploaded if files_uploaded else None,
                            )
                
                # ═══════════════════════════════════════════════════════════════
                # PRIORITY 2: Check for resume intent (switch to idle workflow)
                # ═══════════════════════════════════════════════════════════════
                idle_workflows = await self.idle_manager.get_idle_workflows(session_id=session_id)
                resume_match = await self._detect_resume_intent(message, idle_workflows) if idle_workflows else None
                
                if resume_match:
                    # User wants to resume another workflow while in HITL
                    logger.info(f"🔄 Resume request during HITL: {current_wf.get('workflow_name')} → {resume_match['workflow_name']}")
                    
                    # Save current HITL workflow to idle
                    current_workflow_idle_id = await self._save_current_workflow_to_idle(
                        session_id=session_id,
                        current_wf=current_wf,
                        user_id=user_id,
                        org_id=org_id
                    )
                    
                    if current_workflow_idle_id:
                        logger.info(f"💤 HITL workflow saved as idle: {current_workflow_idle_id[:8]}...")
                    
                    # Resume the requested workflow
                    return await self._resume_idle_workflow(
                        session_id=session_id,
                        instance_id=resume_match["instance_id"],
                        workflow_name=resume_match["workflow_name"],
                        user_id=user_id,
                        org_id=org_id,
                        jwt_token=jwt_token,
                        files_uploaded=files_uploaded
                    )
                
                # ═══════════════════════════════════════════════════════════════
                # PRIORITY 3: Check for new workflow intent
                # ═══════════════════════════════════════════════════════════════
                new_workflow_match = self.matcher.match_by_keywords(message, org_id)
                if not new_workflow_match:
                    all_workflows = self.matcher.get_all_workflows(org_id)
                    new_workflow_match = self.gemini.match_intent_to_workflow(message, all_workflows)
                
                if new_workflow_match and new_workflow_match.get("workflowId") != current_wf.get("workflow_id"):
                    logger.info(f"🔍 New workflow detected during HITL: {new_workflow_match.get('workflowName')}")
                    
                    # Save current HITL workflow to idle
                    current_workflow_idle_id = await self._save_current_workflow_to_idle(
                        session_id=session_id,
                        current_wf=current_wf,
                        user_id=user_id,
                        org_id=org_id
                    )
                    
                    if current_workflow_idle_id:
                        logger.info(f"💤 HITL workflow saved with instance_id: {current_workflow_idle_id[:8]}...")
                    
                    # Store pending switch with idle workflow reference
                    # ✅ OPTION 3: Store triggering message for input extraction after confirmation
                    await self.sessions.update_session(session_id, {
                        "pending_workflow_switch": {
                            "from_workflow_id": current_wf.get("workflow_id"),
                            "from_idle_instance_id": current_workflow_idle_id,  # Store idle reference
                            "to_workflow": new_workflow_match,
                            "detected_at": datetime.utcnow().isoformat(),
                            "from_state": "awaiting_confirmation",
                            "awaiting_user_confirmation": True,  # Flag to indicate pending confirmation
                            "triggering_message": message  # ✅ Store message for later input extraction
                        }
                    })
                    
                    # Ask for confirmation
                    switch_prompt = self.gemini.generate_contextual_response(
                        f"User wants to switch from '{current_wf.get('workflow_name')}' (awaiting confirmation) to '{new_workflow_match.get('workflowName')}'. Ask if they want to pause current and start new. Mention current workflow will be saved.",
                        {"will_pause": True, "current_in_hitl": True}
                    )
                    await self.sessions.add_message(session_id, "assistant", switch_prompt)
                    
                    return RouterResponse(
                        session_id=session_id,
                        status="awaiting_confirmation",
                        response=switch_prompt,
                        requires_confirmation=True,
                        files_uploaded=files_uploaded if files_uploaded else None,
                    )
                
                # ═══════════════════════════════════════════════════════════════
                # PRIORITY 4: Normal HITL confirmation (only if no pending actions)
                # ═══════════════════════════════════════════════════════════════
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
                        message=message,
                        hitl_request=hitl_request,
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
            
            # No clear intent - generate dynamic waiting prompt
            wf_name = current_wf.get("workflow_name", "the workflow")
            response_text = self.gemini.generate_contextual_response(
                f"User sent a message during HITL confirmation for '{wf_name}'. Remind them to confirm or cancel.",
                {"workflow_name": wf_name},
            )
            await self.sessions.add_message(session_id, "assistant", response_text)
            return RouterResponse(
                session_id=session_id,
                status="awaiting_confirmation",
                response=response_text,
                requires_confirmation=True,
                files_uploaded=files_uploaded if files_uploaded else None,
            )

        if wf_status == "executing":
            wf_name = current_wf.get("workflow_name", "the workflow")
            response_text = self.gemini.generate_contextual_response(
                f"'{wf_name}' is currently running. Ask the user to wait.",
                {"workflow_name": wf_name},
            )
            return RouterResponse(
                session_id=session_id,
                status="executing",
                response=response_text,
                files_uploaded=files_uploaded if files_uploaded else None,
            )

        # 5b. Handle conversational questions (RAG-style with Pinecone)
        if message:
            msg_intent = self.intent.detect_execution_intent(message)
            if msg_intent == "question":
                # Build traditional session context
                session_context = self.file_intel.build_session_context(session)
                
                # Enhance with Pinecone semantic context
                pinecone_context = self._get_pinecone_context(
                    query=message,
                    session_id=session_id,
                    org_id=org_id,
                    top_k=5,
                )
                session_context["pinecone_context"] = pinecone_context
                
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
                    total_session_files=len(session.get("file_ids", [])) or None,
                )

        # 5c. Handle rerun requests (before new workflow matching)
        if message and wf_status == "idle":
            rerun_result = await self._handle_rerun_request(
                session_id, session, message, user_id, org_id, jwt_token, files_uploaded
            )
            if rerun_result:
                return rerun_result

        # 6. Handle idle or collecting states
        if wf_status == "idle":
            # Check if user wants to resume an idle workflow before matching new workflow
            if message:
                idle_workflows = await self.idle_manager.get_idle_workflows(session_id=session_id)
                if idle_workflows:
                    # Check if message indicates resuming an idle workflow
                    resume_match = await self._detect_resume_intent(message, idle_workflows)
                    if resume_match:
                        return await self._resume_idle_workflow(
                            session_id=session_id,
                            instance_id=resume_match["instance_id"],
                            workflow_name=resume_match["workflow_name"],
                            user_id=user_id,
                            org_id=org_id,
                            jwt_token=jwt_token,
                            files_uploaded=files_uploaded
                        )
            
            # No active workflow - match new intent
            return await self._identify_workflow_v2(session_id, message, user_id, org_id, files_uploaded)

        elif wf_status in ("collecting", "ready_to_execute"):
            # WORKFLOW SWITCHING: Check if user is trying to start a NEW workflow
            # before continuing with input collection for current workflow
            if message:
                # First check if there's a pending workflow switch
                pending_switch = session.get("pending_workflow_switch")
                
                if pending_switch:
                    logger.info(f"📋 Pending workflow switch detected: {pending_switch.get('from_workflow_id', '')[:8]}... → {pending_switch.get('to_workflow', {}).get('workflowName', '')}")
                
                # Try to match the message to a different workflow
                new_workflow_match = self.matcher.match_by_keywords(message, org_id)
                if not new_workflow_match:
                    all_workflows = self.matcher.get_all_workflows(org_id)
                    new_workflow_match = self.gemini.match_intent_to_workflow(message, all_workflows)
                
                # Log workflow matching attempt
                if new_workflow_match:
                    logger.info(f"🔍 Detected workflow intent: {new_workflow_match.get('workflowName')} (current: {current_wf.get('workflow_name')})")
                
                # If matched a different workflow, ask user if they want to switch
                if new_workflow_match and new_workflow_match.get("workflowId") != current_wf.get("workflow_id"):
                    current_wf_name = current_wf.get("workflow_name", "current workflow")
                    new_wf_name = new_workflow_match.get("workflowName", "new workflow")
                    
                    # Generate switching confirmation message (emphasize workflow will be paused, not canceled)
                    switch_prompt = self.gemini.generate_contextual_response(
                        f"User wants to switch from '{current_wf_name}' to '{new_wf_name}'. Ask if they want to pause current workflow and start new one. Mention they can return to paused workflow later.",
                        {
                            "current_workflow": current_wf_name,
                            "new_workflow": new_wf_name,
                            "status": wf_status,
                            "will_pause": True  # Emphasize it's a pause, not cancel
                        }
                    )
                    
                    await self.sessions.add_message(session_id, "assistant", switch_prompt)
                    
                    # Store the pending workflow switch in session for next interaction
                    # ✅ OPTION 3: Store triggering message for input extraction after confirmation
                    await self.sessions.update_session(session_id, {
                        "pending_workflow_switch": {
                            "from_workflow_id": current_wf.get("workflow_id"),
                            "to_workflow": new_workflow_match,
                            "detected_at": datetime.utcnow().isoformat(),
                            "triggering_message": message  # ✅ Store message for later input extraction
                        }
                    })
                    
                    return RouterResponse(
                        session_id=session_id,
                        status=wf_status,
                        response=switch_prompt,
                        workflow_identified=WorkflowIdentified(
                            id=current_wf.get("workflow_id", ""),
                            name=current_wf_name,
                            endpoint=current_wf.get("workflow_endpoint", ""),
                        ),
                        files_uploaded=files_uploaded if files_uploaded else None,
                    )
                
                # Check if user is confirming a pending workflow switch
                pending_switch = session.get("pending_workflow_switch")
                if pending_switch:
                    intent = self.intent.detect_execution_intent(message)
                    logger.info(f"🎯 Intent detection for '{message}': {intent} (pending switch exists)")
                    
                    if intent == "execute":
                        # User confirmed the switch
                        logger.info(f"✅ Workflow switch confirmed: {pending_switch['from_workflow_id']} → {pending_switch['to_workflow']['workflowId']}")
                        
                        # Extract triggering message for input extraction
                        triggering_message = pending_switch.get("triggering_message")
                        
                        # Clear pending switch
                        await self.sessions.update_session(session_id, {"pending_workflow_switch": None})
                        
                        # Use universal switch method (resumes from idle if exists)
                        # ✅ Pass triggering message for input extraction
                        return await self._switch_to_workflow(
                            session_id=session_id,
                            target_workflow=pending_switch["to_workflow"],
                            user_id=user_id,
                            org_id=org_id,
                            jwt_token=jwt_token,
                            files_uploaded=files_uploaded,
                            save_current=True,
                            current_wf=current_wf,
                            triggering_message=triggering_message  # ✅ NEW: Pass message for input extraction
                        )
                    
                    elif intent == "delay":
                        # User canceled the switch
                        logger.info(f"🛑 Workflow switch canceled")
                        await self.sessions.update_session(session_id, {"pending_workflow_switch": None})
                        
                        # Continue with current workflow
                        response = self.gemini.generate_contextual_response(
                            f"Continuing with '{current_wf.get('workflow_name', 'current workflow')}'.",
                            {"workflow_name": current_wf.get("workflow_name", "")},
                        )
                        await self.sessions.add_message(session_id, "assistant", response)
                        
                        return RouterResponse(
                            session_id=session_id,
                            status=wf_status,
                            response=response,
                            workflow_identified=WorkflowIdentified(
                                id=current_wf.get("workflow_id", ""),
                                name=current_wf.get("workflow_name", ""),
                                endpoint=current_wf.get("workflow_endpoint", ""),
                            ),
                            files_uploaded=files_uploaded if files_uploaded else None,
                        )
            
            # Continue collecting or execute if ready
            return await self._collect_inputs_v2(
                session_id, message, user_id, org_id, jwt_token, files_uploaded
            )

        # Fallback
        file_ids = session.get("file_ids", [])
        response_text = self.gemini.generate_contextual_response(
            "User is idle with no active workflow. Ask what they'd like to do.",
            {"files_count": len(file_ids)},
        )
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
        hitl_request: Optional[Dict[str, Any]] = None,
    ) -> RouterResponse:
        """Handle HITL confirmation (confirm/cancel)."""
        session = await self.sessions.get_session(session_id)
        if not session:
            return RouterResponse(
                status="failed",
                response="Session not found.",
                error=ErrorDetail(code="SESSION_NOT_FOUND", message="Invalid session_id"),
            )
        
        # SAFETY NET: Ensure collected_inputs is synced before confirmation
        await self._ensure_collected_inputs_synced(session_id)
        
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

        # ✅ FIX: Fallback to stored confirmation_data if no explicit hitl_request from frontend
        if not hitl_request and confirmation_data:
            # Extract hitl_request from stored confirmation_data
            if isinstance(confirmation_data, dict) and "hitl_request" in confirmation_data:
                hitl_request = confirmation_data["hitl_request"]
                logger.info(f"🔄 Using stored confirmation_data.hitl_request (no explicit edit from frontend)")
            else:
                # Fallback: use entire confirmation_data as hitl_request
                hitl_request = confirmation_data
                logger.info(f"🔄 Using entire confirmation_data as hitl_request (no explicit edit from frontend)")

        request_data = self.builder.build_confirmation_request(
            workflow, run_id, workflow_id, is_confirmed, jwt_token, session_id, hitl_request,
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
            
            # Cleanup: Delete any active idle workflows for this session
            await self.idle_manager.delete_active_workflows(session_id)
            
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

    async def _handle_rerun_request(
        self,
        session_id: str,
        session: Dict[str, Any],
        message: str,
        user_id: str,
        org_id: str,
        jwt_token: str,
        files_uploaded: Optional[List] = None,
    ) -> Optional[RouterResponse]:
        """
        Detect and handle workflow rerun requests.
        Returns RouterResponse if rerun detected, None otherwise.
        """
        message_lower = message.lower()
        
        # Detect rerun intent
        rerun_keywords = ["rerun", "run again", "redo", "repeat", "do it again", "execute again"]
        if not any(keyword in message_lower for keyword in rerun_keywords):
            return None
        
        workflow_history = session.get("workflow_history", [])
        if not workflow_history:
            response_msg = "No previous workflows to rerun. Please start a new workflow."
            await self.sessions.add_message(session_id, "assistant", response_msg)
            return RouterResponse(
                session_id=session_id,
                status="idle",
                response=response_msg,
                files_uploaded=files_uploaded,
            )
        
        # Determine which workflow to rerun
        target_workflow = None
        
        # Check for "last" or "previous"
        if "last" in message_lower or "previous" in message_lower:
            target_workflow = workflow_history[-1]
        else:
            # Try to match workflow name from history
            for wf in reversed(workflow_history):
                wf_name_lower = wf.get("workflow_name", "").lower()
                if wf_name_lower and wf_name_lower in message_lower:
                    target_workflow = wf
                    break
            
            # If no match, use last workflow
            if not target_workflow:
                target_workflow = workflow_history[-1]
        
        # Check if workflow completed successfully
        if target_workflow.get("status") != "completed":
            response_msg = f"Cannot rerun '{target_workflow.get('workflow_name')}' - it {target_workflow.get('status')}. Only completed workflows can be rerun."
            await self.sessions.add_message(session_id, "assistant", response_msg)
            return RouterResponse(
                session_id=session_id,
                status="idle",
                response=response_msg,
                files_uploaded=files_uploaded,
            )
        
        logger.info(f"🔄 Rerun detected for: {target_workflow.get('workflow_name')}")
        
        # Restore file_ids from previous run
        file_ids_used = target_workflow.get("file_ids_used", [])
        if file_ids_used:
            await self.sessions.add_file_ids(session_id, file_ids_used)
            logger.info(f"📎 Restored {len(file_ids_used)} file_ids for rerun")
        
        # Fetch workflow details from registry
        workflow_id = target_workflow.get("workflow_id")
        workflows = self.matcher.get_all_workflows(org_id)
        matched_workflow = None
        for wf in workflows:
            if wf.get("workflowId") == workflow_id:
                matched_workflow = wf
                break
        
        if not matched_workflow:
            response_msg = f"Workflow '{target_workflow.get('workflow_name')}' not found in registry."
            await self.sessions.add_message(session_id, "assistant", response_msg)
            return RouterResponse(
                session_id=session_id,
                status="idle",
                response=response_msg,
                error=ErrorDetail(code="WORKFLOW_NOT_FOUND", message="Workflow no longer available"),
                files_uploaded=files_uploaded,
            )
        
        # Set up workflow context with restored inputs
        required_inputs = self.collector.parse_workflow_inputs(matched_workflow)
        required_inputs = self.collector.auto_fill_inputs(
            required_inputs,
            {"user_id": user_id, "org_id": org_id},
        )
        await self.sessions.set_workflow_context(session_id, matched_workflow, required_inputs)
        
        # Restore collected inputs from previous run
        previous_inputs = target_workflow.get("collected_inputs", {})
        if previous_inputs:
            # Mark inputs as collected and set values
            for field_name, value in previous_inputs.items():
                await self.sessions.mark_input_collected(session_id, field_name, value)
            
            logger.info(f"🔄 Restored {len(previous_inputs)} input(s) from previous run")
        
        # Mark workflow as ready to execute (skip collection phase)
        await self.sessions.update_workflow_status(session_id, "ready_to_execute")
        
        # Refresh session and proceed to execution
        session = await self.sessions.get_session(session_id)
        
        rerun_msg = f"Rerunning '{target_workflow.get('workflow_name')}' with previous inputs and files..."
        await self.sessions.add_message(session_id, "assistant", rerun_msg)
        
        # Execute the workflow
        return await self._execute_workflow_v2(session_id, session)

    # ============== v2 Methods for Multi-Workflow Sessions ==============

    async def _identify_workflow_v2(
        self,
        session_id: str,
        message: str,
        user_id: str,
        org_id: str,
        files_uploaded: List,
    ) -> RouterResponse:
        """
        Phase 1: Match user intent to a workflow (v2 with multi-workflow support).
        Uses Pinecone context when available to enhance workflow matching.
        """
        session = await self.sessions.get_session(session_id)
        
        if not message:
            # File uploaded without message - prompt for intent
            file_ids = session.get("file_ids", [])
            file_count = len(file_ids)
            
            if files_uploaded:
                file_names = [f.get("original_name", "") for f in files_uploaded]
                prompt = self.gemini.generate_contextual_response(
                    f"User uploaded {file_count} file(s) without a message. Ask what they'd like to do with these files.",
                    {"files": file_names},
                )
            else:
                prompt = self.gemini.generate_contextual_response(
                    "User uploaded files without a message. Ask what they'd like to do.",
                    {"files_count": file_count},
                )
            
            await self.sessions.add_message(session_id, "assistant", prompt)
            return RouterResponse(
                session_id=session_id,
                status="idle",
                response=prompt,
                files_uploaded=files_uploaded if files_uploaded else None,
                total_session_files=file_count if file_count > 0 else None,
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
            # Enhance markdown
            clarification = self._enhance_response(clarification)
            
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
            {"user_id": user_id, "org_id": org_id},
        )

        await self.sessions.set_workflow_context(session_id, matched, required_inputs)

        # Auto-collect file inputs using file_ids from session (if any files uploaded)
        session = await self.sessions.get_session(session_id)
        file_ids = session.get("file_ids", [])
        if file_ids:
            await self._recheck_all_inputs(session_id, session)
            session = await self.sessions.get_session(session_id)
        
        # Try to extract text inputs from message
        current_wf = session.get("current_workflow", {})
        missing = self.collector.get_missing_inputs(current_wf.get("required_inputs", []))

        if missing:
            # Extract text inputs from message
            extracted = self.gemini.extract_inputs_from_message(message, missing)
            for field, value in extracted.items():
                if value is not None:
                    await self.sessions.mark_input_collected(session_id, field, value)

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
            # Enhance markdown
            prompt = self._enhance_response(prompt)
            
            await self.sessions.add_message(session_id, "assistant", prompt)
            await self.sessions.update_workflow_status(session_id, "collecting")
            
            # Get total session files count
            session_files_count = len(session.get("file_ids", []))
            
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
            return await self._execute_workflow_v2(session_id, session)
        else:
            # Dynamic confirmation prompt
            wf_name = matched.get("workflowName", "")
            
            # Check if workflow has zero inputs (API-only workflows)
            if not updated_inputs or len(updated_inputs) == 0:
                prompt = self.gemini.generate_contextual_response(
                    f"Workflow '{wf_name}' is ready to run. Ask user to confirm execution.",
                    {"workflow_name": wf_name, "has_inputs": False},
                )
            else:
                collected_data = {
                    inp.get("label", inp["field"]): self._format_value_for_display(
                        inp.get("value"), inp.get("type", "str")
                    )
                    for inp in updated_inputs if inp.get("collected")
                }
                prompt = self.gemini.generate_contextual_response(
                    f"All inputs for '{wf_name}' are collected. Show what was collected and ask user to confirm execution.",
                    {
                        "workflow_name": wf_name,
                        "collected_data": collected_data,
                        "has_inputs": True,
                    },
                )
            await self.sessions.add_message(session_id, "assistant", prompt)
            await self.sessions.update_workflow_status(session_id, "ready_to_execute")
            
            # Get total session files count
            session_files_count = len(session.get("file_ids", []))
            
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
            wf_name = workflow.get("workflowName", "the workflow")
            response_text = self.gemini.generate_contextual_response(
                f"User wants to delay or cancel '{wf_name}'. Acknowledge and offer to wait.",
                {"workflow_name": wf_name},
            )
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
            # All inputs collected - dynamic confirmation
            wf_name = workflow.get("workflowName", "")
            
            # Check if workflow has zero inputs
            if not required_inputs or len(required_inputs) == 0:
                prompt = self.gemini.generate_contextual_response(
                    f"Workflow '{wf_name}' is ready to run. Ask user to confirm execution.",
                    {"workflow_name": wf_name, "has_inputs": False},
                )
            else:
                collected_data = {
                    inp.get("label", inp["field"]): self._format_value_for_display(
                        inp.get("value"), inp.get("type", "str")
                    )
                    for inp in required_inputs if inp.get("collected")
                }
                prompt = self.gemini.generate_contextual_response(
                    f"All inputs for '{wf_name}' are collected. Show what was collected and ask user to confirm execution.",
                    {
                        "workflow_name": wf_name,
                        "collected_data": collected_data,
                        "has_inputs": True,
                    },
                )
            await self.sessions.add_message(session_id, "assistant", prompt)
            await self.sessions.update_workflow_status(session_id, "ready_to_execute")
            
            # Get total session files count
            session_files_count = len(session.get("file_ids", []))
            
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

        # Auto-collect file inputs using file_ids (if any files in session)
        session = await self.sessions.get_session(session_id)
        file_ids = session.get("file_ids", [])
        if file_ids:
            await self._recheck_all_inputs(session_id, session)

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
            # Enhance markdown
            prompt = self._enhance_response(prompt)
            
            await self.sessions.add_message(session_id, "assistant", prompt)
            
            # Get total session files count
            session_files_count = len(session.get("file_ids", []))
            
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
            # Dynamic confirmation prompt
            wf_name = workflow.get("workflowName", "")
            
            # Check if workflow has zero inputs
            if not updated_inputs or len(updated_inputs) == 0:
                prompt = self.gemini.generate_contextual_response(
                    f"Workflow '{wf_name}' is ready to run. Ask user to confirm execution.",
                    {"workflow_name": wf_name, "has_inputs": False},
                )
            else:
                collected_data = {
                    inp.get("label", inp["field"]): self._format_value_for_display(
                        inp.get("value"), inp.get("type", "str")
                    )
                    for inp in updated_inputs if inp.get("collected")
                }
                prompt = self.gemini.generate_contextual_response(
                    f"All inputs for '{wf_name}' are collected. Show what was collected and ask user to confirm execution.",
                    {
                        "workflow_name": wf_name,
                        "collected_data": collected_data,
                        "has_inputs": True,
                    },
                )
            await self.sessions.add_message(session_id, "assistant", prompt)
            await self.sessions.update_workflow_status(session_id, "ready_to_execute")
            
            # Get total session files count
            session_files_count = len(session.get("file_ids", []))
            
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
        # SAFETY NET: Ensure collected_inputs is synced with required_inputs
        await self._ensure_collected_inputs_synced(session_id)
        
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
        
        # Track file_ids being used for rerun capability
        # Extract file_ids from collected_inputs (more precise than session.file_ids)
        file_ids_in_use = []
        for inp in required_inputs:
            if inp.get("type") in ("file", "files") and inp.get("collected"):
                field_name = inp.get("field")
                file_ids_for_input = collected_inputs.get(field_name, [])
                if isinstance(file_ids_for_input, list):
                    file_ids_in_use.extend(file_ids_for_input)
        
        await self.sessions.update_workflow_status(
            session_id, 
            "executing",
            file_ids_in_use=file_ids_in_use
        )

        # Build request
        request_data = self.builder.build_workflow_request(
            workflow, collected_inputs, required_inputs, run_id, jwt_token, session_id,
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

                    # FILE ID CONVERSION: Convert collected file_ids to binary files
                    # Read file_ids from collected_inputs, not session (more precise)
                    files_list = []
                    
                    # Process each file-type input separately with correct field name
                    for inp in required_inputs:
                        if inp.get("type") in ("file", "files") and inp.get("collected"):
                            field_name = inp.get("field")
                            file_ids_for_input = collected_inputs.get(field_name, [])
                            
                            if file_ids_for_input and isinstance(file_ids_for_input, list):
                                # Get API parameter name (endpoint_field_name) for this input
                                api_field_name = inp.get("endpoint_field_name", field_name)
                                
                                logger.info(f"Converting {len(file_ids_for_input)} file_id(s) for '{field_name}' → '{api_field_name}'")
                                multipart_files = self.file_middleware.files_to_multipart(
                                    file_ids=file_ids_for_input,
                                    field_name=api_field_name  # Use API parameter name
                                )
                                files_list.extend(multipart_files)
                                logger.info(f"✅ Converted {len(multipart_files)} file(s) for '{api_field_name}'")

                    resp = await client.post(
                        url,
                        data=form_data,
                        files=files_list if files_list else None,
                        headers=request_data["headers"],  # Pass ALL headers including X-Session-Id
                    )

                    # Close file handles
                    for _, file_tuple in files_list:
                        if hasattr(file_tuple[1], 'close'):
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
            import traceback
            traceback.print_exc()
            
            await self.sessions.update_workflow_status(session_id, "failed")
            
            # Move to history and reset to idle (preserves session file_ids)
            await self.sessions.complete_workflow(session_id, {"error": str(e)}, status="failed")
            
            # Cleanup: Delete any active idle workflows for this session
            await self.idle_manager.delete_active_workflows(session_id)
            
            # Get session to include file count even on failure
            session = await self.sessions.get_session(session_id)
            session_files_count = len(session.get("file_ids", [])) if session else 0
            
            error_msg = "The workflow encountered an error during execution. Your files and session data are preserved."
            await self.sessions.add_message(session_id, "assistant", error_msg)
            
            return RouterResponse(
                session_id=session_id,
                status="idle",  # Back to idle for next workflow
                response=error_msg,
                error=ErrorDetail(code="EXECUTION_ERROR", message=str(e)),
                total_session_files=session_files_count if session_files_count > 0 else None,
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

            # Use intelligent result analyzer instead of basic formatting
            # Returns structured response for perfect markdown building
            structured_hitl = self.result_analyzer.analyze_workflow_result(
                result, 
                workflow_name, 
                result_type="hitl"
            )
            
            # Build perfect markdown from structured output
            hitl_prompt = self._enhance_response(structured_hitl)
            
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
        
        # Find file paths used by this workflow (blob paths valid even if not local)
        used_file_paths = []
        for field, value in collected_inputs.items():
            if isinstance(value, list):
                used_file_paths.extend(v for v in value if isinstance(v, str))
            elif isinstance(value, str) and "/" in value:
                used_file_paths.append(value)
        
        # Mark files as used
        if used_file_paths:
            session_files = session.get("uploaded_files", [])
            updated_files = self.file_intel.mark_files_used(
                session_files, used_file_paths, workflow_name
            )
            await self.sessions.update_files(session_id, updated_files)
        
        # Generate INTELLIGENT result summary using analyzer
        # This reads the ENTIRE response and curates insights
        # Returns structured response for perfect markdown building
        structured_summary = self.result_analyzer.analyze_workflow_result(
            result, 
            workflow_name, 
            result_type="final"
        )
        
        # Build perfect markdown from structured output
        summary = self._enhance_response(
            structured_summary,
            context={"workflow_type": workflow_name, "has_files": bool(result.get("file_outputs"))}
        )
        
        # Also generate factual summary for history/RAG
        result_summary = self.gemini.generate_result_summary(result, workflow_name)
        
        await self.sessions.complete_workflow(
            session_id, result, status="completed", result_summary=result_summary
        )
        
        # Cleanup: Delete any active idle workflows for this session
        await self.idle_manager.delete_active_workflows(session_id)
        logger.info(f"🧹 Cleaned up active workflows for session {session_id[:8]}...")
        
        await self.sessions.add_message(session_id, "assistant", summary)

        return RouterResponse(
            session_id=session_id,
            status="idle",  # Back to idle for next workflow!
            response=summary,
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

    async def _recheck_all_inputs(
        self, 
        session_id: str, 
        session: Dict[str, Any],
        new_file_ids: Optional[List[str]] = None
    ):
        """
        INTELLIGENT file input matching using FileIntelligence classification.
        Called after file upload to intelligently match files to workflow inputs.
        
        Smart Assignment Logic:
        1. Get file metadata with classification (document_type, keywords)
        2. Score each file against each uncollected input using FileIntelligence
        3. Assign best matches only if score exceeds threshold
        4. Prevents assigning same file to multiple inputs
        5. Allows reusing session files if they're good matches
        
        Args:
            session_id: Session ID
            session: Session data
            new_file_ids: Specific file_ids to prioritize (newly uploaded)
        """
        current_wf = session.get("current_workflow", {})
        if not current_wf.get("workflow_id"):
            return

        required_inputs = current_wf.get("required_inputs", [])
        workflow_name = current_wf.get("workflow_name", "")
        
        # Get uncollected file-type inputs
        uncollected_file_inputs = [
            inp for inp in required_inputs
            if inp.get("type") in ("file", "files") and not inp.get("collected")
        ]
        
        if not uncollected_file_inputs:
            return  # All file inputs already collected
        
        # Get all available file_ids from session (not just new ones)
        # This allows reusing files across workflows if they match
        all_file_ids = session.get("file_ids", [])
        if not all_file_ids:
            return
        
        # Fetch file metadata from file middleware for classification
        # Filter to only files that exist and are accessible
        file_metadata_list = []
        for fid in all_file_ids:
            file_meta = self.file_middleware.files_collection.find_one({"_id": fid})
            if file_meta:
                # Classify file if not already classified
                if "classification" not in file_meta:
                    classification = self.file_intel.classify_file({
                        "original_name": file_meta.get("original_name", ""),
                        "mime_type": file_meta.get("mime_type", ""),
                        "size_bytes": file_meta.get("size_bytes", 0),
                    })
                    # Store classification in MongoDB for future use
                    self.file_middleware.files_collection.update_one(
                        {"_id": fid},
                        {"$set": {"classification": classification}}
                    )
                    file_meta["classification"] = classification
                
                file_metadata_list.append(file_meta)
        
        if not file_metadata_list:
            return  # No file metadata available
        
        # Track which file_ids are already assigned to THIS workflow's inputs
        already_assigned = set()
        for inp in required_inputs:
            if inp.get("collected"):
                value = inp.get("value")
                if isinstance(value, list):
                    already_assigned.update(value)
                elif value:
                    already_assigned.add(value)
        
        matched_any = False
        
        # Match each uncollected input to best available file
        for inp in uncollected_file_inputs:
            input_type = inp.get("type")
            
            if input_type == "file":
                # Single file input: Find best match
                best_file = None
                best_score = 0.0
                
                for file_meta in file_metadata_list:
                    fid = file_meta["file_id"]
                    if fid in already_assigned:
                        continue  # Skip already assigned files
                    
                    score = self.file_intel.match_file_to_input(
                        file_meta,
                        inp,
                        workflow_name
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_file = file_meta
                
                # Only auto-assign if score is above threshold (3.0 = confident match)
                threshold = 3.0
                if best_file and best_score >= threshold:
                    file_value = [best_file["file_id"]]
                    await self.sessions.mark_input_collected(
                        session_id, inp["field"], file_value
                    )
                    already_assigned.add(best_file["file_id"])
                    matched_any = True
                    logger.info(
                        f"✅ Auto-collected '{inp.get('label', inp['field'])}' with file '{best_file['original_name']}' "
                        f"(score: {best_score:.1f}, type: {best_file.get('classification', {}).get('document_type', 'unknown')})"
                    )
                else:
                    logger.info(
                        f"⏸️ No confident match for '{inp.get('label', inp['field'])}' "
                        f"(best score: {best_score:.1f} < threshold: {threshold})"
                    )
            
            elif input_type == "files":
                # Multi-file input: Collect all unassigned files with positive scores
                matched_files = []
                for file_meta in file_metadata_list:
                    fid = file_meta["file_id"]
                    if fid in already_assigned:
                        continue
                    
                    score = self.file_intel.match_file_to_input(
                        file_meta,
                        inp,
                        workflow_name
                    )
                    
                    if score >= 2.0:  # Lower threshold for multi-file inputs
                        matched_files.append(fid)
                        already_assigned.add(fid)
                
                if matched_files:
                    await self.sessions.mark_input_collected(
                        session_id, inp["field"], matched_files
                    )
                    matched_any = True
                    logger.info(
                        f"✅ Auto-collected '{inp.get('label', inp['field'])}' with {len(matched_files)} file(s)"
                    )

        # Auto-transition to ready_to_execute if all inputs now collected
        if matched_any:
            session = await self.sessions.get_session(session_id)
            current_wf = session.get("current_workflow", {})
            updated_inputs = current_wf.get("required_inputs", [])
            still_missing = self.collector.get_missing_inputs(updated_inputs)
            
            if not still_missing and current_wf.get("status") == "collecting":
                await self.sessions.update_workflow_status(session_id, "ready_to_execute")
                logger.info(f"🚀 All inputs collected → status changed to 'ready_to_execute'")
    
    def _get_pinecone_context(
        self,
        query: str,
        session_id: str,
        org_id: str,
        run_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context from Pinecone using semantic search.
        
        Args:
            query: Search query (current message or workflow description)
            session_id: Current session ID
            org_id: Organization ID
            run_id: Optional run ID filter
            top_k: Number of results to retrieve
        
        Returns:
            Dict with relevant_messages and relevant_files
        """
        context = {
            "relevant_messages": [],
            "relevant_files": [],
        }
        
        # Search for relevant messages
        message_matches = pinecone_client.search_context(
            query=query,
            org_id=org_id,
            session_id=session_id,
            run_id=run_id,
            top_k=top_k,
            filter_type="message",
        )
        
        for match in message_matches:
            metadata = match.get("metadata", {})
            context["relevant_messages"].append({
                "content": metadata.get("content", ""),
                "role": metadata.get("role", ""),
                "timestamp": metadata.get("timestamp", ""),
                "score": match.get("score", 0),
                "workflow_name": metadata.get("workflow_name"),
            })
        
        # Search for relevant files
        file_matches = pinecone_client.search_context(
            query=query,
            org_id=org_id,
            session_id=session_id,
            run_id=run_id,
            top_k=top_k,
            filter_type="file",
        )
        
        for match in file_matches:
            metadata = match.get("metadata", {})
            context["relevant_files"].append({
                "filename": metadata.get("filename", ""),
                "file_id": metadata.get("file_id", ""),
                "content_preview": metadata.get("content_preview", ""),
                "score": match.get("score", 0),
                "classification": metadata.get("classification"),
            })
        
        logger.debug(
            f"🔍 Pinecone context: {len(context['relevant_messages'])} messages, "
            f"{len(context['relevant_files'])} files"
        )
        
        return context
    
    async def _handle_file_classification(
        self,
        session_id: str,
        new_file_ids: List[str],
        idle_workflows_summary: List[Dict[str, Any]],
        conversation_context: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Classify uploaded files against idle workflows.
        
        Returns:
            {
                "matched_count": int,
                "unmatched_count": int,
                "response": RouterResponse
            }
        """
        # Get full idle workflow details from IdleWorkflowManager
        idle_workflows_full = []
        for summary in idle_workflows_summary:
            instance_id = summary.get("instance_id")
            if instance_id:
                full_wf = await self.idle_manager.get_idle_workflow(instance_id)
                if full_wf:
                    idle_workflows_full.append(full_wf)
        
        if not idle_workflows_full:
            return None
        
        # Extract recent messages for context
        recent_messages = []
        for entry in conversation_context[-5:]:
            if isinstance(entry, dict):
                recent_messages.append(f"{entry.get('role', 'user')}: {entry.get('content', '')}")
        
        # Classify each file
        matched_files = []
        unmatched_files = []
        
        for file_id in new_file_ids:
            classification = await self.file_classifier.classify_file(
                file_id=file_id,
                idle_workflows=idle_workflows_full,
                conversation_context=recent_messages
            )
            
            if classification["matched"] and classification["confidence"] >= 0.80:
                # High confidence match - auto-assign
                await self.idle_manager.add_file_to_idle_workflow(
                    instance_id=classification["workflow_instance_id"],
                    input_field=classification["input_field"],
                    file_id=file_id
                )
                
                matched_files.append({
                    "file_id": file_id,
                    "workflow_instance_id": classification["workflow_instance_id"],
                    "workflow_name": next(
                        (wf["workflow_name"] for wf in idle_workflows_full 
                         if wf["instance_id"] == classification["workflow_instance_id"]),
                        "Unknown"
                    ),
                    "input_field": classification["input_field"],
                    "confidence": classification["confidence"]
                })
                
                # Check if workflow is now complete
                completeness = await self.idle_manager.check_if_complete(
                    classification["workflow_instance_id"]
                )
                
                if completeness["is_complete"]:
                    # Workflow ready to execute
                    await self.idle_manager.update_status(
                        classification["workflow_instance_id"],
                        "ready_to_execute"
                    )
                    logger.info(f"✅ Idle workflow now complete: {classification['workflow_instance_id'][:8]}...")
            
            elif classification["confidence"] >= 0.60:
                # Medium confidence - add to matched but note ambiguity
                matched_files.append({
                    "file_id": file_id,
                    "workflow_instance_id": classification.get("workflow_instance_id"),
                    "confidence": classification["confidence"],
                    "needs_confirmation": True,
                    "candidates": classification.get("candidates", [])
                })
            else:
                # Low confidence - unmatched
                unmatched_files.append({
                    "file_id": file_id,
                    "confidence": classification["confidence"],
                    "reason": classification.get("reasoning", "Could not match to any workflow")
                })
                
                # Add to unassigned files in session
                file_meta = self.file_middleware.get_files_metadata([file_id])
                if file_meta:
                    await self.sessions.add_unassigned_file(session_id, {
                        "file_id": file_id,
                        "filename": file_meta[0].get("original_name", "unknown"),
                        "uploaded_at": file_meta[0].get("uploaded_at", ""),
                        "classification_attempted": True,
                        "confidence": classification["confidence"]
                    })
        
        # Build response
        if matched_files or unmatched_files:
            response_parts = []
            
            if matched_files:
                auto_matched = [f for f in matched_files if not f.get("needs_confirmation")]
                if auto_matched:
                    response_parts.append(f"✅ Matched {len(auto_matched)} file(s) to idle workflows:")
                    for match in auto_matched:
                        response_parts.append(f"  - {match['workflow_name']} (confidence: {match['confidence']:.0%})")
                
                needs_confirm = [f for f in matched_files if f.get("needs_confirmation")]
                if needs_confirm:
                    response_parts.append(f"\n❓ {len(needs_confirm)} file(s) have multiple possible matches. Please clarify:")
                    # Add disambiguation logic here if needed
            
            if unmatched_files:
                response_parts.append(f"\n❌ Could not match {len(unmatched_files)} file(s). Please specify which workflow they belong to.")
            
            response_text = "\n".join(response_parts)
            await self.sessions.add_message(session_id, "assistant", response_text)
            
            return {
                "matched_count": len([f for f in matched_files if not f.get("needs_confirmation")]),
                "unmatched_count": len(unmatched_files),
                "response": RouterResponse(
                    session_id=session_id,
                    status="files_classified",
                    response=response_text
                )
            }
        
        return None
    
    async def _save_current_workflow_to_idle(
        self,
        session_id: str,
        current_wf: Dict[str, Any],
        user_id: str,
        org_id: str
    ) -> Optional[str]:
        """
        Save current workflow to idle state when user switches to a different workflow.
        Preserves complete workflow state including HITL context, schema, and file_ids.
        Allows user to resume this workflow later from exact same state.
        
        Returns:
            instance_id of the created idle workflow, or None if failed
        """
        try:
            workflow_id = current_wf.get("workflow_id")
            workflow_name = current_wf.get("workflow_name")
            workflow_endpoint = current_wf.get("workflow_endpoint")
            wf_status = current_wf.get("status", "collecting")
            
            if not workflow_id or not workflow_name:
                logger.warning("Cannot save workflow to idle - missing workflow info")
                return None
            
            # Separate collected and missing inputs
            required_inputs = current_wf.get("required_inputs", [])
            collected_inputs = current_wf.get("collected_inputs", {})
            
            missing_inputs = []
            for inp in required_inputs:
                if not inp.get("collected"):
                    # Build file_signature if this is a file input
                    file_signature = None
                    if inp.get("type") in ("file", "files"):
                        # Try to extract file_signature from workflow schema
                        workflow_schema = current_wf.get("workflow_schema", {})
                        body_data = workflow_schema.get("call0", {}).get("body_data", {})
                        field_schema = body_data.get(inp.get("field"), {})
                        file_signature = field_schema.get("file_signature")
                    
                    missing_input = {
                        "field": inp.get("field"),
                        "inputLabel": inp.get("label", inp.get("field")),
                        "inputType": inp.get("type", "str"),
                        "inputSize": inp.get("size", 1),
                        "already_collected": 0,
                        "priority": inp.get("priority", "required")
                    }
                    
                    if file_signature:
                        missing_input["file_signature"] = file_signature
                    
                    missing_inputs.append(missing_input)
            
            # Build execution context to preserve complete state
            execution_context = {
                "original_status": wf_status,
                "workflow_schema": current_wf.get("workflow_schema"),
                "workflow_api_calls": current_wf.get("workflow_api_calls"),
                "file_ids_in_use": current_wf.get("file_ids_in_use", []),
                "required_inputs_full": required_inputs,  # Full input specs with types
            }
            
            # If in HITL state, preserve HITL context
            if wf_status == "awaiting_confirmation":
                execution_context["confirmation_data"] = current_wf.get("confirmation_data")
                execution_context["run_id"] = current_wf.get("run_id")
                execution_context["hitl_request"] = current_wf.get("hitl_request")
                logger.info(f"💤 Preserving HITL state for {workflow_name} (run_id: {execution_context.get('run_id', 'N/A')[:8]}...)")
            elif wf_status == "ready_to_execute":
                # Preserve that it was ready to execute
                execution_context["was_ready"] = True
            
            # Determine idle status based on current status
            idle_status = "idle_awaiting_confirmation" if wf_status == "awaiting_confirmation" else "idle_awaiting_files"
            
            # Create idle workflow entry
            instance_id = await self.idle_manager.create_idle_workflow(
                workflow_id=workflow_id,
                workflow_name=workflow_name,
                workflow_endpoint=workflow_endpoint,
                user_id=user_id,
                org_id=org_id,
                session_id=session_id,
                collected_inputs=collected_inputs,
                missing_inputs=missing_inputs,
                idle_source="workflow_switching",
                execution_context=execution_context
            )
            
            # Update the status to reflect HITL state if applicable
            if idle_status != "idle_awaiting_files":
                await self.idle_manager.update_status(instance_id, idle_status)
            
            # Add to session's idle_workflows summary
            idle_summary = {
                "instance_id": instance_id,
                "workflow_name": workflow_name,
                "status": idle_status,
                "missing_count": len(missing_inputs),
                "original_status": wf_status,
                "last_active": datetime.utcnow().isoformat()
            }
            
            await self.sessions.add_idle_workflow(session_id, idle_summary)
            
            logger.info(f"💤 Workflow saved to idle state: {workflow_name} (status: {wf_status} → {idle_status}, instance: {instance_id[:8]}...)")
            return instance_id
            
        except Exception as e:
            logger.error(f"Failed to save workflow to idle state: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _detect_resume_intent(
        self,
        message: str,
        idle_workflows: List[Dict[str, Any]]
    ) -> Optional[Dict[str, str]]:
        """
        Detect if user wants to resume an idle workflow.
        Checks for keywords like "continue", "resume", "back to" + workflow name.
        
        Returns:
            {instance_id, workflow_name} if match found, None otherwise
        """
        message_lower = message.lower()
        
        # Keywords indicating resume intent
        resume_keywords = ["continue", "resume", "back to", "return to", "go back to", "switch back"]
        
        has_resume_keyword = any(keyword in message_lower for keyword in resume_keywords)
        
        if not has_resume_keyword:
            return None
        
        # Try to match workflow name in message
        for workflow in idle_workflows:
            workflow_name = workflow.get("workflow_name", "").lower()
            if workflow_name and workflow_name in message_lower:
                return {
                    "instance_id": workflow["instance_id"],
                    "workflow_name": workflow["workflow_name"]
                }
        
        # If only one idle workflow, assume user wants that one
        if len(idle_workflows) == 1 and has_resume_keyword:
            return {
                "instance_id": idle_workflows[0]["instance_id"],
                "workflow_name": idle_workflows[0]["workflow_name"]
            }
        
        return None
    
    async def _resume_idle_workflow(
        self,
        session_id: str,
        instance_id: str,
        workflow_name: str,
        user_id: str,
        org_id: str,
        jwt_token: str,
        files_uploaded: List
    ) -> RouterResponse:
        """
        Resume an idle workflow by loading its complete state and making it the active workflow.
        Handles resumption from any state: collecting, ready_to_execute, or awaiting_confirmation (HITL).
        """
        try:
            # Get full idle workflow details
            idle_workflow = await self.idle_manager.get_idle_workflow(instance_id)
            
            if not idle_workflow:
                response = f"Could not find idle workflow: {workflow_name}"
                await self.sessions.add_message(session_id, "assistant", response)
                return RouterResponse(
                    session_id=session_id,
                    status="idle",
                    response=response
                )
            
            # Get execution context to restore complete state
            execution_context = idle_workflow.get("execution_context", {})
            original_status = execution_context.get("original_status", "collecting")
            
            logger.info(f"🔄 Resuming {workflow_name} from status: {original_status}")
            
            # Reconstruct workflow object with full schema
            workflow = {
                "workflowId": idle_workflow["workflow_id"],
                "workflowName": idle_workflow["workflow_name"],
                "workflowEndpoint": idle_workflow["workflow_endpoint"],
                "workflowSchema": execution_context.get("workflow_schema"),
                "workflowApiCalls": execution_context.get("workflow_api_calls"),
            }
            
            # Use full required_inputs from execution_context if available (preserves types, labels, etc.)
            required_inputs = execution_context.get("required_inputs_full", [])
            
            # If not available, reconstruct from collected and missing
            if not required_inputs:
                collected_inputs = idle_workflow.get("collected_inputs", {})
                
                # Add collected inputs (marked as collected)
                for field, value in collected_inputs.items():
                    required_inputs.append({
                        "field": field,
                        "value": value,
                        "collected": True,
                        "type": "file" if isinstance(value, list) else "str",
                        "label": field
                    })
                
                # Add missing inputs (not collected)
                for missing in idle_workflow.get("missing_inputs", []):
                    required_inputs.append({
                        "field": missing["field"],
                        "label": missing.get("inputLabel", missing["field"]),
                        "type": missing.get("inputType", "str"),
                        "size": missing.get("inputSize", 1),
                        "collected": False,
                        "priority": missing.get("priority", "required")
                    })
            
            # Auto-fill user/org inputs
            required_inputs = self.collector.auto_fill_inputs(
                required_inputs,
                {"user_id": user_id, "org_id": org_id}
            )
            
            # Get collected_inputs and file_ids_in_use from idle workflow
            collected_inputs_from_idle = idle_workflow.get("collected_inputs", {})
            file_ids_in_use = execution_context.get("file_ids_in_use", [])
            
            logger.info(f"🔄 Restoring workflow state: {len(collected_inputs_from_idle)} inputs, {len(file_ids_in_use)} files")
            
            # Set as current workflow WITH preserved collected_inputs
            await self.sessions.set_workflow_context(
                session_id, 
                workflow, 
                required_inputs,
                collected_inputs=collected_inputs_from_idle,
                file_ids_in_use=file_ids_in_use
            )
            
            # Check if this was in HITL state - restore HITL context
            if original_status == "awaiting_confirmation":
                confirmation_data = execution_context.get("confirmation_data")
                run_id = execution_context.get("run_id")
                hitl_request = execution_context.get("hitl_request")
                
                logger.info(f"✅ Resuming HITL workflow (run_id: {run_id[:8] if run_id else 'N/A'}...)")
                
                # Update session with HITL context
                await self.sessions.update_session(session_id, {
                    "current_workflow.status": "awaiting_confirmation",
                    "current_workflow.confirmation_data": confirmation_data,
                    "current_workflow.run_id": run_id,
                    "current_workflow.hitl_request": hitl_request,
                })
                
                # Mark workflow as active (persistent state pattern)
                await self.idle_manager.activate_workflow(instance_id)
                await self.sessions.remove_idle_workflow_from_session(session_id, instance_id)
                
                # Generate HITL resume prompt
                prompt = self.gemini.format_hitl_prompt(confirmation_data, workflow_name) if confirmation_data else "Please review the workflow data and confirm to proceed."
                response = f"✅ Resumed **{workflow_name}** (awaiting confirmation).\n\n{prompt}"
                await self.sessions.add_message(session_id, "assistant", response)
                
                return RouterResponse(
                    session_id=session_id,
                    status="awaiting_confirmation",
                    response=response,
                    requires_confirmation=True,
                    workflow_identified=WorkflowIdentified(
                        id=workflow["workflowId"],
                        name=workflow_name,
                        endpoint=workflow["workflowEndpoint"],
                    ),
                    confirmation_data=ConfirmationData(
                        runId=run_id,
                        workflowId=workflow["workflowId"],
                        step_number=confirmation_data.get("stepNumber", 0) if confirmation_data else 0,
                        data_to_review=confirmation_data,
                    ) if confirmation_data and run_id else None,
                    files_uploaded=files_uploaded if files_uploaded else None,
                )
            
            # Not in HITL - mark workflow as active (persistent state pattern)
            await self.idle_manager.activate_workflow(instance_id)
            await self.sessions.remove_idle_workflow_from_session(session_id, instance_id)
            
            # Check what inputs are still needed
            session = await self.sessions.get_session(session_id)
            current_wf = session.get("current_workflow", {})
            updated_inputs = current_wf.get("required_inputs", [])
            still_missing = self.collector.get_missing_inputs(updated_inputs)
            
            if still_missing:
                prompt = self.gemini.generate_input_prompt(
                    workflow_name,
                    still_missing[0],
                    updated_inputs,
                )
                prompt = self._enhance_response(prompt)
                response = f"✅ Resumed **{workflow_name}**.\n\n{prompt}"
                await self.sessions.add_message(session_id, "assistant", response)
                await self.sessions.update_workflow_status(session_id, "collecting")
                
                return RouterResponse(
                    session_id=session_id,
                    status="collecting",
                    response=response,
                    workflow_identified=WorkflowIdentified(
                        id=workflow["workflowId"],
                        name=workflow_name,
                        endpoint=workflow["workflowEndpoint"],
                    ),
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
                )
            else:
                # All inputs collected - ready to execute
                await self.sessions.update_workflow_status(session_id, "ready_to_execute")
                prompt = self.gemini.generate_contextual_response(
                    f"Resumed workflow '{workflow_name}' is ready. Ask user to confirm execution.",
                    {"workflow_name": workflow_name},
                )
                response = f"✅ Resumed **{workflow_name}**. All inputs collected!\n\n{prompt}"
                await self.sessions.add_message(session_id, "assistant", response)
                
                return RouterResponse(
                    session_id=session_id,
                    status="ready_to_execute",
                    response=response,
                    workflow_identified=WorkflowIdentified(
                        id=workflow["workflowId"],
                        name=workflow_name,
                        endpoint=workflow["workflowEndpoint"],
                    ),
                    files_uploaded=files_uploaded if files_uploaded else None,
                )
                
        except Exception as e:
            logger.error(f"Failed to resume idle workflow: {e}")
            response = f"Failed to resume {workflow_name}. Please try starting it fresh."
            await self.sessions.add_message(session_id, "assistant", response)
            return RouterResponse(
                session_id=session_id,
                status="idle",
                response=response
            )
    
    async def _switch_to_workflow(
        self,
        session_id: str,
        target_workflow: Dict[str, Any],
        user_id: str,
        org_id: str,
        jwt_token: str,
        files_uploaded: List,
        save_current: bool = True,
        current_wf: Optional[Dict[str, Any]] = None,
        triggering_message: Optional[str] = None  # ✅ NEW: Message that triggered the switch
    ) -> RouterResponse:
        """
        Universal workflow switching handler - resumes from idle if exists, starts fresh if not.
        
        This is the SINGLE ENTRY POINT for all workflow switches, ensuring consistent behavior:
        - Checks if target workflow exists in idle_workflows database
        - If exists: loads complete state (collected_inputs, file_ids, HITL context)
        - If not: starts fresh, optionally extracting inputs from triggering message
        - Generalizable across all workflow types and states
        
        Args:
            session_id: Session identifier
            target_workflow: Target workflow metadata from registry
            user_id, org_id, jwt_token: Authentication context
            files_uploaded: Files uploaded in this request
            save_current: Whether to save current workflow to idle before switching
            current_wf: Current workflow context (for saving)
            triggering_message: Optional message that triggered the switch (for input extraction)
        
        Returns:
            RouterResponse with appropriate status and prompts
        """
        target_workflow_id = target_workflow.get("workflowId", "")
        target_workflow_name = target_workflow.get("workflowName", "")
        
        # Save current workflow to idle if requested
        if save_current and current_wf and current_wf.get("workflow_id"):
            await self._save_current_workflow_to_idle(
                session_id=session_id,
                current_wf=current_wf,
                user_id=user_id,
                org_id=org_id
            )
        
        # Check if target workflow exists in idle state
        existing_idle = await self.idle_manager.find_idle_workflow_by_workflow_id(
            session_id, 
            target_workflow_id
        )
        
        if existing_idle:
            # RESUME from idle - load complete state
            logger.info(f"🔄 Switching to EXISTING idle workflow: {target_workflow_name}")
            return await self._resume_idle_workflow(
                session_id=session_id,
                instance_id=existing_idle["instance_id"],
                workflow_name=target_workflow_name,
                user_id=user_id,
                org_id=org_id,
                jwt_token=jwt_token,
                files_uploaded=files_uploaded
            )
        else:
            # START FRESH - no previous state
            logger.info(f"🆕 Switching to NEW workflow: {target_workflow_name}")
            
            # Parse and auto-fill inputs
            required_inputs = self.collector.parse_workflow_inputs(target_workflow)
            required_inputs = self.collector.auto_fill_inputs(
                required_inputs,
                {"user_id": user_id, "org_id": org_id}
            )
            
            # ✅ OPTION 3: Extract inputs from triggering message if provided
            collected_inputs = {}
            if triggering_message:
                logger.info(f"🔍 Extracting inputs from triggering message: '{triggering_message[:80]}...'")
                missing_inputs = [inp for inp in required_inputs if not inp.get("collected")]
                if missing_inputs:
                    logger.info(f"📝 Missing inputs for extraction: {[inp['field'] for inp in missing_inputs]}")
                    extracted = self.gemini.extract_inputs_from_message(
                        triggering_message,
                        missing_inputs
                    )
                    if extracted:
                        logger.info(f"✅ Pre-extracted inputs from switch message: {extracted}")
                        # Map extracted values to collected_inputs
                        for inp in required_inputs:
                            field = inp.get("field")
                            if field in extracted and extracted[field]:
                                inp["value"] = extracted[field]
                                inp["collected"] = True
                                collected_inputs[field] = extracted[field]
                                logger.info(f"  ✓ {field} = {extracted[field]}")
                    else:
                        logger.info(f"⚠️ No inputs extracted from triggering message")
                else:
                    logger.info(f"✅ All inputs already collected, no extraction needed")
            else:
                logger.info(f"ℹ️ No triggering message provided, starting with empty inputs")
            
            # Set workflow context with pre-extracted inputs
            await self.sessions.set_workflow_context(
                session_id, 
                target_workflow, 
                required_inputs,
                collected_inputs=collected_inputs if collected_inputs else None
            )
            
            # Auto-collect file inputs if any files in session
            session = await self.sessions.get_session(session_id)
            file_ids = session.get("file_ids", [])
            if file_ids:
                await self._recheck_all_inputs(session_id, session)
                session = await self.sessions.get_session(session_id)
            
            # Check what inputs are still needed
            current_wf = session.get("current_workflow", {})
            updated_inputs = current_wf.get("required_inputs", [])
            still_missing = self.collector.get_missing_inputs(updated_inputs)
            
            if still_missing:
                # Collecting state
                prompt = self.gemini.generate_input_prompt(
                    target_workflow_name,
                    still_missing[0],
                    updated_inputs,
                )
                prompt = self._enhance_response(prompt)
                await self.sessions.add_message(session_id, "assistant", prompt)
                await self.sessions.update_workflow_status(session_id, "collecting")
                
                return RouterResponse(
                    session_id=session_id,
                    status="collecting",
                    response=prompt,
                    workflow_identified=WorkflowIdentified(
                        id=target_workflow_id,
                        name=target_workflow_name,
                        endpoint=target_workflow.get("workflowEndpoint", ""),
                    ),
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
                )
            else:
                # Ready to execute
                await self.sessions.update_workflow_status(session_id, "ready_to_execute")
                prompt = self.gemini.generate_contextual_response(
                    f"Workflow '{target_workflow_name}' is ready. Ask user to confirm execution.",
                    {"workflow_name": target_workflow_name},
                )
                await self.sessions.add_message(session_id, "assistant", prompt)
                
                return RouterResponse(
                    session_id=session_id,
                    status="ready_to_execute",
                    response=prompt,
                    workflow_identified=WorkflowIdentified(
                        id=target_workflow_id,
                        name=target_workflow_name,
                        endpoint=target_workflow.get("workflowEndpoint", ""),
                    ),
                    files_uploaded=files_uploaded if files_uploaded else None,
                )

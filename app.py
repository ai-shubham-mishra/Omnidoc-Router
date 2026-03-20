from fastapi import FastAPI, Request, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import logging

from auth.middleware import JWTAuthMiddleware
from auth.models import UserContext
from auth.dependencies import get_current_user

from core.orchestrator import RouterOrchestrator
from models.api_contracts import (
    ChatRequest,
    MessageRequest,
    ConfirmationRequest,
    RouterResponse,
    ErrorDetail,
    FileUploaded,
)
from utils.config import API_VERSION, LAST_UPDATED
from utils.redis_client import redis_client

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Omnidoc Router",
    description="LLM-powered conversational workflow orchestrator with Redis caching",
    version=API_VERSION,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT Auth Middleware
app.add_middleware(
    JWTAuthMiddleware,
    excluded_paths=["/docs", "/redoc", "/openapi.json", "/health", "/"],
)

# Orchestrator singleton
router_orchestrator = RouterOrchestrator()


# ============== Lifecycle Events ==============

@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection on startup."""
    await redis_client.connect()
    logging.info(f"🚀 Omnidoc Router v{API_VERSION} started")


@app.on_event("shutdown")
async def shutdown_event():
    """Close Redis connection on shutdown."""
    await redis_client.close()
    logging.info("👋 Omnidoc Router shutdown")


# ============== Router Endpoints ==============

@app.post("/api/router/message", response_model=RouterResponse)
async def router_message(
    request: Request,
    session_id: Optional[str] = Form(None),
    message: str = Form(""),
    files: List[UploadFile] = File([]),
    user_context: UserContext = Depends(get_current_user),
):
    """
    Unified endpoint for conversational workflow orchestration.
    Accepts multipart/form-data with message + files in single request.
    
    - Send message to chat
    - Upload files to session context
    - Or both simultaneously
    
    This is the recommended endpoint (replaces /chat and /upload).
    """
    user_id = user_context.user_id
    org_id = user_context.raw_payload.get("organizationId", "")
    jwt_token = request.headers.get("Authorization", "")

    result = await router_orchestrator.handle_message(
        message=message,
        session_id=session_id,
        files=files if files else [],
        user_id=user_id,
        org_id=org_id,
        jwt_token=jwt_token,
    )
    return result


@app.post("/api/router/chat", response_model=RouterResponse)
async def router_chat(
    body: ChatRequest,
    request: Request,
    user_context: UserContext = Depends(get_current_user),
):
    """Chat with the LLM Router. Handles intent detection, input collection, and workflow execution."""
    user_id = user_context.user_id
    org_id = user_context.raw_payload.get("organizationId", "")
    jwt_token = request.headers.get("Authorization", "")

    result = await router_orchestrator.handle_chat(
        message=body.message,
        session_id=body.session_id,
        user_id=user_id,
        org_id=org_id,
        jwt_token=jwt_token,
    )
    return result


@app.post("/api/router/upload", response_model=RouterResponse)
async def router_upload(
    request: Request,
    user_context: UserContext = Depends(get_current_user),
):
    """Upload files for a router session. Expects multipart/form-data with session_id and files."""
    user_id = user_context.user_id
    org_id = user_context.raw_payload.get("organizationId", "")
    jwt_token = request.headers.get("Authorization", "")

    form = await request.form()
    session_id = form.get("session_id")
    field_name = form.get("field_name", "files")
    files = form.getlist("files")

    if not session_id:
        return RouterResponse(
            status="failed",
            response="session_id is required for file uploads.",
            error=ErrorDetail(code="MISSING_SESSION", message="Provide session_id in form data"),
        )

    if not files:
        return RouterResponse(
            session_id=session_id,
            status="failed",
            response="No files provided.",
            error=ErrorDetail(code="NO_FILES", message="Upload at least one file"),
        )

    result = await router_orchestrator.handle_file_upload(
        session_id=session_id,
        files=files,
        field_name=field_name,
        user_id=user_id,
        org_id=org_id,
        jwt_token=jwt_token,
    )
    return result


@app.post("/api/router/confirm", response_model=RouterResponse)
async def router_confirm(
    body: ConfirmationRequest,
    request: Request,
    user_context: UserContext = Depends(get_current_user),
):
    """Confirm or cancel a HITL step."""
    user_id = user_context.user_id
    org_id = user_context.raw_payload.get("organizationId", "")
    jwt_token = request.headers.get("Authorization", "")

    result = await router_orchestrator.handle_confirmation(
        session_id=body.session_id,
        action=body.action,
        user_id=user_id,
        org_id=org_id,
        jwt_token=jwt_token,
        message=body.message,
    )
    return result


@app.get("/api/router/session/{session_id}")
async def router_get_session(
    session_id: str,
    user_context: UserContext = Depends(get_current_user),
):
    """Get session details including conversation history and state."""
    session = router_orchestrator.get_session_info(session_id)
    if not session:
        return JSONResponse(
            content={"status": "error", "message": "Session not found"},
            status_code=404,
        )
    if session.get("user_id") != user_context.user_id:
        return JSONResponse(
            content={"status": "error", "message": "Access denied"},
            status_code=403,
        )
    return JSONResponse(content=session, status_code=200)


@app.delete("/api/router/session/{session_id}")
async def router_delete_session(
    session_id: str,
    user_context: UserContext = Depends(get_current_user),
):
    """Delete a chat session."""
    deleted = await router_orchestrator.delete_session(session_id, user_context.user_id)
    if not deleted:
        return JSONResponse(
            content={"status": "error", "message": "Session not found or access denied"},
            status_code=404,
        )
    return JSONResponse(
        content={"status": "ok", "message": "Session deleted", "session_id": session_id},
        status_code=200,
    )


# ============== Health & Info ==============

@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"}, status_code=200)


@app.get("/")
async def home():
    return JSONResponse(content={
        "message": "Welcome to the Omnidoc Router",
        "version": API_VERSION,
        "last_updated": LAST_UPDATED,
    }, status_code=200)

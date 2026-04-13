from fastapi import FastAPI, Request, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import logging
import os

from dotenv import load_dotenv

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
from components.FileMiddleware import file_middleware

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Suppress verbose Azure SDK HTTP logs
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

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
    file_ids: Optional[str] = Form(None),
    user_context: UserContext = Depends(get_current_user),
):
    """
    Unified endpoint for conversational workflow orchestration.
    Accepts multipart/form-data with message + files + file_ids in single request.
    
    - Send message to chat
    - Upload files to session context (files → middleware → file_ids)
    - Reference existing files via file_ids (for reruns, cross-workflow)
    - Or all simultaneously
    
    Session ID Priority:
    1. X-Session-Id header (required - frontend always provides)
    2. Form data session_id (fallback)
    
    Returns error if no session_id provided.
    """
    user_id = user_context.user_id
    org_id = user_context.raw_payload.get("organizationId", "")
    jwt_token = request.headers.get("Authorization", "")
    
    # Get session ID from header (primary) or form data (fallback)
    final_session_id = request.headers.get("X-Session-Id") or session_id
    
    # Session ID is required - frontend must provide it
    if not final_session_id:
        return RouterResponse(
            status="failed",
            response="Session ID is required. Provide X-Session-Id header.",
            error=ErrorDetail(code="MISSING_SESSION_ID", message="X-Session-Id header is required"),
        )
    
    # Parse file_ids (comes as comma-separated string or JSON array)
    parsed_file_ids = []
    if file_ids:
        import json
        try:
            parsed_file_ids = json.loads(file_ids) if file_ids.startswith('[') else file_ids.split(',')
            parsed_file_ids = [fid.strip() for fid in parsed_file_ids if fid.strip()]
        except:
            parsed_file_ids = []

    result = await router_orchestrator.handle_message(
        message=message,
        session_id=final_session_id,
        files=files if files else [],
        file_ids=parsed_file_ids,
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
    """Chat with the LLM Router. Session ID required via X-Session-Id header or body."""
    user_id = user_context.user_id
    org_id = user_context.raw_payload.get("organizationId", "")
    jwt_token = request.headers.get("Authorization", "")
    
    # Get session ID from header (primary) or body (fallback)
    final_session_id = request.headers.get("X-Session-Id") or body.session_id
    
    # Session ID is required
    if not final_session_id:
        return RouterResponse(
            status="failed",
            response="Session ID is required. Provide X-Session-Id header.",
            error=ErrorDetail(code="MISSING_SESSION_ID", message="X-Session-Id header or body.session_id is required"),
        )

    result = await router_orchestrator.handle_chat(
        message=body.message,
        session_id=final_session_id,
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
    
    # Prioritize X-Session-Id header over form data
    final_session_id = request.headers.get("X-Session-Id") or session_id

    if not final_session_id:
        return RouterResponse(
            status="failed",
            response="session_id is required for file uploads.",
            error=ErrorDetail(code="MISSING_SESSION", message="Provide session_id in X-Session-Id header or form data"),
        )

    if not files:
        return RouterResponse(
            session_id=final_session_id,
            status="failed",
            response="No files provided.",
            error=ErrorDetail(code="NO_FILES", message="Upload at least one file"),
        )

    result = await router_orchestrator.handle_file_upload(
        session_id=final_session_id,
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
    """Confirm or cancel a HITL step. Session ID required."""
    user_id = user_context.user_id
    org_id = user_context.raw_payload.get("organizationId", "")
    jwt_token = request.headers.get("Authorization", "")
    
    # Get session ID from header (primary) or body (fallback)
    final_session_id = request.headers.get("X-Session-Id") or body.session_id
    
    # Session ID is required
    if not final_session_id:
        return RouterResponse(
            status="failed",
            response="Session ID is required. Provide X-Session-Id header.",
            error=ErrorDetail(code="MISSING_SESSION_ID", message="X-Session-Id header or body.session_id is required"),
        )

    result = await router_orchestrator.handle_confirmation(
        session_id=final_session_id,
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


# ============== File Middleware Endpoints ==============

@app.post("/api/files/upload")
async def upload_files(
    files: List[UploadFile],
    user_context: UserContext = Depends(get_current_user),
    session_id: Optional[str] = Form(None)
):
    """
    Standalone file upload endpoint.
    Uploads files to middleware, returns file_ids for later use.
    """
    uploaded = await file_middleware.upload_files(
        files=files,
        user_id=user_context.user_id,
        org_id=user_context.raw_payload.get("organizationId", ""),
        session_id=session_id
    )
    
    return JSONResponse({
        "status": "success",
        "files": uploaded
    })


@app.get("/api/files/{file_id}/metadata")
async def get_file_metadata(
    file_id: str,
    user_context: UserContext = Depends(get_current_user)
):
    """Get file metadata without downloading."""
    metadata = file_middleware.get_file_metadata_by_id(file_id)
    
    if not metadata:
        return JSONResponse(
            {"error": "File not found"},
            status_code=404
        )
    
    return JSONResponse(metadata)


@app.get("/api/files/{file_id}/download")
async def download_file(
    file_id: str,
    user_context: UserContext = Depends(get_current_user)
):
    """Download file by ID."""
    try:
        metadata = file_middleware.get_file_metadata_by_id(file_id)
        if not metadata:
            return JSONResponse({"error": "File not found"}, status_code=404)
        
        file_bytes = file_middleware.download_file(file_id, as_bytes=True)
        
        return Response(
            content=file_bytes,
            media_type=metadata["mime_type"],
            headers={
                "Content-Disposition": f"attachment; filename={metadata['original_name']}"
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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

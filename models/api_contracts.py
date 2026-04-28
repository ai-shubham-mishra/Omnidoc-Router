"""
Standardized API contracts (Pydantic schemas) for the LLM Router.
All router endpoints use these consistent request/response models.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ============== Request Models ==============

class ChatRequest(BaseModel):
    """Request body for POST /api/router/chat (legacy, use MessageRequest instead)"""
    session_id: Optional[str] = Field(None, description="Existing session ID or null for new conversation")
    message: str = Field(..., description="User's natural language message")


class MessageRequest(BaseModel):
    """
    Request body for POST /api/router/message (unified endpoint).
    Note: When using multipart/form-data, these become form fields:
    - session_id: string (optional)
    - message: string (required, can be empty if just uploading files)
    - files: file[] (optional, multiple files)
    """
    session_id: Optional[str] = Field(None, description="Existing session ID or null for new conversation")
    message: str = Field("", description="User's natural language message (can be empty for file-only uploads)")


class ConfirmationRequest(BaseModel):
    """Request body for POST /api/router/confirm"""
    session_id: str = Field(..., description="Session ID")
    action: str = Field(..., description="'confirm' or 'cancel'")
    message: Optional[str] = Field(None, description="Optional user message with confirmation")
    hitl_request: Optional[Dict[str, Any]] = Field(None, description="Modified HITL request from frontend with readonly/editable structure")


# ============== Response Models ==============

class FileUploaded(BaseModel):
    """Uploaded file metadata"""
    file_id: str
    original_name: str
    mime_type: str
    size_bytes: int
    uploaded_at: str

class WorkflowIdentified(BaseModel):
    """Workflow identification info in responses"""
    id: str
    name: str
    endpoint: str


class InputRequired(BaseModel):
    """Single input requirement"""
    field: str
    type: str
    label: str
    collected: bool


class ConfirmationData(BaseModel):
    """HIL confirmation data"""
    runId: str
    workflowId: str
    step_number: int
    data_to_review: Any


class ErrorDetail(BaseModel):
    """Error information"""
    code: str
    message: str
    details: Optional[Any] = None


class RouterResponse(BaseModel):
    """
    Standardized response for all router endpoints.
    Unified response for chat, message, and file upload.
    """
    session_id: Optional[str] = None
    status: str = Field(..., description="idle | collecting | ready_to_execute | executing | awaiting_confirmation | completed | failed | cancelled")
    response: str = Field(..., description="Natural language message for user")

    workflow_identified: Optional[WorkflowIdentified] = None
    inputs_required: Optional[List[InputRequired]] = None
    
    # File upload info
    files_uploaded: Optional[List[FileUploaded]] = None
    total_session_files: Optional[int] = None  # Total files in session context

    requires_confirmation: Optional[bool] = None
    confirmation_data: Optional[ConfirmationData] = None

    final_result: Optional[Any] = None
    error: Optional[ErrorDetail] = None


class SessionResponse(BaseModel):
    """Response for GET /api/router/session/:id"""
    session_id: str
    user_id: str
    org_id: str
    status: str  # Always "active" in new design
    created_at: str
    last_active: str
    conversation_history: List[Dict[str, Any]]
    current_workflow: Dict[str, Any]
    workflow_history: List[Dict[str, Any]]
    uploaded_files: List[Dict[str, Any]]


class DeleteSessionResponse(BaseModel):
    """Response for DELETE /api/router/session/:id"""
    status: str
    message: str
    session_id: str

"""
Standardized API contracts (Pydantic schemas) for the LLM Router.
All router endpoints use these consistent request/response models.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ============== Request Models ==============

class ChatRequest(BaseModel):
    """Request body for POST /api/router/chat"""
    session_id: Optional[str] = Field(None, description="Existing session ID or null for new conversation")
    message: str = Field(..., description="User's natural language message")


class UploadFieldName(BaseModel):
    """Field name mapping for file uploads"""
    field_name: str = Field(default="files", description="Which input field these files belong to")


class ConfirmationRequest(BaseModel):
    """Request body for POST /api/router/confirm"""
    session_id: str = Field(..., description="Session ID")
    action: str = Field(..., description="'confirm' or 'cancel'")
    message: Optional[str] = Field(None, description="Optional user message with confirmation")


# ============== Response Models ==============

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
    All fields except session_id, status, response are optional.
    """
    session_id: Optional[str] = None
    status: str = Field(..., description="collecting | executing | awaiting_confirmation | completed | failed | cancelled")
    response: str = Field(..., description="Natural language message for user")

    workflow_identified: Optional[WorkflowIdentified] = None
    inputs_required: Optional[List[InputRequired]] = None
    files_uploaded: Optional[int] = None

    requires_confirmation: Optional[bool] = None
    confirmation_data: Optional[ConfirmationData] = None

    final_result: Optional[Any] = None

    error: Optional[ErrorDetail] = None


class SessionResponse(BaseModel):
    """Response for GET /api/router/session/:id"""
    session_id: str
    user_id: str
    org_id: str
    created_at: str
    updated_at: str
    status: str
    conversation_history: List[Dict[str, Any]]
    workflow_context: Optional[Dict[str, Any]] = None
    execution_context: Optional[Dict[str, Any]] = None


class DeleteSessionResponse(BaseModel):
    """Response for DELETE /api/router/session/:id"""
    status: str
    message: str
    session_id: str

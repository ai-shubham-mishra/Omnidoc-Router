from pydantic import BaseModel
from typing import Optional, Dict, Any


class UserContext(BaseModel):
    """
    User context model containing authenticated user information.
    Populated from the JWT token payload.
    """
    user_id: str
    email: Optional[str] = None
    role: Optional[str] = None
    username: Optional[str] = None

    raw_payload: Dict[str, Any] = {}

    @classmethod
    def from_jwt_payload(cls, payload: Dict[str, Any]) -> 'UserContext':
        """Create UserContext from JWT payload."""
        user_id = (
            payload.get('userId') or
            payload.get('user_id') or
            payload.get('id') or
            payload.get('sub') or
            str(payload.get('iat', 'unknown'))
        )

        email = payload.get('email')
        role = payload.get('role') or payload.get('roles')
        username = payload.get('username') or payload.get('name')

        return cls(
            user_id=str(user_id),
            email=email,
            role=role,
            username=username,
            raw_payload=payload
        )


class AuthError(BaseModel):
    """Standard authentication error response"""
    success: bool = False
    message: str
    error_code: Optional[str] = None

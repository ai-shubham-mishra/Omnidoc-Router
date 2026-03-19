from fastapi import Request, HTTPException, Depends
from typing import Optional
from .models import UserContext
from .jwt_auth import verify_jwt_token, extract_token_from_header, JWTError, TokenExpiredError, TokenInvalidError


def get_current_user(request: Request) -> UserContext:
    """
    FastAPI dependency to get current authenticated user.

    Usage:
        @app.get("/my-endpoint")
        async def my_endpoint(user: UserContext = Depends(get_current_user)):
            pass
    """
    if hasattr(request.state, 'user') and request.state.user:
        return request.state.user

    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Authorization header missing")

        token = extract_token_from_header(auth_header)
        if not token:
            raise HTTPException(status_code=401, detail="Bearer token missing")

        payload = verify_jwt_token(token)
        user_context = UserContext.from_jwt_payload(payload)
        return user_context

    except TokenExpiredError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except TokenInvalidError:
        raise HTTPException(status_code=403, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=403, detail="Authentication failed")


def get_optional_user(request: Request) -> Optional[UserContext]:
    """FastAPI dependency to get current user if authenticated, None otherwise."""
    try:
        return get_current_user(request)
    except HTTPException:
        return None

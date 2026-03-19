from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, List, Optional
import logging

from .jwt_auth import verify_jwt_token, extract_token_from_header, TokenExpiredError, TokenInvalidError
from .models import UserContext

logger = logging.getLogger(__name__)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    Global JWT Authentication Middleware for FastAPI.
    Checks all incoming requests for JWT tokens, verifies them,
    and adds user context to request state.
    """

    def __init__(
        self,
        app,
        excluded_paths: Optional[List[str]] = None,
    ):
        super().__init__(app)

        self.excluded_paths = excluded_paths if excluded_paths is not None else [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/",
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        if self._should_skip_auth(path):
            return await call_next(request)

        try:
            user_context = await self._authenticate_request(request)
            request.state.user = user_context
            response = await call_next(request)
            return response

        except TokenExpiredError:
            logger.warning(f"Expired token for path: {path}")
            return self._create_auth_error_response("Token has expired", 401)

        except TokenInvalidError as e:
            logger.warning(f"Invalid token for path {path}: {str(e)}")
            return self._create_auth_error_response("Invalid or malformed token", 403)

        except Exception as e:
            logger.error(f"Auth error for path {path}: {str(e)}")
            return self._create_auth_error_response("Authentication failed", 403)

    def _should_skip_auth(self, path: str) -> bool:
        """Check if path is excluded from authentication."""
        for excluded in self.excluded_paths:
            if path == excluded or path.startswith(excluded + "/"):
                return True
        return False

    async def _authenticate_request(self, request: Request) -> UserContext:
        """Extract and verify JWT token from request."""
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise TokenInvalidError("Authorization header missing")

        token = extract_token_from_header(auth_header)
        if not token:
            raise TokenInvalidError("Bearer token missing")

        payload = verify_jwt_token(token)
        return UserContext.from_jwt_payload(payload)

    def _create_auth_error_response(self, message: str, status_code: int) -> JSONResponse:
        """Create a standardized authentication error response."""
        return JSONResponse(
            content={
                "success": False,
                "message": message,
                "error_code": "AUTH_ERROR",
            },
            status_code=status_code,
        )

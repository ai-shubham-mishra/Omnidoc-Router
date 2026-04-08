import jwt
import os
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv
from components.KeyVaultClient import get_secret

load_dotenv()


class JWTError(Exception):
    """Custom JWT error for authentication failures"""
    pass


class TokenExpiredError(JWTError):
    """Token has expired"""
    pass


class TokenInvalidError(JWTError):
    """Token is invalid"""
    pass


def get_jwt_secret() -> str:
    """Get JWT secret from Key Vault (with .env fallback)"""
    secret = get_secret('JWT_SECRET', default=os.getenv('JWT_SECRET'))
    if not secret:
        raise ValueError('JWT_SECRET is not defined in Key Vault or environment variables')
    return secret


def verify_jwt_token(token: str) -> Dict[str, Any]:
    """
    Verify JWT token and return decoded payload.

    Args:
        token: JWT token string

    Returns:
        Dict containing decoded user information

    Raises:
        TokenExpiredError: If token is expired
        TokenInvalidError: If token is invalid
    """
    try:
        secret = get_jwt_secret()

        decoded_payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"]
        )

        if 'exp' in decoded_payload:
            exp_timestamp = decoded_payload['exp']
            current_timestamp = datetime.now(timezone.utc).timestamp()

            if current_timestamp > exp_timestamp:
                raise TokenExpiredError("Token has expired")

        return decoded_payload

    except jwt.ExpiredSignatureError:
        raise TokenExpiredError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise TokenInvalidError(f"Invalid token: {str(e)}")
    except Exception as e:
        raise TokenInvalidError(f"Token verification failed: {str(e)}")


def extract_token_from_header(authorization_header: Optional[str]) -> Optional[str]:
    """Extract JWT token from Authorization header."""
    if not authorization_header:
        return None

    if not authorization_header.startswith('Bearer '):
        return None

    token = authorization_header.split(' ', 1)[1] if ' ' in authorization_header else None
    return token

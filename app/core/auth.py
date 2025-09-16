"""
Simple authentication utilities for the state management API.
For testing purposes, we'll use a simple token validation approach.
"""

import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings


async def verify_token(token: str) -> dict:
    """
    Verify JWT token and return user info.
    For testing, we'll accept any token that can be decoded with our secret.
    """
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


async def get_current_user(token: str) -> dict:
    """
    Get current user from token.
    """
    return await verify_token(token)


def create_test_token(user_id: str = "test_user") -> str:
    """
    Create a test token for development/testing purposes.
    """
    from app.core.security import create_access_token
    from datetime import timedelta

    token_data = {"sub": user_id, "user_id": user_id}
    return create_access_token(token_data, expires_delta=timedelta(hours=24))


# Security scheme
security = HTTPBearer()


async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Get current user from Bearer token"""
    return await verify_token(credentials.credentials)


async def get_current_admin_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Get current admin user with Auth0 role-based access control"""
    from app.auth.auth_utils import AuthUtils
    
    # Use Auth0 validation
    auth_utils = AuthUtils()
    current_user = await auth_utils.verify_token(credentials)
    
    # Check for admin role in custom claims
    namespace = "https://improv-today.com/"
    roles = current_user.get(f"{namespace}roles", [])
    
    # Require admin role for access
    if not roles or "admin" not in roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required - admin role not found in token"
        )
    
    return current_user
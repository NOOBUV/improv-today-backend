"""
Simple authentication utilities for the state management API.
For testing purposes, we'll use a simple token validation approach.
"""

import jwt
from fastapi import HTTPException, status
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
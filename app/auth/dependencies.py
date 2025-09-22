from typing import Dict
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials

from app.auth.auth_utils import AuthUtils, security

# Create auth utils instance
auth_utils = AuthUtils()

async def verify_protected_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """
    Dependency that verifies the Auth0 JWT token.
    
    Args:
        credentials: The HTTP Authorization credentials containing the token
        
    Returns:
        Dict: The decoded token payload containing user information
    """
    return await auth_utils.verify_token(credentials)
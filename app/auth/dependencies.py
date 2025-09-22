from typing import Dict, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.auth.auth_utils import AuthUtils, security
from app.core.database import get_db
from app.models.user import User

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

async def get_current_user(
    token_data: Dict = Depends(verify_protected_token),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency that gets the current user from the database based on Auth0 token.
    Creates a new user if one doesn't exist.
    
    Args:
        token_data: The verified token data from Auth0
        db: Database session
        
    Returns:
        User: The current user object
    """
    auth0_sub = token_data.get("sub")
    email = token_data.get("email")
    name = token_data.get("name")
    
    if not auth0_sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user identifier"
        )
    
    # Try to find existing user by Auth0 subject ID
    result = db.execute(select(User).where(User.auth0_sub == auth0_sub))
    user = result.scalar_one_or_none()
    
    if not user:
        # Create new user
        user = User(
            auth0_sub=auth0_sub,
            email=email,
            is_active=True,
            is_anonymous=False
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        # Update user information if it has changed
        updated = False
        if user.email != email and email:
            user.email = email
            updated = True
            
        if updated:
            db.commit()
            db.refresh(user)
    
    return user

async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Dependency that gets the current user if authenticated, otherwise returns None.
    Useful for endpoints that work for both authenticated and anonymous users.
    
    Args:
        credentials: Optional HTTP Authorization credentials
        db: Database session
        
    Returns:
        Optional[User]: The current user object or None if not authenticated
    """
    if not credentials:
        return None
        
    try:
        token_data = await auth_utils.verify_token(credentials)
        return await get_current_user(token_data, db)
    except HTTPException:
        return None
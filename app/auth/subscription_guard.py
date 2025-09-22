"""Subscription access control guards and middleware."""
import logging
from typing import Optional
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.auth.dependencies import get_current_user, get_current_user_optional
from app.models.user import User
from app.services.subscription_management_service import SubscriptionManagementService

logger = logging.getLogger(__name__)


class SubscriptionGuard:
    """
    Guard class for checking subscription access permissions.
    """
    
    def __init__(self):
        self._subscription_service = None
    
    @property
    def subscription_service(self):
        if self._subscription_service is None:
            self._subscription_service = SubscriptionManagementService()
        return self._subscription_service
    
    def verify_conversation_access(
        self,
        user: User,
        db: Session
    ) -> bool:
        """
        Verify if user has access to conversation features.
        
        Args:
            user: The user to check
            db: Database session
            
        Returns:
            bool: True if user has access, False otherwise
        """
        if not user:
            return False
            
        try:
            status = self.subscription_service.check_user_subscription_status(db, user.id)
            
            # Allow access for active subscriptions or active trials
            has_access = status.is_active or status.is_trial
            
            logger.info(
                f"Conversation access check for user {user.id}: "
                f"active={status.is_active}, trial={status.is_trial}, "
                f"status={status.status}, access={has_access}"
            )
            
            return has_access
            
        except Exception as e:
            logger.error(f"Error checking subscription status for user {user.id}: {str(e)}")
            return False


# Global subscription guard instance
subscription_guard = SubscriptionGuard()


def require_active_subscription(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency that requires an active subscription for access.
    
    Args:
        current_user: The authenticated user
        db: Database session
        
    Returns:
        User: The current user if they have an active subscription
        
    Raises:
        HTTPException: If user doesn't have an active subscription
    """
    has_access = subscription_guard.verify_conversation_access(current_user, db)
    
    if not has_access:
        # Get detailed status for error message
        try:
            subscription_status = subscription_guard.subscription_service.check_user_subscription_status(
                db, current_user.id
            )
            
            if subscription_status.status == "no_subscription":
                detail = "This feature requires an active subscription. Please subscribe to continue."
            elif subscription_status.status in ["canceled", "past_due", "unpaid"]:
                detail = "Your subscription is not active. Please update your payment method or renew your subscription."
            elif subscription_status.is_trial and not subscription_status.is_active:
                detail = "Your free trial has expired. Please subscribe to continue using this feature."
            else:
                detail = "Access denied. Please check your subscription status."
                
        except Exception:
            detail = "Access denied. Please check your subscription status."
        
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=detail,
            headers={"X-Subscription-Required": "true"}
        )
    
    return current_user


def subscription_access_optional(
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
) -> dict:
    """
    Dependency that provides subscription access information without requiring it.
    Useful for endpoints that have different behavior for subscribed vs non-subscribed users.
    
    Args:
        current_user: The optionally authenticated user
        db: Database session
        
    Returns:
        dict: Dictionary containing user and access information
    """
    if not current_user:
        return {
            "user": None,
            "has_subscription_access": False,
            "subscription_status": "not_authenticated"
        }
    
    has_access = subscription_guard.verify_conversation_access(current_user, db)
    
    try:
        subscription_status_obj = subscription_guard.subscription_service.check_user_subscription_status(
            db, current_user.id
        )
        subscription_status = subscription_status_obj.status
    except Exception:
        subscription_status = "error"
    
    return {
        "user": current_user,
        "has_subscription_access": has_access,
        "subscription_status": subscription_status
    }
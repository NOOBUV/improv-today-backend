"""Service for managing trial periods and expiration notifications."""
import logging
from typing import List, Optional
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import select, and_

from app.models.subscription import UserSubscription
from app.models.user import User
from app.services.subscription_management_service import SubscriptionManagementService

logger = logging.getLogger(__name__)


class TrialManagementService:
    """
    Service for managing trial periods, expiration notifications, 
    and subscription upgrade flows.
    """
    
    def __init__(self):
        self.subscription_service = SubscriptionManagementService()
    
    def get_users_with_expiring_trials(
        self, 
        db: Session, 
        days_until_expiry: int = 3
    ) -> List[UserSubscription]:
        """
        Get users whose trials are expiring within the specified number of days.
        
        Args:
            db: Database session
            days_until_expiry: Number of days before expiry to include
            
        Returns:
            List[UserSubscription]: Users with expiring trials
        """
        expiry_threshold = datetime.now(timezone.utc) + timedelta(days=days_until_expiry)
        
        result = db.execute(
            select(UserSubscription)
            .where(
                and_(
                    UserSubscription.status == "trialing",
                    UserSubscription.trial_end.isnot(None),
                    UserSubscription.trial_end <= expiry_threshold,
                    UserSubscription.trial_end > datetime.now(timezone.utc)
                )
            )
        )
        
        return result.scalars().all()
    
    def get_expired_trials(self, db: Session) -> List[UserSubscription]:
        """
        Get users whose trials have expired but subscription is still marked as trialing.
        
        Args:
            db: Database session
            
        Returns:
            List[UserSubscription]: Users with expired trials
        """
        now = datetime.now(timezone.utc)
        
        result = db.execute(
            select(UserSubscription)
            .where(
                and_(
                    UserSubscription.status == "trialing",
                    UserSubscription.trial_end.isnot(None),
                    UserSubscription.trial_end < now
                )
            )
        )
        
        return result.scalars().all()
    
    def expire_trial(
        self, 
        db: Session, 
        subscription: UserSubscription
    ) -> bool:
        """
        Expire a trial subscription and update its status.
        
        Args:
            db: Database session
            subscription: The subscription to expire
            
        Returns:
            bool: True if successfully expired, False otherwise
        """
        try:
            # Update subscription status
            subscription.status = "incomplete_expired"
            subscription.ended_at = datetime.now(timezone.utc)
            
            db.commit()
            db.refresh(subscription)
            
            logger.info(f"Expired trial for subscription {subscription.id} (user {subscription.user_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to expire trial for subscription {subscription.id}: {str(e)}")
            db.rollback()
            return False
    
    def process_expired_trials(self, db: Session) -> int:
        """
        Process all expired trials and update their status.
        
        Args:
            db: Database session
            
        Returns:
            int: Number of trials processed
        """
        expired_trials = self.get_expired_trials(db)
        processed_count = 0
        
        for trial in expired_trials:
            if self.expire_trial(db, trial):
                processed_count += 1
        
        logger.info(f"Processed {processed_count} expired trials")
        return processed_count
    
    def get_trial_status(
        self, 
        db: Session, 
        user_id: int
    ) -> dict:
        """
        Get detailed trial status for a user.
        
        Args:
            db: Database session
            user_id: User ID to check
            
        Returns:
            dict: Trial status information
        """
        subscription = self.subscription_service.get_user_active_subscription(db, user_id)
        
        if not subscription:
            # Check if user had any previous subscriptions
            result = db.execute(
                select(UserSubscription)
                .where(UserSubscription.user_id == user_id)
                .order_by(UserSubscription.created_at.desc())
            )
            previous_subscription = result.scalars().first()
            
            user_type = "previous_subscriber" if previous_subscription else "new_user"
            
            return {
                "user_type": user_type,
                "has_trial": False,
                "trial_active": False,
                "trial_expired": False,
                "days_remaining": 0,
                "trial_end": None,
                "can_access_conversation": False,
                "is_trial_active": False,
                "has_subscription": False
            }
        
        if subscription.status != "trialing" or not subscription.trial_end:
            # Check if they have an active subscription (paid)
            has_active_subscription = subscription.status == "active"
            user_type = "subscriber" if has_active_subscription else "previous_subscriber"
            
            return {
                "user_type": user_type,
                "has_trial": False,
                "trial_active": False,
                "trial_expired": False,
                "days_remaining": 0,
                "trial_end": None,
                "can_access_conversation": has_active_subscription,
                "is_trial_active": False,
                "has_subscription": has_active_subscription
            }
        
        now = datetime.now(timezone.utc)
        trial_active = subscription.trial_end > now
        trial_expired = subscription.trial_end <= now
        
        if trial_active:
            days_remaining = (subscription.trial_end - now).days
        else:
            days_remaining = 0
        
        return {
            "user_type": "trial_user",
            "has_trial": True,
            "trial_active": trial_active,
            "trial_expired": trial_expired,
            "days_remaining": days_remaining,
            "trial_end": subscription.trial_end,
            "subscription_id": subscription.id,
            "can_access_conversation": trial_active,  # User can access if trial is active
            "is_trial_active": trial_active,  # Alternative naming for frontend compatibility
            "has_subscription": subscription.status in ["active", "trialing"]  # Include subscription status
        }
    
    async def send_trial_expiration_warning(
        self, 
        db: Session, 
        subscription: UserSubscription,
        days_until_expiry: int
    ) -> bool:
        """
        Send trial expiration warning to user.
        
        Note: This is a placeholder for email/notification service integration.
        In a real implementation, this would integrate with your notification system.
        
        Args:
            db: Database session
            subscription: The subscription with expiring trial
            days_until_expiry: Number of days until trial expires
            
        Returns:
            bool: True if notification was sent successfully
        """
        try:
            # Get user information
            result = db.execute(select(User).where(User.id == subscription.user_id))
            user = result.scalar_one_or_none()
            
            if not user or not user.email:
                logger.warning(f"Cannot send trial warning: user {subscription.user_id} has no email")
                return False
            
            # Log the notification (in real implementation, send email/push notification)
            logger.info(
                f"TRIAL EXPIRATION WARNING: User {user.email} trial expires in {days_until_expiry} days "
                f"(subscription {subscription.id})"
            )
            
            # TODO: Integrate with email service
            # email_service.send_trial_expiration_warning(
            #     to=user.email,
            #     days_until_expiry=days_until_expiry,
            #     subscription_id=subscription.id
            # )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send trial warning for subscription {subscription.id}: {str(e)}")
            return False
    
    async def process_trial_expiration_warnings(self, db: Session) -> int:
        """
        Process and send trial expiration warnings for users with expiring trials.
        
        Args:
            db: Database session
            
        Returns:
            int: Number of warnings sent
        """
        warnings_sent = 0
        
        # Send warnings for trials expiring in 3 days
        expiring_3_days = await self.get_users_with_expiring_trials(db, days_until_expiry=3)
        for subscription in expiring_3_days:
            if await self.send_trial_expiration_warning(db, subscription, 3):
                warnings_sent += 1
        
        # Send warnings for trials expiring in 1 day
        expiring_1_day = await self.get_users_with_expiring_trials(db, days_until_expiry=1)
        for subscription in expiring_1_day:
            if await self.send_trial_expiration_warning(db, subscription, 1):
                warnings_sent += 1
        
        logger.info(f"Sent {warnings_sent} trial expiration warnings")
        return warnings_sent
    
    async def create_upgrade_checkout_session(
        self,
        db: Session,
        user_id: int,
        plan_price_id: str,
        success_url: str,
        cancel_url: str
    ) -> Optional[dict]:
        """
        Create a checkout session for upgrading from trial to paid subscription.
        
        Args:
            db: Database session
            user_id: User ID
            plan_price_id: Stripe price ID for the plan
            success_url: URL to redirect on successful upgrade
            cancel_url: URL to redirect on cancelled upgrade
            
        Returns:
            Optional[dict]: Checkout session information or None if failed
        """
        try:
            # Get user
            result = db.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()
            
            if not user or not user.email:
                logger.error(f"Cannot create upgrade session: user {user_id} not found or missing email")
                return None
            
            # Use the existing stripe service to create checkout
            from app.services.stripe_service import StripeService
            stripe_service = StripeService()
            
            metadata = {
                "user_id": str(user_id),
                "upgrade_from_trial": "true",
                "email": user.email
            }
            
            return await stripe_service.create_checkout_session(
                price_id=plan_price_id,
                customer_email=user.email,
                success_url=success_url,
                cancel_url=cancel_url,
                trial_days=None,  # No additional trial for upgrades
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to create upgrade checkout session for user {user_id}: {str(e)}")
            return None
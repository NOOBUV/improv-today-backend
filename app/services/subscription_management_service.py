"""Service for managing subscription plans and user subscriptions."""
import logging
from typing import Optional, List
from datetime import datetime, timezone
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, and_

from app.models.subscription import SubscriptionPlan, UserSubscription, PaymentRecord
from app.models.user import User
from app.services.stripe_service import StripeService
from app.schemas.stripe_schemas import SubscriptionStatus

logger = logging.getLogger(__name__)


class SubscriptionManagementService:
    """
    Service for managing subscription plans, user subscriptions, and integrating with Stripe.
    """

    def __init__(self):
        self._stripe_service = None
    
    @property
    def stripe_service(self):
        if self._stripe_service is None:
            self._stripe_service = StripeService()
        return self._stripe_service

    def create_subscription_plan(
        self,
        db: Session,
        name: str,
        description: str,
        price_cents: int,
        interval: str,
        stripe_price_id: str,
        stripe_product_id: Optional[str] = None,
        features: Optional[dict] = None,
        trial_period_days: int = 14
    ) -> SubscriptionPlan:
        """
        Create a new subscription plan.
        """
        plan = SubscriptionPlan(
            name=name,
            description=description,
            price_cents=price_cents,
            interval=interval,
            stripe_price_id=stripe_price_id,
            stripe_product_id=stripe_product_id,
            features=features or {},
            trial_period_days=trial_period_days,
            is_active=True
        )
        
        db.add(plan)
        db.commit()
        db.refresh(plan)
        
        logger.info(f"Created subscription plan: {plan.name} ({plan.id})")
        return plan

    def get_active_plans(self, db: Session) -> List[SubscriptionPlan]:
        """
        Get all active subscription plans.
        """
        result = db.execute(
            select(SubscriptionPlan)
            .where(SubscriptionPlan.is_active == True)
            .order_by(SubscriptionPlan.price_cents)
        )
        return result.scalars().all()

    def get_plan_by_stripe_price_id(
        self, 
        db: Session, 
        stripe_price_id: str
    ) -> Optional[SubscriptionPlan]:
        """
        Get subscription plan by Stripe price ID.
        """
        result = db.execute(
            select(SubscriptionPlan)
            .where(SubscriptionPlan.stripe_price_id == stripe_price_id)
        )
        return result.scalar_one_or_none()

    def create_user_subscription(
        self,
        db: Session,
        user_id: int,
        plan_id: int,
        stripe_customer_id: str,
        stripe_subscription_id: str,
        status: str = "active",
        current_period_start: Optional[datetime] = None,
        current_period_end: Optional[datetime] = None,
        trial_start: Optional[datetime] = None,
        trial_end: Optional[datetime] = None
    ) -> UserSubscription:
        """
        Create a new user subscription.
        """
        subscription = UserSubscription(
            user_id=user_id,
            plan_id=plan_id,
            stripe_customer_id=stripe_customer_id,
            stripe_subscription_id=stripe_subscription_id,
            status=status,
            current_period_start=current_period_start,
            current_period_end=current_period_end,
            trial_start=trial_start,
            trial_end=trial_end,
            started_at=datetime.now(timezone.utc),
            cancel_at_period_end=False
        )
        
        db.add(subscription)
        db.commit()
        db.refresh(subscription)
        
        logger.info(f"Created user subscription: {subscription.id} for user {user_id}")
        return subscription

    def get_user_active_subscription(
        self, 
        db: Session, 
        user_id: int
    ) -> Optional[UserSubscription]:
        """
        Get the user's active subscription.
        """
        result = db.execute(
            select(UserSubscription)
            .options(selectinload(UserSubscription.plan))
            .where(
                and_(
                    UserSubscription.user_id == user_id,
                    UserSubscription.status.in_(["active", "trialing"])
                )
            )
            .order_by(UserSubscription.created_at.desc())
        )
        # Use first() instead of scalar_one_or_none() to handle multiple records gracefully
        # This returns the most recent active/trialing subscription
        return result.scalars().first()

    def update_user_stripe_customer_id(
        self,
        db: Session,
        user_id: int,
        stripe_customer_id: str
    ) -> None:
        """
        Update user's Stripe customer ID.
        """
        result = db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        
        if user:
            user.stripe_customer_id = stripe_customer_id
            db.commit()
            logger.info(f"Updated user {user_id} with Stripe customer ID: {stripe_customer_id}")

    def check_user_subscription_status(
        self,
        db: Session,
        user_id: int
    ) -> SubscriptionStatus:
        """
        Check user's subscription status, combining database and Stripe data.
        """
        try:
            # Get user from database
            result = db.execute(
                select(User)
                .options(selectinload(User.subscription).selectinload(UserSubscription.plan))
                .where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                return SubscriptionStatus(
                    is_active=False,
                    is_trial=False,
                    status="user_not_found"
                )
            
            if not user.subscription:
                return SubscriptionStatus(
                    is_active=False,
                    is_trial=False,
                    status="no_subscription"
                )
            
            subscription = user.subscription
            now = datetime.now(timezone.utc)
            
            # Check if subscription is active
            is_active = subscription.status in ["active", "trialing"]
            is_trial = subscription.status == "trialing"
            
            # Check trial expiration
            trial_expired = False
            if subscription.trial_end and now > subscription.trial_end:
                trial_expired = True
                is_trial = False
            
            # Check subscription expiration
            subscription_expired = False
            if subscription.current_period_end and now > subscription.current_period_end:
                subscription_expired = True
                is_active = False
            
            return SubscriptionStatus(
                is_active=is_active and not subscription_expired,
                is_trial=is_trial and not trial_expired,
                trial_ends_at=subscription.trial_end,
                subscription_ends_at=subscription.current_period_end,
                plan_name=subscription.plan.name if subscription.plan else None,
                status=subscription.status
            )
            
        except Exception as e:
            logger.error(f"Failed to check subscription status for user {user_id}: {str(e)}")
            return SubscriptionStatus(
                is_active=False,
                is_trial=False,
                status="error"
            )

    def cancel_user_subscription(
        self,
        db: Session,
        user_id: int,
        at_period_end: bool = True
    ) -> Optional[UserSubscription]:
        """
        Cancel user's subscription.
        """
        subscription = self.get_user_active_subscription(db, user_id)
        
        if not subscription or not subscription.stripe_subscription_id:
            return None
        
        try:
            # Cancel in Stripe
            self.stripe_service.cancel_subscription(
                subscription.stripe_subscription_id, at_period_end
            )
            
            # Update local database
            if at_period_end:
                subscription.cancel_at_period_end = True
            else:
                subscription.status = "canceled"
                subscription.canceled_at = datetime.now(timezone.utc)
                subscription.ended_at = datetime.now(timezone.utc)
            
            db.commit()
            db.refresh(subscription)
            
            logger.info(f"Cancelled subscription {subscription.id} for user {user_id}")
            return subscription
            
        except Exception as e:
            logger.error(f"Failed to cancel subscription for user {user_id}: {str(e)}")
            return None

    def record_payment(
        self,
        db: Session,
        user_id: int,
        subscription_id: Optional[int],
        amount_cents: int,
        currency: str,
        status: str,
        payment_type: str,
        stripe_payment_intent_id: Optional[str] = None,
        stripe_invoice_id: Optional[str] = None,
        stripe_charge_id: Optional[str] = None,
        payment_method: Optional[str] = None,
        description: Optional[str] = None
    ) -> PaymentRecord:
        """
        Record a payment transaction.
        """
        payment = PaymentRecord(
            user_id=user_id,
            subscription_id=subscription_id,
            amount_cents=amount_cents,
            currency=currency,
            status=status,
            payment_type=payment_type,
            stripe_payment_intent_id=stripe_payment_intent_id,
            stripe_invoice_id=stripe_invoice_id,
            stripe_charge_id=stripe_charge_id,
            payment_method=payment_method,
            description=description,
            processed_at=datetime.now(timezone.utc)
        )
        
        db.add(payment)
        db.commit()
        db.refresh(payment)
        
        logger.info(f"Recorded payment: {payment.id} for user {user_id}")
        return payment

    def sync_subscription_from_stripe(
        self,
        db: Session,
        stripe_subscription_id: str
    ) -> Optional[UserSubscription]:
        """
        Sync subscription data from Stripe to local database.
        """
        try:
            # Get subscription from Stripe
            stripe_sub = self.stripe_service.get_subscription(stripe_subscription_id)
            
            # Find local subscription
            result = db.execute(
                select(UserSubscription)
                .where(UserSubscription.stripe_subscription_id == stripe_subscription_id)
            )
            local_sub = result.scalar_one_or_none()
            
            if not local_sub:
                logger.warning(f"No local subscription found for Stripe ID: {stripe_subscription_id}")
                return None
            
            # Update local subscription with Stripe data
            local_sub.status = stripe_sub.status
            local_sub.current_period_start = stripe_sub.current_period_start
            local_sub.current_period_end = stripe_sub.current_period_end
            local_sub.trial_end = stripe_sub.trial_end
            local_sub.cancel_at_period_end = stripe_sub.cancel_at_period_end
            
            db.commit()
            db.refresh(local_sub)
            
            logger.info(f"Synced subscription {local_sub.id} from Stripe")
            return local_sub
            
        except Exception as e:
            logger.error(f"Failed to sync subscription from Stripe: {str(e)}")
            return None

    def get_subscription_by_stripe_id(
        self,
        db: Session,
        stripe_subscription_id: str
    ) -> Optional[UserSubscription]:
        """
        Get subscription by Stripe subscription ID.
        """
        result = db.execute(
            select(UserSubscription)
            .options(selectinload(UserSubscription.plan))
            .where(UserSubscription.stripe_subscription_id == stripe_subscription_id)
        )
        return result.scalar_one_or_none()

    def get_user_by_stripe_customer_id(
        self,
        db: Session,
        stripe_customer_id: str
    ) -> Optional['User']:
        """
        Get user by Stripe customer ID.
        """
        from app.models.user import User
        result = db.execute(
            select(User)
            .where(User.stripe_customer_id == stripe_customer_id)
        )
        return result.scalar_one_or_none()

    def update_subscription_status(
        self,
        db: Session,
        subscription: UserSubscription,
        status: str,
        current_period_start: Optional[datetime] = None,
        current_period_end: Optional[datetime] = None,
        trial_end: Optional[datetime] = None,
        cancel_at_period_end: Optional[bool] = None
    ) -> UserSubscription:
        """
        Update subscription status and related fields.
        """
        subscription.status = status
        
        if current_period_start:
            subscription.current_period_start = current_period_start
        if current_period_end:
            subscription.current_period_end = current_period_end
        if trial_end:
            subscription.trial_end = trial_end
        if cancel_at_period_end is not None:
            subscription.cancel_at_period_end = cancel_at_period_end

        if status == "canceled":
            subscription.canceled_at = datetime.now(timezone.utc)
            subscription.ended_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(subscription)
        
        logger.info(f"Updated subscription {subscription.id} status to {status}")
        return subscription
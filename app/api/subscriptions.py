"""Subscription management API endpoints for Stripe integration."""
import logging
from typing import List
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.stripe_service import StripeService
from app.schemas.stripe_schemas import (
    CustomerCreate,
    CustomerResponse,
    SubscriptionCreate,
    SubscriptionResponse,
    PaymentIntentCreate,
    PaymentIntentResponse,
    SubscriptionStatus,
    WebhookEvent,
    SubscriptionPlan
)
from app.auth.dependencies import get_current_user
from app.services.subscription_management_service import SubscriptionManagementService
from app.services.trial_management_service import TrialManagementService
from app.services.background_tasks import background_task_manager
from app.models.user import User
from app.models.subscription import UserSubscription, SubscriptionPlan as DBSubscriptionPlan

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/subscriptions", tags=["subscriptions"])

# Initialize services
stripe_service = StripeService()
subscription_service = SubscriptionManagementService()
trial_service = TrialManagementService()


@router.get("/plans", response_model=List[SubscriptionPlan])
def get_subscription_plans(db: Session = Depends(get_db)):
    """
    Get all available subscription plans.
    """
    try:
        plans = subscription_service.get_active_plans(db)
        return plans
        
    except Exception as e:
        logger.error(f"Failed to get subscription plans: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get subscription plans"
        )


@router.post("/admin/create-test-plans")
def create_test_subscription_plans(db: Session = Depends(get_db)):
    """
    Create test subscription plans for development.
    This should be removed or secured in production.
    """
    try:
        # Check if plans already exist
        existing_plans = subscription_service.get_active_plans(db)
        if existing_plans:
            return {
                "success": True,
                "message": "Test plans already exist",
                "existing_plans": len(existing_plans)
            }
        
        # Create basic plan
        basic_plan = subscription_service.create_subscription_plan(
            db=db,
            name="Basic Plan",
            description="Access to Ava conversations with 14-day trial",
            price_cents=999,  # $9.99
            interval="month",
            stripe_price_id="price_test_basic_999",  # Placeholder for now
            features={"conversations_per_month": 100, "trial_days": 14}
        )
        
        # Create pro plan  
        pro_plan = subscription_service.create_subscription_plan(
            db=db,
            name="Pro Plan", 
            description="Unlimited conversations with premium features",
            price_cents=1999,  # $19.99
            interval="month",
            stripe_price_id="price_test_pro_1999",  # Placeholder for now
            features={"conversations_per_month": -1, "premium_features": True, "trial_days": 14}
        )
        
        return {
            "success": True,
            "message": "Test plans created successfully",
            "plans": [
                {"id": basic_plan.id, "name": basic_plan.name, "price": "$9.99/month"},
                {"id": pro_plan.id, "name": pro_plan.name, "price": "$19.99/month"}
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to create test plans: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create test plans: {str(e)}"
        )


@router.get("/subscription-status")
async def get_subscription_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed subscription status for the authenticated user.
    """
    try:
        trial_status = trial_service.get_trial_status(db, current_user.id)
        return trial_status
        
    except Exception as e:
        logger.error(f"Failed to get trial status for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get trial status"
        )


@router.post("/start-trial")
async def start_trial(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Start a 14-day trial for the authenticated user.
    """
    try:
        # Double-check for existing active subscriptions directly in database
        # to prevent race conditions and duplicates
        from sqlalchemy import select, and_
        existing_subscription = db.execute(
            select(UserSubscription)
            .where(
                and_(
                    UserSubscription.user_id == current_user.id,
                    UserSubscription.status.in_(["active", "trialing"])
                )
            )
        ).scalars().first()
        
        if existing_subscription:
            if existing_subscription.status == "active":
                return {
                    "success": False,
                    "message": "User already has an active subscription"
                }
            elif existing_subscription.status == "trialing":
                return {
                    "success": False, 
                    "message": "User already has an active trial"
                }
        
        # Check if user has any previous subscription (used trial already)
        previous_subscription = db.execute(
            select(UserSubscription)
            .where(UserSubscription.user_id == current_user.id)
        ).scalars().first()
        
        if previous_subscription:
            return {
                "success": False,
                "message": "Trial period has already been used"
            }
        
        # Create a new trial subscription
        from datetime import timedelta
        
        # Get the basic plan for the trial (both plans include 14-day trials)
        # We'll assign trials to the Basic Plan by default
        basic_plan = db.query(DBSubscriptionPlan).filter(
            DBSubscriptionPlan.name == "Basic Plan",
            DBSubscriptionPlan.is_active == True
        ).first()
        
        if not basic_plan:
            logger.error("No active Basic Plan found for trial creation")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No active subscription plans available for trial"
            )
        
        trial_end = datetime.now(timezone.utc) + timedelta(days=14)
        
        new_trial = UserSubscription(
            user_id=current_user.id,
            plan_id=basic_plan.id,  # Assign to Basic Plan for trial
            status="trialing",
            trial_start=datetime.now(timezone.utc),
            trial_end=trial_end,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        db.add(new_trial)
        db.commit()
        db.refresh(new_trial)
        
        logger.info(f"Started trial for user {current_user.id}, expires: {trial_end}")
        
        return {
            "success": True,
            "message": "14-day trial started successfully",
            "trial_end_date": trial_end.isoformat(),
            "days_remaining": 14
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to start trial for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start trial"
        )


@router.post("/upgrade-checkout")
async def create_upgrade_checkout_session(
    plan_price_id: str,
    success_url: str,
    cancel_url: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a checkout session for upgrading from trial to paid subscription.
    """
    try:
        checkout_session = await trial_service.create_upgrade_checkout_session(
            db=db,
            user_id=current_user.id,
            plan_price_id=plan_price_id,
            success_url=success_url,
            cancel_url=cancel_url
        )
        
        if not checkout_session:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create upgrade checkout session"
            )
        
        return checkout_session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create upgrade checkout for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create upgrade checkout session"
        )


from pydantic import BaseModel

class PlanUpdateRequest(BaseModel):
    stripe_price_id: str

@router.put("/admin/update-plan/{plan_id}")
def update_subscription_plan(
    plan_id: int,
    request: PlanUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    Update a subscription plan's Stripe price ID.
    This endpoint should be secured with admin authentication in production.
    """
    try:
        from sqlalchemy import select, update
        
        # Check if plan exists
        result = db.execute(select(DBSubscriptionPlan).where(DBSubscriptionPlan.id == plan_id))
        plan = result.scalar_one_or_none()
        
        if not plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Subscription plan with ID {plan_id} not found"
            )
        
        # Store old value before update
        old_stripe_price_id = plan.stripe_price_id
        
        # Update the stripe_price_id
        db.execute(
            update(SubscriptionPlan)
            .where(SubscriptionPlan.id == plan_id)
            .values(stripe_price_id=request.stripe_price_id)
        )
        db.commit()
        
        # Refresh to get updated data
        db.refresh(plan)
        
        logger.info(f"Updated plan {plan_id} with Stripe price ID: {request.stripe_price_id}")
        
        return {
            "success": True,
            "plan_id": plan.id,
            "name": plan.name,
            "old_stripe_price_id": old_stripe_price_id,
            "new_stripe_price_id": request.stripe_price_id,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update plan {plan_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update subscription plan: {str(e)}"
        )


@router.post("/admin/process-trials")
def admin_process_trials(
    db: Session = Depends(get_db)
):
    """
    Administrative endpoint to manually process trial expirations and send notifications.
    This endpoint should be secured with admin authentication in production.
    """
    try:
        # Process expired trials
        expired_count = trial_service.process_expired_trials(db)
        
        # Send trial expiration warnings (this stays async for now due to email/notification calls)
        warnings_sent = 0  # Placeholder - this would need async handling for notifications
        
        return {
            "success": True,
            "expired_trials_processed": expired_count,
            "warnings_sent": warnings_sent,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to process trials: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process trial management tasks"
        )


@router.post("/admin/cleanup-duplicate-trials")
def admin_cleanup_duplicate_trials(
    db: Session = Depends(get_db)
):
    """
    Administrative endpoint to clean up duplicate trial subscriptions.
    Keeps the most recent trial for each user and removes older duplicates.
    This endpoint should be secured with admin authentication in production.
    """
    try:
        from sqlalchemy import select, and_, func
        # Find users with multiple active/trialing subscriptions
        subquery = (
            select(
                UserSubscription.user_id,
                func.count(UserSubscription.id).label('count'),
                func.max(UserSubscription.id).label('latest_id')
            )
            .where(UserSubscription.status.in_(["active", "trialing"]))
            .group_by(UserSubscription.user_id)
            .having(func.count(UserSubscription.id) > 1)
            .subquery()
        )
        
        # Get users with duplicates
        users_with_duplicates = db.execute(select(subquery)).fetchall()
        
        cleaned_users = []
        total_removed = 0
        
        for user_row in users_with_duplicates:
            user_id = user_row.user_id
            latest_id = user_row.latest_id
            
            # Get all subscriptions for this user except the latest one
            old_subscriptions = db.execute(
                select(UserSubscription)
                .where(
                    and_(
                        UserSubscription.user_id == user_id,
                        UserSubscription.status.in_(["active", "trialing"]),
                        UserSubscription.id != latest_id
                    )
                )
            ).scalars().all()
            
            # Remove old duplicates
            for old_sub in old_subscriptions:
                db.delete(old_sub)
                total_removed += 1
            
            cleaned_users.append({
                "user_id": user_id,
                "removed_subscriptions": len(old_subscriptions),
                "kept_subscription_id": latest_id
            })
        
        db.commit()
        
        return {
            "success": True,
            "users_cleaned": len(cleaned_users),
            "total_subscriptions_removed": total_removed,
            "cleanup_details": cleaned_users,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to cleanup duplicate trials: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup duplicate trials: {str(e)}"
        )


@router.post("/customers", response_model=CustomerResponse)
async def create_customer(
    customer_data: CustomerCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new Stripe customer for the authenticated user.
    """
    try:
        # Add user ID to metadata for tracking
        customer_data.metadata["user_id"] = str(current_user.id)
        
        customer = await stripe_service.create_customer(customer_data)
        
        # Store customer ID in user model
        subscription_service.update_user_stripe_customer_id(
            db, current_user.id, customer.stripe_customer_id
        )
        
        return customer
        
    except Exception as e:
        logger.error(f"Failed to create customer for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create customer"
        )


@router.post("/checkout", response_model=dict)
async def create_checkout_session(
    price_id: str,
    success_url: str,
    cancel_url: str,
    trial_days: int = 14,
    current_user: User = Depends(get_current_user)
):
    """
    Create a Stripe checkout session for subscription signup.
    """
    try:
        metadata = {
            "user_id": str(current_user.id),
            "email": current_user.email
        }
        
        session = await stripe_service.create_checkout_session(
            price_id=price_id,
            customer_email=current_user.email,
            success_url=success_url,
            cancel_url=cancel_url,
            trial_days=trial_days,
            metadata=metadata
        )
        
        return session
        
    except Exception as e:
        logger.error(f"Failed to create checkout session for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create checkout session"
        )


@router.post("/payment-intents", response_model=PaymentIntentResponse)
async def create_payment_intent(
    payment_data: PaymentIntentCreate,
    current_user: User = Depends(get_current_user)
):
    """
    Create a payment intent for one-time payments.
    """
    try:
        # Add user metadata
        payment_data.metadata["user_id"] = str(current_user.id)
        
        payment_intent = await stripe_service.create_payment_intent(payment_data)
        
        return payment_intent
        
    except Exception as e:
        logger.error(f"Failed to create payment intent for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create payment intent"
        )


@router.get("/status", response_model=SubscriptionStatus)
async def get_subscription_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the current subscription status for the authenticated user.
    """
    try:
        return await subscription_service.check_user_subscription_status(db, current_user.id)
        
    except Exception as e:
        logger.error(f"Failed to get subscription status for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get subscription status"
        )


@router.get("/{subscription_id}", response_model=SubscriptionResponse)
async def get_subscription(
    subscription_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get subscription details by ID.
    """
    try:
        subscription = await stripe_service.get_subscription(subscription_id)
        
        # TODO: Verify subscription belongs to current user
        
        return subscription
        
    except Exception as e:
        logger.error(f"Failed to get subscription {subscription_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subscription not found"
        )


@router.delete("/{subscription_id}", response_model=SubscriptionResponse)
async def cancel_subscription(
    subscription_id: str,
    at_period_end: bool = True,
    current_user: User = Depends(get_current_user)
):
    """
    Cancel a subscription.
    """
    try:
        # TODO: Verify subscription belongs to current user
        
        subscription = await stripe_service.cancel_subscription(
            subscription_id, at_period_end
        )
        
        return subscription
        
    except Exception as e:
        logger.error(f"Failed to cancel subscription {subscription_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel subscription"
        )


@router.post("/webhooks/stripe")
async def handle_stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """
    Handle Stripe webhook events.
    
    This endpoint processes various Stripe events like:
    - customer.subscription.created
    - customer.subscription.updated
    - customer.subscription.deleted
    - invoice.payment_succeeded
    - invoice.payment_failed
    """
    try:
        # Verify webhook signature
        event = await stripe_service.verify_webhook_signature(request)
        
        # Process different event types
        if event.type == "customer.subscription.created":
            await _handle_subscription_created(event, db)
        elif event.type == "customer.subscription.updated":
            await _handle_subscription_updated(event, db)
        elif event.type == "customer.subscription.deleted":
            await _handle_subscription_deleted(event, db)
        elif event.type == "invoice.payment_succeeded":
            await _handle_payment_succeeded(event, db)
        elif event.type == "invoice.payment_failed":
            await _handle_payment_failed(event, db)
        else:
            logger.info(f"Unhandled webhook event type: {event.type}")
        
        return {"status": "success"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook processing failed"
        )


async def _handle_subscription_created(event: WebhookEvent, db: Session = None):
    """Handle subscription created webhook event."""
    subscription_data = event.data["object"]
    stripe_subscription_id = subscription_data["id"]
    customer_id = subscription_data["customer"]
    
    logger.info(f"Subscription created: {stripe_subscription_id} for customer {customer_id}")
    
    try:
        if not db:
            from app.core.database import get_db
            db = next(get_db())
        
        # Find user by Stripe customer ID
        user = subscription_service.get_user_by_stripe_customer_id(db, customer_id)
        if not user:
            logger.warning(f"No user found for Stripe customer ID: {customer_id}")
            return
        
        # Check if subscription already exists
        existing_subscription = subscription_service.get_subscription_by_stripe_id(
            db, stripe_subscription_id
        )
        
        if existing_subscription:
            logger.info(f"Subscription {stripe_subscription_id} already exists, updating status")
            subscription_service.update_subscription_status(
                db=db,
                subscription=existing_subscription,
                status=subscription_data["status"],
                current_period_start=datetime.fromtimestamp(
                    subscription_data["current_period_start"], tz=timezone.utc
                ) if subscription_data.get("current_period_start") else None,
                current_period_end=datetime.fromtimestamp(
                    subscription_data["current_period_end"], tz=timezone.utc
                ) if subscription_data.get("current_period_end") else None,
                trial_end=datetime.fromtimestamp(
                    subscription_data["trial_end"], tz=timezone.utc
                ) if subscription_data.get("trial_end") else None,
            )
        else:
            logger.info(f"New subscription created via webhook for user {user.id}")
        
        logger.info(f"Successfully processed subscription created for user {user.id}")
        
    except Exception as e:
        logger.error(f"Failed to handle subscription created webhook: {str(e)}")
        raise


async def _handle_subscription_updated(event: WebhookEvent, db: Session = None):
    """Handle subscription updated webhook event."""
    subscription_data = event.data["object"]
    stripe_subscription_id = subscription_data["id"]
    
    logger.info(f"Subscription updated: {stripe_subscription_id}")
    
    try:
        if not db:
            from app.core.database import get_db
            db = next(get_db())
        
        # Find subscription by Stripe ID
        local_subscription = subscription_service.get_subscription_by_stripe_id(
            db, stripe_subscription_id
        )
        
        if not local_subscription:
            logger.warning(f"No local subscription found for Stripe ID: {stripe_subscription_id}")
            return
        
        # Update subscription with new status and details
        subscription_service.update_subscription_status(
            db=db,
            subscription=local_subscription,
            status=subscription_data["status"],
            current_period_start=datetime.fromtimestamp(
                subscription_data["current_period_start"], tz=timezone.utc
            ) if subscription_data.get("current_period_start") else None,
            current_period_end=datetime.fromtimestamp(
                subscription_data["current_period_end"], tz=timezone.utc
            ) if subscription_data.get("current_period_end") else None,
            trial_end=datetime.fromtimestamp(
                subscription_data["trial_end"], tz=timezone.utc
            ) if subscription_data.get("trial_end") else None,
            cancel_at_period_end=subscription_data.get("cancel_at_period_end", False)
        )
        
        logger.info(f"Successfully updated subscription {local_subscription.id} status to {subscription_data['status']}")
        
    except Exception as e:
        logger.error(f"Failed to handle subscription updated webhook: {str(e)}")
        raise


async def _handle_subscription_deleted(event: WebhookEvent, db: Session = None):
    """Handle subscription deleted webhook event."""
    subscription_data = event.data["object"]
    stripe_subscription_id = subscription_data["id"]
    
    logger.info(f"Subscription deleted: {stripe_subscription_id}")
    
    try:
        if not db:
            from app.core.database import get_db
            db = next(get_db())
        
        # Find local subscription
        local_subscription = subscription_service.get_subscription_by_stripe_id(
            db, stripe_subscription_id
        )
        
        if not local_subscription:
            logger.warning(f"No local subscription found for Stripe ID: {stripe_subscription_id}")
            return
        
        # Update subscription status to canceled
        subscription_service.update_subscription_status(
            db=db,
            subscription=local_subscription,
            status="canceled"
        )
        
        logger.info(f"Successfully canceled subscription {local_subscription.id}")
        
    except Exception as e:
        logger.error(f"Failed to handle subscription deleted webhook: {str(e)}")
        raise


async def _handle_payment_succeeded(event: WebhookEvent, db: Session = None):
    """Handle successful payment webhook event."""
    invoice_data = event.data["object"]
    invoice_id = invoice_data["id"]
    
    logger.info(f"Payment succeeded for invoice: {invoice_id}")
    
    try:
        if not db:
            from app.core.database import get_db
            db = next(get_db())
        
        # Get subscription ID from invoice
        stripe_subscription_id = invoice_data.get("subscription")
        if not stripe_subscription_id:
            logger.info(f"Invoice {invoice_id} not associated with a subscription")
            return
        
        # Find local subscription
        local_subscription = subscription_service.get_subscription_by_stripe_id(
            db, stripe_subscription_id
        )
        
        if not local_subscription:
            logger.warning(f"No local subscription found for invoice {invoice_id}")
            return
        
        # TODO: Record the payment (requires AsyncSession)
        # For now, focusing on subscription status update
        logger.info(f"Payment of {invoice_data.get('amount_paid', 0)} {invoice_data.get('currency', 'usd')} succeeded for subscription {local_subscription.id}")
        
        # Ensure subscription is marked as active if payment succeeded
        if local_subscription.status != "active":
            subscription_service.update_subscription_status(
                db=db,
                subscription=local_subscription,
                status="active"
            )
        
        logger.info(f"Successfully processed payment for subscription {local_subscription.id}")
        
    except Exception as e:
        logger.error(f"Failed to handle payment succeeded webhook: {str(e)}")
        raise


async def _handle_payment_failed(event: WebhookEvent, db: Session = None):
    """Handle failed payment webhook event."""
    invoice_data = event.data["object"]
    invoice_id = invoice_data["id"]
    
    logger.info(f"Payment failed for invoice: {invoice_id}")
    
    try:
        if not db:
            from app.core.database import get_db
            db = next(get_db())
        
        # Get subscription ID from invoice
        stripe_subscription_id = invoice_data.get("subscription")
        if not stripe_subscription_id:
            logger.info(f"Invoice {invoice_id} not associated with a subscription")
            return
        
        # Find local subscription
        local_subscription = subscription_service.get_subscription_by_stripe_id(
            db, stripe_subscription_id
        )
        
        if not local_subscription:
            logger.warning(f"No local subscription found for failed payment invoice {invoice_id}")
            return
        
        # Update subscription status to past_due if payment failed
        if local_subscription.status == "active":
            subscription_service.update_subscription_status(
                db=db,
                subscription=local_subscription,
                status="past_due"
            )
            logger.info(f"Updated subscription {local_subscription.id} to past_due due to payment failure")
        
        logger.info(f"Successfully processed payment failed for subscription {local_subscription.id}")
        
    except Exception as e:
        logger.error(f"Failed to handle payment failed webhook: {str(e)}")
        raise
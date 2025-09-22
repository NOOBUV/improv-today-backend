"""Stripe service for payment processing and subscription management.

Implements modern async patterns and 2025 best practices for Stripe integration.
"""
import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import stripe
from stripe.error import StripeError, CardError, SignatureVerificationError
from fastapi import HTTPException, Request

from app.schemas.stripe_schemas import (
    CustomerCreate,
    CustomerResponse,
    SubscriptionCreate,
    SubscriptionResponse,
    PaymentIntentCreate,
    PaymentIntentResponse,
    SubscriptionStatus,
    WebhookEvent
)

logger = logging.getLogger(__name__)


class StripeService:
    """
    Stripe service implementing async patterns and modern best practices.
    Handles customers, subscriptions, payments, and webhook processing.
    """
    
    def __init__(self):
        """Initialize Stripe service with API key from environment."""
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        self.webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        
        if not stripe.api_key:
            raise ValueError("STRIPE_SECRET_KEY environment variable is required")
        
        logger.info("Stripe service initialized successfully")

    async def create_customer(self, customer_data: CustomerCreate) -> CustomerResponse:
        """
        Create a new Stripe customer with async error handling.
        
        Args:
            customer_data: Customer creation data
            
        Returns:
            CustomerResponse with Stripe customer details
            
        Raises:
            HTTPException: If customer creation fails
        """
        try:
            customer = stripe.Customer.create(
                email=customer_data.email,
                name=customer_data.name,
                metadata=customer_data.metadata
            )
            
            logger.info(f"Created Stripe customer: {customer.id} for {customer_data.email}")
            
            return CustomerResponse(
                stripe_customer_id=customer.id,
                email=customer.email,
                name=customer.name
            )
            
        except StripeError as e:
            logger.error(f"Failed to create Stripe customer: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Customer creation failed: {str(e)}")

    async def create_checkout_session(
        self,
        price_id: str,
        customer_email: str,
        success_url: str,
        cancel_url: str,
        trial_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Create a Stripe checkout session for subscription signup.
        
        Args:
            price_id: Stripe price ID for the subscription
            customer_email: Customer email for the session
            success_url: URL to redirect on successful payment
            cancel_url: URL to redirect on cancelled payment
            trial_days: Number of trial days (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Dictionary with checkout session URL and ID
            
        Raises:
            HTTPException: If checkout session creation fails
        """
        try:
            session_params = {
                "payment_method_types": ["card"],
                "line_items": [{
                    "price": price_id,
                    "quantity": 1,
                }],
                "mode": "subscription",
                "customer_email": customer_email,
                "success_url": success_url,
                "cancel_url": cancel_url,
                "metadata": metadata or {}
            }
            
            # Add trial period if specified
            if trial_days:
                session_params["subscription_data"] = {
                    "trial_period_days": trial_days
                }
            
            checkout_session = stripe.checkout.Session.create(**session_params)
            
            logger.info(f"Created checkout session: {checkout_session.id} for {customer_email}")
            
            return {
                "checkout_url": checkout_session.url,
                "session_id": checkout_session.id
            }
            
        except StripeError as e:
            logger.error(f"Failed to create checkout session: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Checkout session creation failed: {str(e)}")

    async def create_payment_intent(
        self,
        payment_data: PaymentIntentCreate
    ) -> PaymentIntentResponse:
        """
        Create a payment intent for one-time payments.
        
        Args:
            payment_data: Payment intent creation data
            
        Returns:
            PaymentIntentResponse with client secret
            
        Raises:
            HTTPException: If payment intent creation fails
        """
        try:
            intent = stripe.PaymentIntent.create(
                amount=payment_data.amount,
                currency=payment_data.currency,
                customer=payment_data.customer_id,
                metadata=payment_data.metadata
            )
            
            logger.info(f"Created payment intent: {intent.id}")
            
            return PaymentIntentResponse(
                id=intent.id,
                client_secret=intent.client_secret,
                status=intent.status,
                amount=intent.amount,
                currency=intent.currency
            )
            
        except StripeError as e:
            logger.error(f"Failed to create payment intent: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Payment intent creation failed: {str(e)}")

    async def get_subscription(self, subscription_id: str) -> SubscriptionResponse:
        """
        Get subscription details from Stripe.
        
        Args:
            subscription_id: Stripe subscription ID
            
        Returns:
            SubscriptionResponse with subscription details
            
        Raises:
            HTTPException: If subscription retrieval fails
        """
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            return SubscriptionResponse(
                id=subscription.id,
                customer_id=subscription.customer,
                status=subscription.status,
                current_period_start=datetime.fromtimestamp(
                    subscription.current_period_start, tz=timezone.utc
                ),
                current_period_end=datetime.fromtimestamp(
                    subscription.current_period_end, tz=timezone.utc
                ),
                trial_end=datetime.fromtimestamp(
                    subscription.trial_end, tz=timezone.utc
                ) if subscription.trial_end else None,
                cancel_at_period_end=subscription.cancel_at_period_end
            )
            
        except StripeError as e:
            logger.error(f"Failed to retrieve subscription: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Subscription not found: {str(e)}")

    async def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True
    ) -> SubscriptionResponse:
        """
        Cancel a Stripe subscription.
        
        Args:
            subscription_id: Stripe subscription ID
            at_period_end: Whether to cancel at period end or immediately
            
        Returns:
            SubscriptionResponse with updated subscription details
            
        Raises:
            HTTPException: If subscription cancellation fails
        """
        try:
            if at_period_end:
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True
                )
            else:
                subscription = stripe.Subscription.cancel(subscription_id)
            
            logger.info(f"Cancelled subscription: {subscription_id}")
            
            return await self.get_subscription(subscription_id)
            
        except StripeError as e:
            logger.error(f"Failed to cancel subscription: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Subscription cancellation failed: {str(e)}")

    async def get_customer_subscriptions(self, customer_id: str) -> List[SubscriptionResponse]:
        """
        Get all subscriptions for a customer.
        
        Args:
            customer_id: Stripe customer ID
            
        Returns:
            List of SubscriptionResponse objects
            
        Raises:
            HTTPException: If subscription retrieval fails
        """
        try:
            subscriptions = stripe.Subscription.list(customer=customer_id)
            
            result = []
            for sub in subscriptions.data:
                result.append(SubscriptionResponse(
                    id=sub.id,
                    customer_id=sub.customer,
                    status=sub.status,
                    current_period_start=datetime.fromtimestamp(
                        sub.current_period_start, tz=timezone.utc
                    ),
                    current_period_end=datetime.fromtimestamp(
                        sub.current_period_end, tz=timezone.utc
                    ),
                    trial_end=datetime.fromtimestamp(
                        sub.trial_end, tz=timezone.utc
                    ) if sub.trial_end else None,
                    cancel_at_period_end=sub.cancel_at_period_end
                ))
            
            return result
            
        except StripeError as e:
            logger.error(f"Failed to retrieve customer subscriptions: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to retrieve subscriptions: {str(e)}")

    async def verify_webhook_signature(self, request: Request) -> WebhookEvent:
        """
        Verify Stripe webhook signature and return event data.
        
        Args:
            request: FastAPI request object with webhook data
            
        Returns:
            WebhookEvent with verified event data
            
        Raises:
            HTTPException: If signature verification fails
        """
        if not self.webhook_secret:
            raise HTTPException(status_code=500, detail="Webhook secret not configured")
            
        try:
            payload = await request.body()
            signature = request.headers.get("stripe-signature")
            
            if not signature:
                raise HTTPException(status_code=400, detail="Missing Stripe signature")
            
            event = stripe.Webhook.construct_event(
                payload, signature, self.webhook_secret
            )
            
            logger.info(f"Verified webhook event: {event['type']} - {event['id']}")
            
            return WebhookEvent(
                id=event["id"],
                type=event["type"],
                data=event["data"],
                created=event["created"]
            )
            
        except SignatureVerificationError as e:
            logger.error(f"Webhook signature verification failed: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid webhook signature")
        except Exception as e:
            logger.error(f"Webhook processing failed: {str(e)}")
            raise HTTPException(status_code=400, detail="Webhook processing failed")

    async def check_subscription_status(
        self,
        customer_id: str
    ) -> SubscriptionStatus:
        """
        Check the current subscription status for a customer.
        
        Args:
            customer_id: Stripe customer ID
            
        Returns:
            SubscriptionStatus with current status details
        """
        try:
            subscriptions = await self.get_customer_subscriptions(customer_id)
            
            if not subscriptions:
                return SubscriptionStatus(
                    is_active=False,
                    is_trial=False,
                    status="no_subscription"
                )
            
            # Get the most recent active subscription
            active_sub = None
            for sub in subscriptions:
                if sub.status in ["active", "trialing"]:
                    active_sub = sub
                    break
            
            if not active_sub:
                return SubscriptionStatus(
                    is_active=False,
                    is_trial=False,
                    status="inactive"
                )
            
            is_trial = active_sub.status == "trialing"
            now = datetime.now(timezone.utc)
            
            return SubscriptionStatus(
                is_active=True,
                is_trial=is_trial,
                trial_ends_at=active_sub.trial_end if is_trial else None,
                subscription_ends_at=active_sub.current_period_end,
                status=active_sub.status
            )
            
        except Exception as e:
            logger.error(f"Failed to check subscription status: {str(e)}")
            return SubscriptionStatus(
                is_active=False,
                is_trial=False,
                status="error"
            )
"""Pydantic schemas for Stripe integration and subscription management."""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class SubscriptionPlanBase(BaseModel):
    """Base schema for subscription plans."""
    name: str
    description: Optional[str] = None
    price_cents: int
    interval: str = Field(..., description="billing interval: 'month' or 'year'")
    stripe_price_id: str
    features: Optional[dict] = None
    is_active: bool = True


class SubscriptionPlanCreate(SubscriptionPlanBase):
    """Schema for creating subscription plans."""
    pass


class SubscriptionPlan(SubscriptionPlanBase):
    """Schema for subscription plan responses."""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    currency: str = "USD"
    interval_count: int = 1
    stripe_product_id: Optional[str] = None
    trial_period_days: Optional[int] = 14

    class Config:
        from_attributes = True


class CustomerCreate(BaseModel):
    """Schema for creating Stripe customers."""
    email: str = Field(..., description="Customer email address")
    name: Optional[str] = None
    metadata: Dict[str, Any] = {}


class CustomerResponse(BaseModel):
    """Schema for Stripe customer responses."""
    stripe_customer_id: str
    email: str
    name: Optional[str] = None


class SubscriptionCreate(BaseModel):
    """Schema for creating subscriptions."""
    customer_email: str
    price_id: str
    trial_days: Optional[int] = 14
    metadata: Dict[str, Any] = {}


class SubscriptionResponse(BaseModel):
    """Schema for subscription responses."""
    id: str
    customer_id: str
    status: str
    current_period_start: datetime
    current_period_end: datetime
    trial_end: Optional[datetime] = None
    cancel_at_period_end: bool = False


class WebhookEvent(BaseModel):
    """Schema for Stripe webhook events."""
    id: str
    type: str
    data: Dict[str, Any]
    created: int


class PaymentIntentCreate(BaseModel):
    """Schema for creating payment intents."""
    amount: int = Field(..., description="Amount in cents")
    currency: str = "usd"
    customer_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class PaymentIntentResponse(BaseModel):
    """Schema for payment intent responses."""
    id: str
    client_secret: str
    status: str
    amount: int
    currency: str


class SubscriptionStatus(BaseModel):
    """Schema for subscription status responses."""
    is_active: bool
    is_trial: bool
    trial_ends_at: Optional[datetime] = None
    subscription_ends_at: Optional[datetime] = None
    plan_name: Optional[str] = None
    status: str
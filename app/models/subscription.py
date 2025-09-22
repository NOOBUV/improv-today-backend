"""Database models for subscription and payment management."""
from sqlalchemy import Integer, String, DateTime, Boolean, Text, JSON, DECIMAL, ForeignKey, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime
from app.core.database import Base


class SubscriptionPlan(Base):
    """Model for subscription plans and pricing."""
    __tablename__ = "subscription_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False)  # Price in cents
    currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)
    interval: Mapped[str] = mapped_column(String(10), nullable=False)  # 'month', 'year'
    interval_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    
    # Stripe integration
    stripe_price_id: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    stripe_product_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Plan configuration
    features: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    trial_period_days: Mapped[Optional[int]] = mapped_column(Integer, default=14, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user_subscriptions: Mapped[List["UserSubscription"]] = relationship(
        "UserSubscription", back_populates="plan", cascade="all, delete-orphan"
    )


class UserSubscription(Base):
    """Model for user subscription status and details."""
    __tablename__ = "user_subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    plan_id: Mapped[int] = mapped_column(Integer, ForeignKey("subscription_plans.id"), nullable=False)
    
    # Stripe integration
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(100), index=True, nullable=True)
    stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String(100), unique=True, index=True, nullable=True)
    
    # Subscription status
    status: Mapped[str] = mapped_column(String(20), default="inactive", nullable=False)
    # Possible statuses: 'active', 'trialing', 'past_due', 'canceled', 'unpaid', 'incomplete', 'incomplete_expired', 'paused'
    
    # Billing periods
    current_period_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    current_period_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    trial_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    trial_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Subscription lifecycle
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    canceled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    cancel_at_period_end: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Metadata and tracking
    subscription_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="subscription")
    plan: Mapped["SubscriptionPlan"] = relationship("SubscriptionPlan", back_populates="user_subscriptions")
    payment_records: Mapped[List["PaymentRecord"]] = relationship(
        "PaymentRecord", back_populates="subscription", cascade="all, delete-orphan"
    )

    # Indexes for better query performance
    __table_args__ = (
        Index('ix_user_subscriptions_user_status', 'user_id', 'status'),
        Index('ix_user_subscriptions_stripe_customer', 'stripe_customer_id'),
        Index('ix_user_subscriptions_period_end', 'current_period_end'),
        Index('ix_user_subscriptions_trial_end', 'trial_end'),
    )


class PaymentRecord(Base):
    """Model for tracking payment transactions and invoices."""
    __tablename__ = "payment_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    subscription_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("user_subscriptions.id"), nullable=True
    )
    
    # Stripe integration
    stripe_payment_intent_id: Mapped[Optional[str]] = mapped_column(String(100), unique=True, index=True, nullable=True)
    stripe_invoice_id: Mapped[Optional[str]] = mapped_column(String(100), unique=True, index=True, nullable=True)
    stripe_charge_id: Mapped[Optional[str]] = mapped_column(String(100), index=True, nullable=True)
    
    # Payment details
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    # Possible statuses: 'pending', 'succeeded', 'failed', 'canceled', 'refunded'
    
    payment_method: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # 'card', 'bank_account', etc.
    payment_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'subscription', 'one_time'
    
    # Transaction details
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    failure_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    receipt_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Billing period (for subscription payments)
    billing_period_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    billing_period_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Refund information
    refunded_amount_cents: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    refunded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    payment_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Timestamps
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="payment_records")
    subscription: Mapped[Optional["UserSubscription"]] = relationship("UserSubscription", back_populates="payment_records")

    # Indexes for better query performance
    __table_args__ = (
        Index('ix_payment_records_user_status', 'user_id', 'status'),
        Index('ix_payment_records_subscription', 'subscription_id'),
        Index('ix_payment_records_processed_at', 'processed_at'),
        Index('ix_payment_records_type', 'payment_type'),
    )


# Update the User model to include subscription relationships
# This should be added to the existing User model
if TYPE_CHECKING:
    from app.models.user import User  # noqa: F401
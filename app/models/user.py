from sqlalchemy import Integer, String, DateTime, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional, List, TYPE_CHECKING
from app.core.database import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # Anonymous identity cookie UUID
    anon_uuid: Mapped[Optional[str]] = mapped_column(String, unique=True, index=True, nullable=True)
    # Auth0 subject identifier for authenticated users
    auth0_sub: Mapped[Optional[str]] = mapped_column(String, unique=True, index=True, nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String, unique=True, index=True, nullable=True)
    hashed_password: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_anonymous: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Optional[DateTime]] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    
    # Stripe integration
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String, unique=True, index=True, nullable=True)
    
    # Subscription relationships
    subscription: Mapped[Optional["UserSubscription"]] = relationship("UserSubscription", back_populates="user", uselist=False)
    payment_records: Mapped[List["PaymentRecord"]] = relationship("PaymentRecord", back_populates="user", cascade="all, delete-orphan")

if TYPE_CHECKING:
    from app.models.subscription import UserSubscription, PaymentRecord  # noqa: F401
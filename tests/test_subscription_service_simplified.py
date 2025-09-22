"""
Simplified tests for subscription service functionality.

Tests focus on the actual methods that exist in the service to provide
reliable validation for the QA gate requirements.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session

from app.services.subscription_management_service import SubscriptionManagementService
from app.models.subscription import UserSubscription, SubscriptionPlan, PaymentRecord
from app.models.user import User
from app.schemas.stripe_schemas import SubscriptionStatus


@pytest.fixture
def mock_db():
    """Mock database session."""
    return Mock(spec=Session)


@pytest.fixture
def subscription_service():
    """Subscription management service instance."""
    service = SubscriptionManagementService()
    service._stripe_service = Mock()
    return service


@pytest.fixture
def sample_user():
    """Sample user for testing."""
    user = User()
    user.id = 1
    user.email = "test@example.com"
    user.auth0_sub = "auth0|123"
    user.stripe_customer_id = "cus_test123"
    user.is_active = True
    user.is_anonymous = False
    return user


@pytest.fixture
def sample_plan():
    """Sample subscription plan."""
    plan = SubscriptionPlan()
    plan.id = 1
    plan.stripe_price_id = "price_test123"
    plan.name = "Premium"
    plan.price_cents = 1999
    plan.interval = "month"
    plan.trial_period_days = 7
    plan.is_active = True
    return plan


@pytest.fixture
def active_subscription(sample_user, sample_plan):
    """Active subscription for testing."""
    subscription = UserSubscription()
    subscription.id = 1
    subscription.user_id = sample_user.id
    subscription.plan_id = sample_plan.id
    subscription.stripe_subscription_id = "sub_test123"
    subscription.status = "active"
    subscription.current_period_start = datetime.now(timezone.utc)
    subscription.current_period_end = datetime.now(timezone.utc) + timedelta(days=30)
    subscription.created_at = datetime.now(timezone.utc)
    return subscription


class TestSubscriptionManagementServiceCore:
    """Test cases for core subscription management service methods."""

    def test_check_user_subscription_status_active(
        self, subscription_service, mock_db, sample_user, active_subscription, sample_plan
    ):
        """Test checking subscription status for user with active subscription."""
        # Set up proper relationships for the active subscription
        sample_user.subscription = active_subscription
        active_subscription.plan = sample_plan

        # Mock SQLAlchemy 2.0 style database query
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_user
        mock_db.execute.return_value = mock_result

        # Test
        status = subscription_service.check_user_subscription_status(mock_db, sample_user.id)

        # Assertions
        assert status.status == "active"
        assert status.is_active == True

    def test_check_user_subscription_status_no_subscription(
        self, subscription_service, mock_db, sample_user
    ):
        """Test checking subscription status for user without subscription."""
        # Set user with no subscription
        sample_user.subscription = None

        # Mock SQLAlchemy 2.0 style database query
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_user
        mock_db.execute.return_value = mock_result

        # Test
        status = subscription_service.check_user_subscription_status(mock_db, sample_user.id)

        # Assertions
        assert status.status == "no_subscription"
        assert status.is_active == False

    def test_get_user_active_subscription(
        self, subscription_service, mock_db, sample_user, active_subscription
    ):
        """Test retrieving user's active subscription."""
        # Mock SQLAlchemy 2.0 style database query
        mock_result = Mock()
        mock_result.scalars.return_value.first.return_value = active_subscription
        mock_db.execute.return_value = mock_result

        # Test
        result = subscription_service.get_user_active_subscription(mock_db, sample_user.id)

        # Assertions
        assert result is not None
        assert result.id == active_subscription.id
        assert result.status == "active"

    def test_get_user_active_subscription_none(
        self, subscription_service, mock_db, sample_user
    ):
        """Test retrieving user's active subscription when none exists."""
        # Mock SQLAlchemy 2.0 style database query returning None
        mock_result = Mock()
        mock_result.scalars.return_value.first.return_value = None
        mock_db.execute.return_value = mock_result

        # Test
        result = subscription_service.get_user_active_subscription(mock_db, sample_user.id)

        # Assertions
        assert result is None

    def test_create_user_subscription(
        self, subscription_service, mock_db, sample_user, sample_plan
    ):
        """Test creating a new user subscription."""
        # Mock database operations
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()

        # Test
        result = subscription_service.create_user_subscription(
            db=mock_db,
            user_id=sample_user.id,
            plan_id=sample_plan.id,
            stripe_customer_id=sample_user.stripe_customer_id,
            stripe_subscription_id="sub_new123",
            status="active"
        )

        # Assertions
        assert result is not None
        assert result.user_id == sample_user.id
        assert result.plan_id == sample_plan.id
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    def test_cancel_user_subscription(
        self, subscription_service, mock_db, active_subscription
    ):
        """Test canceling a user subscription."""
        # Mock database query
        mock_db.query.return_value.filter.return_value.first.return_value = active_subscription
        mock_db.commit = Mock()

        # Test
        result = subscription_service.cancel_user_subscription(
            mock_db, active_subscription.user_id
        )

        # Assertions
        assert result is not None
        assert result.cancel_at_period_end == True
        mock_db.commit.assert_called_once()

    def test_get_active_plans(
        self, subscription_service, mock_db, sample_plan
    ):
        """Test retrieving active subscription plans."""
        # Mock SQLAlchemy 2.0 style database query
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_plan]
        mock_db.execute.return_value = mock_result

        # Test
        result = subscription_service.get_active_plans(mock_db)

        # Assertions
        assert len(result) == 1
        assert result[0].id == sample_plan.id
        assert result[0].is_active == True

    def test_record_payment(
        self, subscription_service, mock_db, sample_user, active_subscription
    ):
        """Test recording a payment."""
        # Mock database operations
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()

        # Test
        result = subscription_service.record_payment(
            db=mock_db,
            user_id=sample_user.id,
            subscription_id=active_subscription.id,
            stripe_payment_intent_id="pi_test123",
            amount_cents=1999,
            currency="USD",
            status="succeeded",
            payment_type="subscription"
        )

        # Assertions
        assert result is not None
        assert result.user_id == sample_user.id
        assert result.amount_cents == 1999
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    def test_update_subscription_status(
        self, subscription_service, mock_db, active_subscription
    ):
        """Test updating subscription status."""
        # Mock database operations
        mock_db.commit = Mock()

        # Test
        result = subscription_service.update_subscription_status(
            mock_db, active_subscription, "past_due"
        )

        # Assertions
        assert result is not None
        assert result.status == "past_due"
        mock_db.commit.assert_called_once()

    def test_get_subscription_by_stripe_id(
        self, subscription_service, mock_db, active_subscription
    ):
        """Test retrieving subscription by Stripe ID."""
        # Mock SQLAlchemy 2.0 style database query
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = active_subscription
        mock_db.execute.return_value = mock_result

        # Test
        result = subscription_service.get_subscription_by_stripe_id(
            mock_db, active_subscription.stripe_subscription_id
        )

        # Assertions
        assert result is not None
        assert result.id == active_subscription.id
        assert result.stripe_subscription_id == active_subscription.stripe_subscription_id

    def test_service_error_handling(
        self, subscription_service, mock_db, sample_user
    ):
        """Test service error handling for database failures."""
        # Mock database error
        mock_db.query.side_effect = Exception("Database connection failed")

        # Test
        status = subscription_service.check_user_subscription_status(mock_db, sample_user.id)

        # Assertions - should return error status instead of raising
        assert status.status == "error"
        assert status.is_active == False
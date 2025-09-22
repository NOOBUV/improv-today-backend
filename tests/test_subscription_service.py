"""
Tests for subscription service functionality.

Comprehensive test suite covering subscription lifecycle, payment processing,
and business logic validation.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
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
def mock_stripe_service():
    """Mock Stripe service."""
    return Mock()


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
    plan.price_cents = 1999  # $19.99
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
    subscription.trial_start = datetime.now(timezone.utc) - timedelta(days=5)
    subscription.trial_end = datetime.now(timezone.utc) + timedelta(days=2)
    subscription.current_period_start = datetime.now(timezone.utc)
    subscription.current_period_end = datetime.now(timezone.utc) + timedelta(days=30)
    subscription.created_at = datetime.now(timezone.utc)
    return subscription


class TestSubscriptionManagementService:
    """Test cases for subscription management service."""

    @pytest.mark.asyncio
    async def test_check_user_subscription_status_active_trial(
        self, subscription_service, mock_db, sample_user, active_subscription
    ):
        """Test checking subscription status for user with active trial."""
        # Mock database query
        mock_db.query.return_value.filter.return_value.first.return_value = active_subscription

        # Test
        status = subscription_service.check_user_subscription_status(mock_db, sample_user.id)

        # Assertions
        assert status.status == "trialing"
        assert status.is_trial == True
        assert status.is_active == True
        assert status.trial_end is not None

    @pytest.mark.asyncio
    async def test_check_user_subscription_status_no_subscription(
        self, subscription_service, mock_db, sample_user
    ):
        """Test checking subscription status for user without subscription."""
        # Mock database query returning None
        mock_db.query.return_value.filter.return_value.first.return_value = None

        # Test
        status = subscription_service.check_user_subscription_status(mock_db, sample_user.id)

        # Assertions
        assert status.status == "no_subscription"
        assert status.is_trial == False
        assert status.is_active == False

    @pytest.mark.asyncio
    async def test_check_user_subscription_status_expired_trial(
        self, subscription_service, mock_db, sample_user, sample_plan
    ):
        """Test checking subscription status for user with expired trial."""
        # Create expired trial subscription
        expired_subscription = UserSubscription()
        expired_subscription.id = 1
        expired_subscription.user_id = sample_user.id
        expired_subscription.plan_id = sample_plan.id
        expired_subscription.stripe_subscription_id = "sub_expired123"
        expired_subscription.status = "trialing"
        expired_subscription.trial_start = datetime.now(timezone.utc) - timedelta(days=10)
        expired_subscription.trial_end = datetime.now(timezone.utc) - timedelta(days=2)
        expired_subscription.current_period_start = datetime.now(timezone.utc) - timedelta(days=10)
        expired_subscription.current_period_end = datetime.now(timezone.utc) - timedelta(days=2)
        expired_subscription.created_at = datetime.now(timezone.utc) - timedelta(days=10)

        mock_db.query.return_value.filter.return_value.first.return_value = expired_subscription

        # Test
        status = subscription_service.check_user_subscription_status(mock_db, sample_user.id)

        # Assertions
        assert status.is_trial == True
        assert status.is_active == False

    @pytest.mark.asyncio
    async def test_create_trial_subscription(
        self, subscription_service, mock_db, sample_user, sample_plan
    ):
        """Test creating a trial subscription."""
        # Mock Stripe service response
        subscription_service.stripe_service.create_subscription.return_value = Mock(
            id="sub_trial123",
            status="trialing",
            trial_end=int((datetime.now(timezone.utc) + timedelta(days=7)).timestamp()),
            current_period_start=int(datetime.now(timezone.utc).timestamp()),
            current_period_end=int((datetime.now(timezone.utc) + timedelta(days=7)).timestamp())
        )

        # Mock database operations
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()

        # Test using the correct method name
        result = await subscription_service.create_user_subscription(
            mock_db, sample_user.id, sample_plan.id
        )

        # Assertions
        assert result is not None
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        subscription_service.stripe_service.create_subscription.assert_called_once()

    @pytest.mark.asyncio
    async def test_upgrade_subscription(
        self, subscription_service, mock_db, active_subscription, sample_plan
    ):
        """Test upgrading an existing subscription."""
        # Create premium plan
        premium_plan = SubscriptionPlan()
        premium_plan.id = 2
        premium_plan.stripe_price_id = "price_premium123"
        premium_plan.name = "Premium Plus"
        premium_plan.price_cents = 2999
        premium_plan.interval = "month"
        premium_plan.trial_period_days = 0
        premium_plan.is_active = True

        # Mock Stripe service response
        subscription_service.stripe_service.modify_subscription.return_value = Mock(
            id=active_subscription.stripe_subscription_id,
            status="active",
            current_period_start=int(datetime.now(timezone.utc).timestamp()),
            current_period_end=int((datetime.now(timezone.utc) + timedelta(days=30)).timestamp())
        )

        # Mock database queries
        mock_db.query.return_value.filter.return_value.first.return_value = active_subscription
        mock_db.commit = Mock()

        # Test
        result = await subscription_service.upgrade_subscription(
            mock_db, active_subscription.user_id, premium_plan.id
        )

        # Assertions
        assert result is not None
        assert result.plan_id == premium_plan.id
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_subscription(
        self, subscription_service, mock_db, active_subscription
    ):
        """Test canceling a subscription."""
        # Mock Stripe service response
        subscription_service.stripe_service.cancel_subscription.return_value = Mock(
            id=active_subscription.stripe_subscription_id,
            status="canceled",
            cancel_at_period_end=True
        )

        # Mock database queries
        mock_db.query.return_value.filter.return_value.first.return_value = active_subscription
        mock_db.commit = Mock()

        # Test
        result = await subscription_service.cancel_subscription(
            mock_db, active_subscription.user_id
        )

        # Assertions
        assert result is not None
        assert result.status == "canceled"
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_payment_history(
        self, subscription_service, mock_db, sample_user
    ):
        """Test retrieving user payment history."""
        # Create mock payment records with proper field names
        payment_record_1 = PaymentRecord()
        payment_record_1.id = 1
        payment_record_1.user_id = sample_user.id
        payment_record_1.stripe_payment_intent_id = "pi_test123"
        payment_record_1.amount_cents = 1999
        payment_record_1.currency = "USD"
        payment_record_1.status = "succeeded"
        payment_record_1.payment_type = "subscription"
        payment_record_1.created_at = datetime.now(timezone.utc)

        payment_record_2 = PaymentRecord()
        payment_record_2.id = 2
        payment_record_2.user_id = sample_user.id
        payment_record_2.stripe_payment_intent_id = "pi_test456"
        payment_record_2.amount_cents = 1999
        payment_record_2.currency = "USD"
        payment_record_2.status = "succeeded"
        payment_record_2.payment_type = "subscription"
        payment_record_2.created_at = datetime.now(timezone.utc) - timedelta(days=30)

        payment_records = [payment_record_1, payment_record_2]

        # Mock database query
        mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = payment_records

        # Test
        result = subscription_service.get_user_payment_history(mock_db, sample_user.id)

        # Assertions
        assert len(result) == 2
        assert all(record.user_id == sample_user.id for record in result)
        assert all(record.status == "succeeded" for record in result)

    @pytest.mark.asyncio
    async def test_subscription_expiry_handling(
        self, subscription_service, mock_db, sample_user, sample_plan
    ):
        """Test handling of subscription expiry logic."""
        # Create nearly expired subscription
        nearly_expired = UserSubscription()
        nearly_expired.id = 1
        nearly_expired.user_id = sample_user.id
        nearly_expired.plan_id = sample_plan.id
        nearly_expired.stripe_subscription_id = "sub_expiring123"
        nearly_expired.status = "active"
        nearly_expired.trial_start = None
        nearly_expired.trial_end = None
        nearly_expired.current_period_start = datetime.now(timezone.utc) - timedelta(days=25)
        nearly_expired.current_period_end = datetime.now(timezone.utc) + timedelta(days=5)
        nearly_expired.created_at = datetime.now(timezone.utc) - timedelta(days=25)

        mock_db.query.return_value.filter.return_value.first.return_value = nearly_expired

        # Test
        status = subscription_service.check_user_subscription_status(mock_db, sample_user.id)

        # Assertions
        assert status.status == "active"
        assert status.is_active == True
        assert status.days_until_expiry is not None
        assert status.days_until_expiry <= 5
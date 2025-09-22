"""
Integration tests for Stripe webhook handling.

Tests validate webhook signature verification and basic webhook infrastructure.
"""
import pytest
import json
import hmac
import hashlib
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone
from fastapi import Request, HTTPException
from sqlalchemy.orm import Session

from app.services.stripe_service import StripeService
from app.models.user import User
from app.models.subscription import UserSubscription, SubscriptionPlan
from stripe.error import SignatureVerificationError


@pytest.fixture
def mock_db():
    """Mock database session."""
    return Mock(spec=Session)


@pytest.fixture
def stripe_service():
    """Stripe service instance with test configuration."""
    with patch.dict('os.environ', {
        'STRIPE_SECRET_KEY': 'sk_test_123',
        'STRIPE_WEBHOOK_SECRET': 'whsec_test_secret'
    }):
        service = StripeService()
        return service


@pytest.fixture
def sample_user():
    """Sample user with Stripe customer ID."""
    user = User()
    user.id = 1
    user.email = "webhook@example.com"
    user.auth0_sub = "auth0|webhook123"
    user.stripe_customer_id = "cus_webhook123"
    user.is_active = True
    user.is_anonymous = False
    return user


def create_stripe_signature(payload: str, secret: str, timestamp: int = None) -> str:
    """Create valid Stripe webhook signature for testing."""
    if timestamp is None:
        timestamp = int(time.time())

    signed_payload = f"{timestamp}.{payload}"
    signature = hmac.new(
        secret.encode('utf-8'),
        signed_payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    return f"t={timestamp},v1={signature}"


class TestStripeWebhookHandling:
    """Test Stripe webhook signature verification and infrastructure."""

    @pytest.mark.asyncio
    async def test_webhook_signature_verification_valid(self, stripe_service):
        """Test valid webhook signature verification."""
        payload = '{"id": "evt_test_webhook", "type": "customer.subscription.updated"}'
        secret = "whsec_test_secret"
        timestamp = int(time.time())
        signature = create_stripe_signature(payload, secret, timestamp)

        # Create mock request
        mock_request = Mock(spec=Request)
        mock_request.headers = {"stripe-signature": signature}

        # Mock the request.body() method to return the payload
        async def mock_body():
            return payload.encode()
        mock_request.body = mock_body

        with patch('stripe.Webhook.construct_event') as mock_construct:
            mock_construct.return_value = {
                "id": "evt_test_webhook",
                "type": "customer.subscription.updated",
                "data": {"object": {"id": "test"}},
                "created": int(time.time())
            }

            # Test
            event = await stripe_service.verify_webhook_signature(mock_request)

            # Assertions
            assert event is not None
            assert event.id == "evt_test_webhook"
            assert event.type == "customer.subscription.updated"
            mock_construct.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_signature_verification_invalid(self, stripe_service):
        """Test invalid webhook signature rejection."""
        payload = '{"id": "evt_test_webhook", "type": "customer.subscription.updated"}'
        invalid_signature = "t=123456789,v1=invalid_signature"

        # Create mock request
        mock_request = Mock(spec=Request)
        mock_request.headers = {"stripe-signature": invalid_signature}

        # Mock the request.body() method
        async def mock_body():
            return payload.encode()
        mock_request.body = mock_body

        with patch('stripe.Webhook.construct_event') as mock_construct:
            mock_construct.side_effect = SignatureVerificationError(
                message="Invalid signature",
                sig_header=invalid_signature
            )

            # Test that HTTPException is raised
            with pytest.raises(HTTPException) as exc_info:
                await stripe_service.verify_webhook_signature(mock_request)

            assert exc_info.value.status_code == 400
            assert "Invalid webhook signature" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_webhook_missing_signature(self, stripe_service):
        """Test webhook request missing signature header."""
        payload = '{"id": "evt_test_webhook", "type": "customer.subscription.updated"}'

        # Create mock request without signature header
        mock_request = Mock(spec=Request)
        mock_request.headers = {}

        # Mock the request.body() method
        async def mock_body():
            return payload.encode()
        mock_request.body = mock_body

        # Test that HTTPException is raised
        with pytest.raises(HTTPException) as exc_info:
            await stripe_service.verify_webhook_signature(mock_request)

        assert exc_info.value.status_code == 400
        assert "Webhook processing failed" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_webhook_without_secret_configured(self):
        """Test webhook verification when secret is not configured."""
        # Create service without webhook secret
        with patch.dict('os.environ', {
            'STRIPE_SECRET_KEY': 'sk_test_123'
            # No STRIPE_WEBHOOK_SECRET
        }):
            service = StripeService()
            service.webhook_secret = None

        payload = '{"id": "evt_test_webhook", "type": "customer.subscription.updated"}'
        mock_request = Mock(spec=Request)
        mock_request.headers = {"stripe-signature": "test_signature"}

        # Mock the request.body() method
        async def mock_body():
            return payload.encode()
        mock_request.body = mock_body

        # Test that HTTPException is raised
        with pytest.raises(HTTPException) as exc_info:
            await service.verify_webhook_signature(mock_request)

        assert exc_info.value.status_code == 500
        assert "Webhook secret not configured" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_webhook_signature_timestamp_tolerance(self, stripe_service):
        """Test webhook signature with different timestamp tolerances."""
        payload = '{"id": "evt_test_webhook", "type": "customer.subscription.updated"}'
        secret = "whsec_test_secret"

        # Test with old timestamp (5 minutes ago)
        old_timestamp = int(time.time()) - 300
        signature = create_stripe_signature(payload, secret, old_timestamp)

        mock_request = Mock(spec=Request)
        mock_request.headers = {"stripe-signature": signature}

        # Mock the request.body() method
        async def mock_body():
            return payload.encode()
        mock_request.body = mock_body

        with patch('stripe.Webhook.construct_event') as mock_construct:
            mock_construct.return_value = {
                "id": "evt_test_webhook",
                "type": "customer.subscription.updated",
                "data": {"object": {"id": "test"}},
                "created": old_timestamp
            }

            # Test
            event = await stripe_service.verify_webhook_signature(mock_request)

            # Assertions
            assert event is not None
            assert event.id == "evt_test_webhook"

    @pytest.mark.asyncio
    async def test_webhook_signature_with_multiple_signatures(self, stripe_service):
        """Test webhook signature verification with multiple signature versions."""
        payload = '{"id": "evt_test_webhook", "type": "customer.subscription.updated"}'
        secret = "whsec_test_secret"
        timestamp = int(time.time())

        # Create signature with multiple versions
        v1_signature = hmac.new(
            secret.encode('utf-8'),
            f"{timestamp}.{payload}".encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        # Simulate multiple signature format
        signature = f"t={timestamp},v1={v1_signature},v0=fake_signature"

        mock_request = Mock(spec=Request)
        mock_request.headers = {"stripe-signature": signature}

        # Mock the request.body() method
        async def mock_body():
            return payload.encode()
        mock_request.body = mock_body

        with patch('stripe.Webhook.construct_event') as mock_construct:
            mock_construct.return_value = {
                "id": "evt_test_webhook",
                "type": "customer.subscription.updated",
                "data": {"object": {"id": "test"}},
                "created": timestamp
            }

            # Test
            event = await stripe_service.verify_webhook_signature(mock_request)

            # Assertions
            assert event is not None
            assert event.id == "evt_test_webhook"

    @pytest.mark.asyncio
    async def test_webhook_malformed_json_handling(self, stripe_service):
        """Test webhook handling with malformed JSON payload."""
        payload = '{"id": "evt_test_webhook", "type": "customer.subscription.updated", malformed'
        secret = "whsec_test_secret"
        timestamp = int(time.time())
        signature = create_stripe_signature(payload, secret, timestamp)

        mock_request = Mock(spec=Request)
        mock_request.headers = {"stripe-signature": signature}

        # Mock the request.body() method
        async def mock_body():
            return payload.encode()
        mock_request.body = mock_body

        with patch('stripe.Webhook.construct_event') as mock_construct:
            mock_construct.side_effect = ValueError("Invalid JSON")

            # Test that HTTPException is raised
            with pytest.raises(HTTPException) as exc_info:
                await stripe_service.verify_webhook_signature(mock_request)

            assert exc_info.value.status_code == 400
            assert "Webhook processing failed" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_webhook_event_types(self, stripe_service):
        """Test webhook signature verification for different event types."""
        event_types = [
            "customer.subscription.created",
            "customer.subscription.updated",
            "customer.subscription.deleted",
            "payment_intent.succeeded",
            "payment_intent.payment_failed",
            "customer.subscription.trial_will_end"
        ]

        for event_type in event_types:
            payload = f'{{"id": "evt_test_{event_type}", "type": "{event_type}"}}'
            secret = "whsec_test_secret"
            timestamp = int(time.time())
            signature = create_stripe_signature(payload, secret, timestamp)

            mock_request = Mock(spec=Request)
            mock_request.headers = {"stripe-signature": signature}

            # Mock the request.body() method
            async def mock_body():
                return payload.encode()
            mock_request.body = mock_body

            with patch('stripe.Webhook.construct_event') as mock_construct:
                mock_construct.return_value = {
                    "id": f"evt_test_{event_type}",
                    "type": event_type,
                    "data": {"object": {"id": "test"}},
                    "created": timestamp
                }

                # Test
                event = await stripe_service.verify_webhook_signature(mock_request)

                # Assertions
                assert event is not None
                assert event.type == event_type

    @pytest.mark.asyncio
    async def test_webhook_signature_case_sensitivity(self, stripe_service):
        """Test webhook signature header case sensitivity."""
        payload = '{"id": "evt_test_webhook", "type": "customer.subscription.updated"}'
        secret = "whsec_test_secret"
        timestamp = int(time.time())
        signature = create_stripe_signature(payload, secret, timestamp)

        # Test with different case header
        mock_request = Mock(spec=Request)
        mock_request.headers = {"Stripe-Signature": signature}  # Capital case

        # Mock the request.body() method
        async def mock_body():
            return payload.encode()
        mock_request.body = mock_body

        # Test that missing signature is detected (case sensitive)
        with pytest.raises(HTTPException) as exc_info:
            await stripe_service.verify_webhook_signature(mock_request)

        assert exc_info.value.status_code == 400
        assert "Webhook processing failed" in str(exc_info.value.detail)
"""
Tests for simulation engine API endpoints.
Tests admin interfaces, monitoring, and control endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import json

from app.main import app
from app.models.user import User


class TestSimulationAdminAPI:
    """Test simulation admin API endpoints."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch('app.api.simulation.admin.get_current_user')
    @patch('app.api.simulation.admin.celery_app')
    def test_get_simulation_status_unauthorized(self, mock_celery, mock_get_user):
        """Test simulation status endpoint without authentication."""
        mock_get_user.return_value = None

        response = self.client.get("/api/simulation/admin/status")

        assert response.status_code == 401
        assert "Authentication required" in response.json()["detail"]

    @patch('app.api.simulation.admin.get_current_user')
    @patch('app.api.simulation.admin.celery_app')
    @patch('app.api.simulation.admin.get_async_session')
    def test_get_simulation_status_success(self, mock_get_session, mock_celery, mock_get_user):
        """Test successful simulation status retrieval."""
        # Mock authenticated user
        mock_user = User(id="test-user", email="test@example.com")
        mock_get_user.return_value = mock_user

        # Mock Celery inspect
        mock_inspect = MagicMock()
        mock_inspect.active.return_value = {"worker1": []}
        mock_inspect.reserved.return_value = {"worker1": []}
        mock_celery.control.inspect.return_value = mock_inspect

        # Mock database session
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar.return_value = datetime.utcnow()
        mock_get_session.return_value.__aiter__ = AsyncMock(return_value=iter([mock_session]))

        response = self.client.get("/api/simulation/admin/status")

        assert response.status_code == 200
        data = response.json()
        assert "is_running" in data
        assert "active_workers" in data
        assert "pending_tasks" in data
        assert "last_event_time" in data

    @patch('app.api.simulation.admin.get_current_user')
    def test_control_simulation_unauthorized(self, mock_get_user):
        """Test simulation control without authentication."""
        mock_get_user.return_value = None

        response = self.client.post(
            "/api/simulation/admin/control",
            json={"action": "start"}
        )

        assert response.status_code == 401

    @patch('app.api.simulation.admin.get_current_user')
    def test_control_simulation_invalid_action(self, mock_get_user):
        """Test simulation control with invalid action."""
        mock_user = User(id="test-user", email="test@example.com")
        mock_get_user.return_value = mock_user

        response = self.client.post(
            "/api/simulation/admin/control",
            json={"action": "invalid"}
        )

        assert response.status_code == 400
        assert "Invalid action" in response.json()["detail"]

    @patch('app.api.simulation.admin.get_current_user')
    def test_control_simulation_valid_actions(self, mock_get_user):
        """Test simulation control with valid actions."""
        mock_user = User(id="test-user", email="test@example.com")
        mock_get_user.return_value = mock_user

        for action in ["start", "stop", "restart"]:
            response = self.client.post(
                "/api/simulation/admin/control",
                json={"action": action}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["action"] == action
            assert data["status"] == "acknowledged"
            assert "timestamp" in data

    def test_health_check_public(self):
        """Test health check endpoint (public)."""
        with patch('app.api.simulation.admin.celery_app') as mock_celery:
            mock_inspect = MagicMock()
            mock_inspect.ping.return_value = {"worker1": "pong"}
            mock_celery.control.inspect.return_value = mock_inspect

            response = self.client.get("/api/simulation/admin/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["component"] == "simulation-engine"

    def test_health_check_unhealthy(self):
        """Test health check when service is unhealthy."""
        with patch('app.api.simulation.admin.celery_app') as mock_celery:
            mock_inspect = MagicMock()
            mock_inspect.ping.return_value = None
            mock_celery.control.inspect.return_value = mock_inspect

            response = self.client.get("/api/simulation/admin/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"

    @patch('app.api.simulation.admin.get_current_user')
    @patch('app.api.simulation.admin.celery_app')
    def test_get_metrics_success(self, mock_celery, mock_get_user):
        """Test metrics endpoint."""
        mock_user = User(id="test-user", email="test@example.com")
        mock_get_user.return_value = mock_user

        # Mock Celery inspect
        mock_inspect = MagicMock()
        mock_inspect.stats.return_value = {"worker1": {"total": 100}}
        mock_inspect.active.return_value = {"worker1": []}
        mock_inspect.scheduled.return_value = {"worker1": []}
        mock_inspect.reserved.return_value = {"worker1": []}
        mock_celery.control.inspect.return_value = mock_inspect

        response = self.client.get("/api/simulation/admin/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "worker_stats" in data
        assert "total_workers" in data
        assert "total_active_tasks" in data

    @patch('app.api.simulation.admin.get_current_user')
    @patch('app.api.simulation.admin.get_async_session')
    def test_get_recent_events_success(self, mock_get_session, mock_get_user):
        """Test recent events endpoint."""
        mock_user = User(id="test-user", email="test@example.com")
        mock_get_user.return_value = mock_user

        # Mock database session and repository
        mock_session = AsyncMock()
        mock_get_session.return_value.__aiter__ = AsyncMock(return_value=iter([mock_session]))

        # Mock repository and events
        with patch('app.api.simulation.admin.SimulationRepository') as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo

            # Mock events
            mock_events = [
                MagicMock(
                    event_id="event-1",
                    event_type="work",
                    summary="Test event 1",
                    timestamp=datetime.utcnow(),
                    status="processed",
                    intensity=5,
                    impact_mood="positive",
                    impact_energy="neutral",
                    impact_stress="neutral"
                )
            ]
            mock_repo.get_events_by_timeframe.return_value = mock_events

            response = self.client.get("/api/simulation/admin/events/recent?limit=10")

            assert response.status_code == 200
            data = response.json()
            assert "events" in data
            assert "total_returned" in data
            assert "filter" in data

    @patch('app.api.simulation.admin.get_current_user')
    def test_get_recent_events_invalid_type(self, mock_get_user):
        """Test recent events with invalid event type."""
        mock_user = User(id="test-user", email="test@example.com")
        mock_get_user.return_value = mock_user

        response = self.client.get("/api/simulation/admin/events/recent?event_type=invalid")

        assert response.status_code == 400
        assert "Invalid event type" in response.json()["detail"]

    @patch('app.api.simulation.admin.get_current_user')
    @patch('app.api.simulation.admin.StateManagerService')
    def test_get_current_ava_state_success(self, mock_state_manager_class, mock_get_user):
        """Test current Ava state endpoint."""
        mock_user = User(id="test-user", email="test@example.com")
        mock_get_user.return_value = mock_user

        # Mock state manager
        mock_state_manager = AsyncMock()
        mock_state_manager_class.return_value = mock_state_manager
        mock_state_manager.get_current_global_state.return_value = {
            "stress": {"value": "50", "numeric_value": 50, "trend": "stable"},
            "energy": {"value": "70", "numeric_value": 70, "trend": "stable"}
        }

        response = self.client.get("/api/simulation/admin/state/current")

        assert response.status_code == 200
        data = response.json()
        assert "ava_global_state" in data
        assert "timestamp" in data

    @patch('app.api.simulation.admin.get_current_user')
    @patch('app.api.simulation.admin.StateManagerService')
    def test_initialize_default_states_success(self, mock_state_manager_class, mock_get_user):
        """Test state initialization endpoint."""
        mock_user = User(id="test-user", email="test@example.com")
        mock_get_user.return_value = mock_user

        # Mock state manager
        mock_state_manager = AsyncMock()
        mock_state_manager_class.return_value = mock_state_manager
        mock_state_manager.initialize_default_states.return_value = {
            "success": True,
            "initialized_traits": ["stress", "energy"],
            "message": "Initialized 2 default traits"
        }

        response = self.client.post("/api/simulation/admin/state/initialize")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "initialized_traits" in data

    @patch('app.api.simulation.admin.get_current_user')
    @patch('app.api.simulation.admin.get_async_session')
    def test_get_statistics_success(self, mock_get_session, mock_get_user):
        """Test statistics endpoint."""
        mock_user = User(id="test-user", email="test@example.com")
        mock_get_user.return_value = mock_user

        # Mock database session
        mock_session = AsyncMock()
        mock_get_session.return_value.__aiter__ = AsyncMock(return_value=iter([mock_session]))

        # Mock repository
        with patch('app.api.simulation.admin.SimulationRepository') as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_event_statistics.return_value = {
                "total_events": 50,
                "events_by_type": {"work": 20, "social": 15, "personal": 15},
                "unprocessed_events": 5,
                "avg_events_per_day": 7.1
            }

            response = self.client.get("/api/simulation/admin/statistics?days=7")

            assert response.status_code == 200
            data = response.json()
            assert "statistics" in data
            assert "timestamp" in data

    @patch('app.api.simulation.admin.get_current_user')
    def test_get_statistics_invalid_days(self, mock_get_user):
        """Test statistics endpoint with invalid days parameter."""
        mock_user = User(id="test-user", email="test@example.com")
        mock_get_user.return_value = mock_user

        # Test days too low
        response = self.client.get("/api/simulation/admin/statistics?days=0")
        assert response.status_code == 400

        # Test days too high
        response = self.client.get("/api/simulation/admin/statistics?days=50")
        assert response.status_code == 400


class TestSimulationAPIErrorHandling:
    """Test error handling in simulation API endpoints."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch('app.api.simulation.admin.get_current_user')
    @patch('app.api.simulation.admin.celery_app')
    def test_status_endpoint_celery_error(self, mock_celery, mock_get_user):
        """Test status endpoint when Celery is unavailable."""
        mock_user = User(id="test-user", email="test@example.com")
        mock_get_user.return_value = mock_user

        # Mock Celery error
        mock_celery.control.inspect.side_effect = Exception("Celery connection failed")

        response = self.client.get("/api/simulation/admin/status")

        assert response.status_code == 500
        assert "Failed to get simulation status" in response.json()["detail"]

    @patch('app.api.simulation.admin.get_current_user')
    @patch('app.api.simulation.admin.get_async_session')
    def test_recent_events_database_error(self, mock_get_session, mock_get_user):
        """Test recent events endpoint with database error."""
        mock_user = User(id="test-user", email="test@example.com")
        mock_get_user.return_value = mock_user

        # Mock database error
        mock_session = AsyncMock()
        mock_session.side_effect = Exception("Database connection failed")
        mock_get_session.return_value.__aiter__ = AsyncMock(return_value=iter([mock_session]))

        response = self.client.get("/api/simulation/admin/events/recent")

        assert response.status_code == 500

    @patch('app.api.simulation.admin.get_current_user')
    @patch('app.api.simulation.admin.StateManagerService')
    def test_state_manager_error_handling(self, mock_state_manager_class, mock_get_user):
        """Test state manager error handling."""
        mock_user = User(id="test-user", email="test@example.com")
        mock_get_user.return_value = mock_user

        # Mock state manager error
        mock_state_manager = AsyncMock()
        mock_state_manager_class.return_value = mock_state_manager
        mock_state_manager.get_current_global_state.side_effect = Exception("State manager error")

        response = self.client.get("/api/simulation/admin/state/current")

        assert response.status_code == 500
        assert "Failed to get current Ava state" in response.json()["detail"]


class TestSimulationAPIIntegration:
    """Integration tests for simulation API endpoints."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_health_check_integration(self):
        """Test health check endpoint integration."""
        # This test doesn't require authentication
        response = self.client.get("/api/simulation/admin/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "component" in data
        assert data["component"] == "simulation-engine"

    @patch('app.api.simulation.admin.get_current_user')
    def test_multiple_endpoint_auth_consistency(self, mock_get_user):
        """Test authentication consistency across endpoints."""
        # Test without auth
        mock_get_user.return_value = None

        endpoints = [
            "/api/simulation/admin/status",
            "/api/simulation/admin/metrics",
            "/api/simulation/admin/events/recent",
            "/api/simulation/admin/state/current",
            "/api/simulation/admin/statistics"
        ]

        for endpoint in endpoints:
            response = self.client.get(endpoint)
            assert response.status_code == 401, f"Endpoint {endpoint} should require auth"

        # Test POST endpoints
        post_endpoints = [
            ("/api/simulation/admin/control", {"action": "start"}),
            ("/api/simulation/admin/state/initialize", {})
        ]

        for endpoint, data in post_endpoints:
            response = self.client.post(endpoint, json=data)
            assert response.status_code == 401, f"Endpoint {endpoint} should require auth"


class TestSimulationAPIMocking:
    """Test proper mocking of simulation components."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    @patch('app.api.simulation.admin.get_current_user')
    @patch('app.api.simulation.admin.celery_app')
    def test_celery_mocking_comprehensive(self, mock_celery, mock_get_user):
        """Test comprehensive Celery mocking."""
        mock_user = User(id="test-user", email="test@example.com")
        mock_get_user.return_value = mock_user

        # Mock all Celery inspect methods
        mock_inspect = MagicMock()
        mock_inspect.active.return_value = {"worker1": ["task1"], "worker2": []}
        mock_inspect.reserved.return_value = {"worker1": ["task2"], "worker2": ["task3"]}
        mock_inspect.stats.return_value = {"worker1": {"total": 100}, "worker2": {"total": 150}}
        mock_inspect.scheduled.return_value = {"worker1": [], "worker2": []}
        mock_inspect.ping.return_value = {"worker1": "pong", "worker2": "pong"}

        mock_celery.control.inspect.return_value = mock_inspect

        # Test status endpoint
        response = self.client.get("/api/simulation/admin/status")
        assert response.status_code == 200

        # Test metrics endpoint
        response = self.client.get("/api/simulation/admin/metrics")
        assert response.status_code == 200

        # Test health endpoint
        response = self.client.get("/api/simulation/admin/health")
        assert response.status_code == 200


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
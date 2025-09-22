import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import json
from datetime import datetime

from app.main import app as fastapi_app
from app.core.database import Base, get_db
from app.models.ava_state import AvaState
# Import all models to ensure they're registered with Base
import app.models


class TestAvaAPI:
    """Integration tests for Ava API endpoints"""
    
    @pytest.fixture
    def db_session(self):
        """Create test database session"""
        engine = create_engine(
            "sqlite:///:memory:", 
            echo=False,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()
    
    @pytest.fixture
    def client(self, db_session):
        """Create test client with database dependency override"""
        def override_get_db():
            try:
                yield db_session
            finally:
                pass
        
        fastapi_app.dependency_overrides[get_db] = override_get_db
        client = TestClient(fastapi_app)
        yield client
        fastapi_app.dependency_overrides.clear()
    
    def test_conversation_endpoint_success(self, client):
        """Test successful conversation with Ava"""
        request_data = {
            "message": "Hello Ava, how are you today?",
            "user_id": "test-user-123"
        }
        
        response = client.post("/api/ava/conversation", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "message" in data
        assert "emotional_state" in data
        assert "timestamp" in data
        assert "conversation_id" in data
        
        # Verify message is not empty
        assert len(data["message"]) > 0
        
        # Verify emotional state structure
        emotional_state = data["emotional_state"]
        assert "mood" in emotional_state
        assert "energy" in emotional_state
        assert "stress" in emotional_state
        
        # Verify energy and stress are within valid range
        assert 1 <= emotional_state["energy"] <= 10
        assert 1 <= emotional_state["stress"] <= 10
        
        # Verify timestamp format
        timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        assert isinstance(timestamp, datetime)
    
    def test_conversation_endpoint_minimal_request(self, client):
        """Test conversation endpoint with minimal request data"""
        request_data = {"message": "Hi!"}
        
        response = client.post("/api/ava/conversation", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["message"]) > 0
    
    def test_conversation_endpoint_invalid_request(self, client):
        """Test conversation endpoint with invalid request data"""
        # Empty message
        response = client.post("/api/ava/conversation", json={"message": ""})
        assert response.status_code == 422
        
        # Missing message
        response = client.post("/api/ava/conversation", json={"user_id": "test"})
        assert response.status_code == 422
        
        # Message too long
        long_message = "a" * 2001
        response = client.post("/api/ava/conversation", json={"message": long_message})
        assert response.status_code == 422
    
    def test_conversation_response_variety(self, client):
        """Test that conversation endpoint returns varied responses"""
        messages = ["Hello", "How are you?", "Tell me something interesting", "What do you think?"]
        responses = []
        
        for message in messages:
            response = client.post("/api/ava/conversation", json={"message": message})
            assert response.status_code == 200
            responses.append(response.json()["message"])
        
        # Responses should vary (at least not all identical)
        unique_responses = set(responses)
        assert len(unique_responses) > 1
    
    def test_get_ava_states_empty(self, client):
        """Test getting Ava states when none exist"""
        response = client.get("/api/ava/state")
        assert response.status_code == 200
        assert response.json() == []
    
    def test_create_ava_state(self, client):
        """Test creating a new Ava state"""
        state_data = {"trait_name": "mood", "value": "cheerful"}
        
        response = client.post("/api/ava/state", json=state_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["trait_name"] == "mood"
        assert data["value"] == "cheerful"
        assert "state_id" in data
        assert "last_updated" in data
    
    def test_create_duplicate_ava_state(self, client):
        """Test creating duplicate state trait fails"""
        state_data = {"trait_name": "energy", "value": "7"}
        
        # First creation should succeed
        response = client.post("/api/ava/state", json=state_data)
        assert response.status_code == 200
        
        # Duplicate should fail
        response = client.post("/api/ava/state", json=state_data)
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]
    
    def test_get_ava_states_with_data(self, client, db_session):
        """Test getting Ava states when data exists"""
        # Add some test data
        states = [
            AvaState(trait_name="mood", value="happy"),
            AvaState(trait_name="energy", value="8"),
            AvaState(trait_name="stress", value="3")
        ]
        
        for state in states:
            db_session.add(state)
        db_session.commit()
        
        response = client.get("/api/ava/state")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 3
        
        trait_names = {item["trait_name"] for item in data}
        assert trait_names == {"mood", "energy", "stress"}
    
    def test_get_specific_ava_state(self, client, db_session):
        """Test getting a specific Ava state by trait name"""
        # Add test data
        state = AvaState(trait_name="confidence", value="9")
        db_session.add(state)
        db_session.commit()
        
        response = client.get("/api/ava/state/confidence")
        assert response.status_code == 200
        
        data = response.json()
        assert data["trait_name"] == "confidence"
        assert data["value"] == "9"
    
    def test_get_nonexistent_ava_state(self, client):
        """Test getting a non-existent Ava state"""
        response = client.get("/api/ava/state/nonexistent")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_update_ava_state(self, client, db_session):
        """Test updating an existing Ava state"""
        # Add test data
        state = AvaState(trait_name="focus", value="5")
        db_session.add(state)
        db_session.commit()
        
        # Update the state
        update_data = {"value": "8"}
        response = client.put("/api/ava/state/focus", json=update_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["trait_name"] == "focus"
        assert data["value"] == "8"
    
    def test_update_nonexistent_ava_state(self, client):
        """Test updating a non-existent Ava state"""
        update_data = {"value": "10"}
        response = client.put("/api/ava/state/nonexistent", json=update_data)
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
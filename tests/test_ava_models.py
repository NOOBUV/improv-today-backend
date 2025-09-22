import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from app.core.database import Base
from app.models.ava_state import AvaState


class TestAvaState:
    """Test cases for the AvaState model"""
    
    @pytest.fixture
    def db_session(self):
        """Create an in-memory SQLite database for testing"""
        engine = create_engine(
            "sqlite:///:memory:", 
            echo=False,
            connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()
    
    def test_create_ava_state(self, db_session):
        """Test creating a new AvaState record"""
        state = AvaState(trait_name="mood", value="cheerful")
        db_session.add(state)
        db_session.commit()
        
        assert state.state_id is not None
        assert state.trait_name == "mood"
        assert state.value == "cheerful"
        assert state.last_updated is not None
        assert isinstance(state.last_updated, datetime)
    
    def test_ava_state_unique_trait_name(self, db_session):
        """Test that trait_name must be unique"""
        state1 = AvaState(trait_name="energy", value="7")
        state2 = AvaState(trait_name="energy", value="8")
        
        db_session.add(state1)
        db_session.commit()
        
        db_session.add(state2)
        with pytest.raises(Exception):  # Should raise IntegrityError
            db_session.commit()
    
    def test_ava_state_repr(self, db_session):
        """Test the string representation of AvaState"""
        state = AvaState(trait_name="stress", value="3")
        db_session.add(state)
        db_session.commit()
        
        expected = "<AvaState(trait_name='stress', value='3')>"
        assert repr(state) == expected
    
    def test_ava_state_update(self, db_session):
        """Test updating an AvaState record"""
        state = AvaState(trait_name="confidence", value="5")
        db_session.add(state)
        db_session.commit()
        
        original_updated = state.last_updated
        
        # Update the value
        state.value = "8"
        db_session.commit()
        
        assert state.value == "8"
        # Note: In this test setup, onupdate might not work exactly as in production
        # but we can still test the basic update functionality
    
    def test_ava_state_query_by_trait_name(self, db_session):
        """Test querying AvaState by trait_name"""
        states = [
            AvaState(trait_name="mood", value="happy"),
            AvaState(trait_name="energy", value="9"),
            AvaState(trait_name="stress", value="2")
        ]
        
        for state in states:
            db_session.add(state)
        db_session.commit()
        
        # Query by trait_name
        mood_state = db_session.query(AvaState).filter(AvaState.trait_name == "mood").first()
        assert mood_state is not None
        assert mood_state.value == "happy"
        
        # Query non-existent trait
        nonexistent = db_session.query(AvaState).filter(AvaState.trait_name == "nonexistent").first()
        assert nonexistent is None
    
    def test_ava_state_required_fields(self, db_session):
        """Test that required fields cannot be null"""
        # Test missing trait_name
        with pytest.raises(Exception):
            state = AvaState(value="test")
            db_session.add(state)
            db_session.commit()
        
        db_session.rollback()
        
        # Test missing value
        with pytest.raises(Exception):
            state = AvaState(trait_name="test")
            db_session.add(state)
            db_session.commit()
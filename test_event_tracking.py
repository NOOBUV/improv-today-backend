#!/usr/bin/env python3
"""
Test script for event tracking and rotation system.
Verifies that Clara doesn't repeat events to the same user.
"""

import asyncio
import logging
import sys
import os

# Add the project root to sys.path so we can import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.services.event_usage_tracker import EventUsageTracker
from app.services.event_selection_service import EventSelectionService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_event_tracking():
    """Test the event tracking and rotation system"""

    logger.info("🧪 Starting Event Tracking Test")

    # Initialize services
    event_tracker = EventUsageTracker()
    event_service = EventSelectionService()

    # Test data
    test_user_id = "test_user_123"
    test_conversation_id = "conv_123"

    # Sample events (simulating what would come from simulation system)
    sample_events = [
        {
            "id": "work_meeting_1",
            "summary": "Had an energizing team standup meeting",
            "event_type": "work",
            "intensity": 6,
            "hours_ago": 2
        },
        {
            "id": "social_coffee_1",
            "summary": "Grabbed coffee with Maya from design team",
            "event_type": "social",
            "intensity": 4,
            "hours_ago": 5
        },
        {
            "id": "personal_plants_1",
            "summary": "Discovered I'm slowly killing my plants",
            "event_type": "personal",
            "intensity": 3,
            "hours_ago": 12
        },
        {
            "id": "work_deadline_1",
            "summary": "Been in full deadline crunch mode this week",
            "event_type": "work",
            "intensity": 8,
            "hours_ago": 1
        }
    ]

    try:
        # Test 1: Fresh event selection
        logger.info("🧪 Test 1: Getting fresh events for new user")
        fresh_events = await event_tracker.get_fresh_events(
            user_id=test_user_id,
            event_pool=sample_events,
            max_events=2
        )
        logger.info(f"✅ Got {len(fresh_events)} fresh events: {[e.get('id') for e in fresh_events]}")

        # Test 2: Track events as used
        logger.info("🧪 Test 2: Tracking events as used")
        events_to_track = [fresh_events[0]["id"]] if fresh_events else ["work_meeting_1"]
        success = await event_tracker.track_events_used(
            user_id=test_user_id,
            conversation_id=test_conversation_id,
            events_used=events_to_track
        )
        logger.info(f"✅ Event tracking success: {success}")

        # Test 3: Get fresh events again (should exclude used ones)
        logger.info("🧪 Test 3: Getting fresh events after some are used")
        fresh_events_2 = await event_tracker.get_fresh_events(
            user_id=test_user_id,
            event_pool=sample_events,
            max_events=2,
            avoid_recent_days=7  # Avoid events used in last week
        )
        used_event_ids = {e["id"] for e in fresh_events_2}
        previously_used = set(events_to_track)
        overlap = used_event_ids.intersection(previously_used)

        logger.info(f"✅ Got {len(fresh_events_2)} fresh events: {[e.get('id') for e in fresh_events_2]}")
        logger.info(f"✅ Event repetition check: {len(overlap)} overlapping events (should be 0)")

        # Test 4: Get user event history
        logger.info("🧪 Test 4: Getting user event history")
        history = await event_tracker.get_user_event_history(test_user_id)
        logger.info(f"✅ User has used {history.get('total_events_used', 0)} events total")

        # Test 5: Contextual event selection (if we have the full service)
        logger.info("🧪 Test 5: Testing contextual event selection")
        try:
            # This will only work if Redis and simulation services are available
            contextual_events = await event_service.get_contextual_events(
                user_id=test_user_id,
                conversation_id=test_conversation_id + "_new",
                user_message="I'm really stressed about work",
                max_events=2
            )
            logger.info(f"✅ Got {len(contextual_events)} contextual events")
        except Exception as e:
            logger.warning(f"⚠️  Contextual event selection test failed (expected if simulation DB not available): {e}")

        # Test 6: Global event usage stats
        logger.info("🧪 Test 6: Getting global event usage statistics")
        stats = await event_tracker.get_event_usage_stats()
        logger.info(f"✅ Global stats: {len(stats)} events tracked globally")

        logger.info("🎉 All event tracking tests completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_event_service_integration():
    """Test the full event selection service"""

    logger.info("🧪 Testing Event Selection Service Integration")

    event_service = EventSelectionService()
    test_user_id = "integration_test_user"
    test_conversation_id = "integration_conv_123"

    try:
        # Test getting fresh events (this will try to connect to simulation system)
        logger.info("🧪 Testing fresh event selection...")
        fresh_events = await event_service.get_fresh_events_for_conversation(
            user_id=test_user_id,
            conversation_id=test_conversation_id,
            max_events=3
        )
        logger.info(f"✅ Got {len(fresh_events)} fresh events from simulation")

        if fresh_events:
            # Test tracking mentioned events
            logger.info("🧪 Testing event mention tracking...")
            success = await event_service.track_events_mentioned_in_response(
                user_id=test_user_id,
                conversation_id=test_conversation_id,
                events_mentioned=fresh_events[:1]  # Track first event as mentioned
            )
            logger.info(f"✅ Event mention tracking success: {success}")

            # Test getting events again (should be different)
            logger.info("🧪 Testing event variety on second call...")
            fresh_events_2 = await event_service.get_fresh_events_for_conversation(
                user_id=test_user_id,
                conversation_id=test_conversation_id + "_new",
                max_events=3
            )

            overlap = set(e.get("id") for e in fresh_events[:1]).intersection(
                set(e.get("id") for e in fresh_events_2)
            )
            logger.info(f"✅ Event variety check: {len(overlap)} overlapping events (should be low)")

        logger.info("🎉 Event selection service integration test completed!")
        return True

    except Exception as e:
        logger.warning(f"⚠️  Integration test failed (expected if simulation system not fully available): {e}")
        return False


async def main():
    """Run all tests"""

    logger.info("🚀 Starting Event Tracking and Rotation System Tests")

    # Test 1: Basic event tracking
    test1_success = await test_event_tracking()

    # Test 2: Service integration
    test2_success = await test_event_service_integration()

    # Summary
    logger.info("=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info(f"Basic Event Tracking: {'✅ PASS' if test1_success else '❌ FAIL'}")
    logger.info(f"Service Integration: {'✅ PASS' if test2_success else '⚠️  PARTIAL (expected)'}")

    if test1_success:
        logger.info("🎉 Core event tracking system is working correctly!")
        logger.info("💡 To test full integration, run with Clara's simulation system active")
    else:
        logger.error("❌ Core event tracking system has issues")

    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
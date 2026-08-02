"""
Performance monitoring for the conversation pipeline.
Tracks timing metrics, generates correlation IDs, and provides detailed logging.
"""
import logging
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict


class ConversationPerformanceMonitor:
    """
    Comprehensive performance monitoring for conversation pipeline.
    Tracks timing metrics, generates correlation IDs, and provides detailed logging.
    """

    def __init__(self):
        self.performance_logger = logging.getLogger(f"{__name__}.performance")
        self.alert_thresholds = {
            "total_response_time_ms": 3000,  # <3s total response time requirement
            "consciousness_generation_ms": 2000,  # <2s consciousness generation
            "context_gathering_ms": 1000,  # <1s context processing
            "response_formatting_ms": 500,  # <500ms response formatting

            # Granular conversation response component thresholds
            "context_extraction_ms": 50,  # <50ms to extract simulation context components
            "emotion_selection_ms": 100,  # <100ms for emotion selection with mood awareness
            "prompt_construction_ms": 200,  # <200ms for enhanced prompt building
            "openai_api_call_ms": 1500,  # <1.5s for OpenAI API response
            "response_parsing_ms": 100,  # <100ms for JSON parsing and formatting

            # Context gathering sub-components
            "global_state_retrieval_ms": 100,  # <100ms for database state retrieval
            "event_selection_ms": 200,  # <200ms for event selection service
            "backstory_selection_ms": 150,  # <150ms for backstory content selection
            "sentiment_analysis_ms": 50,  # <50ms for message sentiment analysis
            "state_influence_calculation_ms": 100,  # <100ms for state influence calculations
        }

    def create_conversation_correlation_id(self, user_id: str, conversation_id: str) -> str:
        """Generate unique correlation ID for conversation tracking."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"conv_{user_id[:8]}_{conversation_id[:8]}_{timestamp}_{unique_id}"

    def start_timing_context(self, correlation_id: str, operation: str) -> Dict[str, Any]:
        """Start timing context for a specific operation."""
        start_time = time.time()
        context = {
            "correlation_id": correlation_id,
            "operation": operation,
            "start_time": start_time,
            "start_timestamp": datetime.now(timezone.utc).isoformat(),
            "sub_operations": {}
        }

        self.performance_logger.info(
            f"[{correlation_id}] Starting {operation}",
            extra={
                "correlation_id": correlation_id,
                "operation": operation,
                "event_type": "operation_start",
                "timestamp": context["start_timestamp"]
            }
        )

        return context

    def end_timing_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """End timing context and calculate final metrics."""
        end_time = time.time()
        total_duration_ms = (end_time - context["start_time"]) * 1000

        metrics = {
            "correlation_id": context["correlation_id"],
            "operation": context["operation"],
            "total_duration_ms": round(total_duration_ms, 2),
            "start_timestamp": context["start_timestamp"],
            "end_timestamp": datetime.now(timezone.utc).isoformat(),
            "sub_operations": context["sub_operations"]
        }

        # Check thresholds and alert if exceeded
        self._check_performance_thresholds(metrics)

        self.performance_logger.info(
            f"[{context['correlation_id']}] Completed {context['operation']} in {total_duration_ms:.2f}ms",
            extra={
                **metrics,
                "event_type": "operation_complete"
            }
        )

        return metrics

    @contextmanager
    def step(self, ctx, name):
        """Time a sub-operation; records on exit, re-raises on error.

        Yields a plain dict — callers stash metadata via s.update(...).
        """
        s: Dict[str, Any] = {}
        start = time.time()
        sub_started = datetime.now(timezone.utc).isoformat()
        try:
            yield s
        except Exception as e:
            self._record(ctx, name, start, sub_started, error=str(e), **s)
            raise  # NEVER swallow — caller's try/except owns control flow
        self._record(ctx, name, start, sub_started, **s)

    def _record(self, context: Dict[str, Any], sub_operation: str, start_time: float, start_timestamp: str, **metadata) -> None:
        """Record timing and metadata for a completed sub-operation."""
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        context["sub_operations"][sub_operation] = {
            "duration_ms": round(duration_ms, 2),
            "start_timestamp": start_timestamp,
            "end_timestamp": datetime.now(timezone.utc).isoformat(),
            **metadata
        }

        self.performance_logger.debug(
            f"[{context['correlation_id']}] Completed sub-operation {sub_operation} in {duration_ms:.2f}ms",
            extra={
                "correlation_id": context["correlation_id"],
                "parent_operation": context["operation"],
                "sub_operation": sub_operation,
                "duration_ms": duration_ms,
                "event_type": "sub_operation_complete",
                **metadata
            }
        )

        # Check sub-operation thresholds
        threshold_key = f"{sub_operation}_ms"
        if threshold_key in self.alert_thresholds and duration_ms > self.alert_thresholds[threshold_key]:
            self.performance_logger.warning(
                f"[{context['correlation_id']}] {sub_operation} exceeded threshold: {duration_ms:.2f}ms > {self.alert_thresholds[threshold_key]}ms",
                extra={
                    "correlation_id": context["correlation_id"],
                    "sub_operation": sub_operation,
                    "duration_ms": duration_ms,
                    "threshold_ms": self.alert_thresholds[threshold_key],
                    "event_type": "threshold_exceeded"
                }
            )

    def _check_performance_thresholds(self, metrics: Dict[str, Any]) -> None:
        """Check if total response time exceeds the defined threshold."""
        total_duration = metrics["total_duration_ms"]

        if total_duration > self.alert_thresholds["total_response_time_ms"]:
            self.performance_logger.warning(
                f"[{metrics['correlation_id']}] Total conversation response time exceeded threshold: {total_duration:.2f}ms > {self.alert_thresholds['total_response_time_ms']}ms",
                extra={
                    "correlation_id": metrics["correlation_id"],
                    "total_duration_ms": total_duration,
                    "threshold_ms": self.alert_thresholds["total_response_time_ms"],
                    "event_type": "total_threshold_exceeded",
                    "sub_operations_breakdown": metrics["sub_operations"]
                }
            )

    def log_error_with_context(self, context: Dict[str, Any], error: Exception, operation: str = None) -> None:
        """Log error with full performance context."""
        operation_name = operation or context.get("operation", "unknown")
        correlation_id = context.get("correlation_id", "unknown")

        current_time = time.time()
        duration_so_far = (current_time - context.get("start_time", current_time)) * 1000

        self.performance_logger.error(
            f"[{correlation_id}] Error in {operation_name} after {duration_so_far:.2f}ms: {str(error)}",
            extra={
                "correlation_id": correlation_id,
                "operation": operation_name,
                "error_message": str(error),
                "error_type": type(error).__name__,
                "duration_until_error_ms": duration_so_far,
                "traceback": traceback.format_exc(),
                "event_type": "operation_error",
                "sub_operations_completed": context.get("sub_operations", {})
            }
        )

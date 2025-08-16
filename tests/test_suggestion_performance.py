import pytest
import time
from app.services.suggestion_service import SuggestionService


class TestSuggestionPerformance:
    """Performance tests for the SuggestionService"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.service = SuggestionService()
    
    def test_suggestion_generation_speed(self):
        """Test that suggestion generation is fast enough for synchronous use"""
        test_messages = [
            "I'm really happy about this good news!",
            "The meeting was bad but we learned something",
            "Let's walk to the nice place and talk about work",
            "This thing is too small for such a big project",
            "I need to think and study more to improve my skills",
            "No matching keywords here: quantum mechanics philosophy",
        ]
        
        total_time = 0
        iterations = 100
        
        for _ in range(iterations):
            for message in test_messages:
                start_time = time.perf_counter()
                self.service.generate_suggestion(message)
                end_time = time.perf_counter()
                total_time += (end_time - start_time)
        
        avg_time_per_call = total_time / (iterations * len(test_messages))
        
        # Should be much faster than 1ms per call for synchronous use
        assert avg_time_per_call < 0.001, f"Average time per call: {avg_time_per_call:.6f}s is too slow"
        print(f"Average time per suggestion generation: {avg_time_per_call:.6f}s")
    
    def test_suggestion_worst_case_performance(self):
        """Test performance with very long text"""
        # Create a long message with the trigger word at the end
        long_message = " ".join(["quantum"] * 1000) + " I am happy"
        
        start_time = time.perf_counter()
        result = self.service.generate_suggestion(long_message)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        # Even with long text, should be fast
        assert execution_time < 0.01, f"Execution time {execution_time:.6f}s too slow for long text"
        assert result is not None
        assert result["word"] == "elated"
        
        print(f"Long text processing time: {execution_time:.6f}s")
    
    def test_memory_usage_stability(self):
        """Test that repeated calls don't cause memory issues"""
        import gc
        import sys
        
        # Get initial memory state
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Run many suggestion generations
        for i in range(1000):
            self.service.generate_suggestion(f"I'm happy number {i}")
        
        # Check memory state after
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not have significant memory growth
        object_growth = final_objects - initial_objects
        assert object_growth < 100, f"Too many objects created: {object_growth}"
        
        print(f"Object count growth after 1000 calls: {object_growth}")
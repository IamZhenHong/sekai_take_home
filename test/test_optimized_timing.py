#!/usr/bin/env python3
"""
Performance comparison test between original and optimized agents.
"""

import sys
import os
import time
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph import run_one_user
from agents.recommendation_agent_optimized import optimized_recommendation_agent_node
from agents.evaluation_agent_optimized import optimized_evaluation_agent
from agents.optimiser_agent_optimized import optimized_optimiser_agent_node

def test_original_performance():
    """Test original system performance"""
    print("🧪 Testing ORIGINAL system performance...")
    print("=" * 50)
    
    test_user_id = "606827"
    
    start_time = time.time()
    try:
        # Run original system
        run_one_user(test_user_id, k=5, seed=42)
        end_time = time.time()
        total_time = round(end_time - start_time, 2)
        
        print(f"\n✅ Original system completed in {total_time}s")
        return total_time
        
    except Exception as e:
        print(f"❌ Original system test failed: {e}")
        return None

def test_optimized_performance():
    """Test optimized system performance"""
    print("\n🧪 Testing OPTIMIZED system performance...")
    print("=" * 50)
    
    test_user_id = "606827"
    
    # Create test state
    test_state = {
        "user_id": test_user_id,
        "user_tags": ["romance", "anime", "school", "drama"],
        "full_interest_tags": ["romance", "anime", "school", "drama", "action"],
        "recommendation_prompt": "User prefers romantic anime with school settings and dramatic elements",
        "recommendation_list": [],
        "ground_truth_list": [],
        "precision": 0.0,
        "prompt_feedback": "",
        "taste_profile": "",
        "iteration": 0,
        "max_iterations": 2,
        "remaining_steps": 0,
        "structured_response": None,
        "recommendation_time": 0.0,
        "evaluation_time": 0.0,
        "optimiser_time": 0.0,
        "total_time": 0.0
    }
    
    start_time = time.time()
    try:
        # Test optimized recommendation agent
        print("Testing optimized recommendation agent...")
        rec_result = optimized_recommendation_agent_node(test_state)
        rec_time = rec_result.get("recommendation_time", 0.0)
        print(f"✅ Optimized recommendation: {rec_time}s")
        
        # Test optimized evaluation agent
        print("Testing optimized evaluation agent...")
        eval_result = optimized_evaluation_agent(rec_result)
        eval_time = eval_result.get("evaluation_time", 0.0)
        print(f"✅ Optimized evaluation: {eval_time}s")
        
        # Test optimized optimiser agent
        print("Testing optimized optimiser agent...")
        opt_result = optimized_optimiser_agent_node(eval_result)
        opt_time = opt_result.get("optimiser_time", 0.0)
        print(f"✅ Optimized optimiser: {opt_time}s")
        
        end_time = time.time()
        total_time = round(end_time - start_time, 2)
        
        print(f"\n✅ Optimized system completed in {total_time}s")
        print(f"  - Recommendation: {rec_time}s")
        print(f"  - Evaluation: {eval_time}s")
        print(f"  - Optimiser: {opt_time}s")
        
        return total_time, rec_time, eval_time, opt_time
        
    except Exception as e:
        print(f"❌ Optimized system test failed: {e}")
        return None, None, None, None

def compare_performance():
    """Compare original vs optimized performance"""
    print("🚀 PERFORMANCE COMPARISON TEST")
    print("=" * 60)
    
    # Test original system
    original_time = test_original_performance()
    
    # Test optimized system
    optimized_results = test_optimized_performance()
    optimized_time = optimized_results[0] if optimized_results[0] else None
    
    # Compare results
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE COMPARISON RESULTS")
    print("=" * 60)
    
    if original_time and optimized_time:
        improvement = round(((original_time - optimized_time) / original_time) * 100, 1)
        
        print(f"Original System Time: {original_time}s")
        print(f"Optimized System Time: {optimized_time}s")
        print(f"Performance Improvement: {improvement}% faster")
        
        if optimized_results[1]:
            print(f"\nDetailed Optimized Times:")
            print(f"  - Recommendation: {optimized_results[1]}s")
            print(f"  - Evaluation: {optimized_results[2]}s")
            print(f"  - Optimiser: {optimized_results[3]}s")
        
        if improvement > 50:
            print(f"\n🎉 EXCELLENT! {improvement}% performance improvement achieved!")
        elif improvement > 30:
            print(f"\n✅ GOOD! {improvement}% performance improvement achieved!")
        elif improvement > 10:
            print(f"\n📈 DECENT! {improvement}% performance improvement achieved!")
        else:
            print(f"\n⚠️ MINIMAL improvement of {improvement}%. May need further optimization.")
            
    else:
        print("❌ Could not complete performance comparison due to errors.")
    
    print("\n" + "=" * 60)

def test_memory_usage():
    """Test memory usage comparison"""
    print("\n🧠 MEMORY USAGE TEST")
    print("=" * 40)
    
    import psutil
    import gc
    
    def get_memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    # Test original system memory
    gc.collect()
    memory_before = get_memory_usage()
    
    try:
        test_original_performance()
        memory_after_original = get_memory_usage()
        original_memory = memory_after_original - memory_before
        print(f"Original system memory usage: {original_memory:.2f} MB")
    except:
        print("❌ Could not test original system memory usage")
        original_memory = None
    
    # Test optimized system memory
    gc.collect()
    memory_before = get_memory_usage()
    
    try:
        test_optimized_performance()
        memory_after_optimized = get_memory_usage()
        optimized_memory = memory_after_optimized - memory_before
        print(f"Optimized system memory usage: {optimized_memory:.2f} MB")
    except:
        print("❌ Could not test optimized system memory usage")
        optimized_memory = None
    
    # Compare memory usage
    if original_memory and optimized_memory:
        memory_improvement = round(((original_memory - optimized_memory) / original_memory) * 100, 1)
        print(f"Memory improvement: {memory_improvement}% less memory used")

if __name__ == "__main__":
    print("🚀 Starting comprehensive performance comparison...")
    
    # Run performance comparison
    compare_performance()
    
    # Run memory usage test (optional)
    try:
        test_memory_usage()
    except ImportError:
        print("\n⚠️ psutil not available, skipping memory usage test")
    
    print("\n✅ Performance comparison completed!") 
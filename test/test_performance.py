#!/usr/bin/env python3
"""
Performance testing script for the optimized recommendation agent.
"""

import time
import sys
import os
from agents.recommendation_agent import recommendation_agent_node

def test_performance():
    """Test the performance of the recommendation agent"""
    
    # Test cases with different user profiles
    test_cases = [
        {
            "user_tags": ["romance", "anime", "school"],
            "recommendation_prompt": "User prefers romantic school anime stories with slice-of-life elements."
        },
        {
            "user_tags": ["action", "superhero", "adventure"],
            "recommendation_prompt": "User enjoys action-packed superhero adventures with dramatic conflicts."
        },
        {
            "user_tags": ["fantasy", "magic", "adventure"],
            "recommendation_prompt": "User loves fantasy worlds with magical elements and epic adventures."
        }
    ]
    
    print("🚀 Performance Testing for Recommendation Agent")
    print("=" * 50)
    
    total_time = 0
    successful_runs = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['user_tags']}")
        print("-" * 30)
        
        # Prepare state
        state = {
            "user_tags": test_case["user_tags"],
            "recommendation_prompt": test_case["recommendation_prompt"],
            "iteration": 0
        }
        
        try:
            # Time the execution
            start_time = time.time()
            result = recommendation_agent_node(state)
            end_time = time.time()
            
            execution_time = round(end_time - start_time, 2)
            total_time += execution_time
            successful_runs += 1
            
            print(f"✅ Execution time: {execution_time}s")
            
            if result.get("recommendation_list"):
                print(f"📊 Recommendations found: {len(result['recommendation_list'])}")
                print(f"⏱️ Agent time: {result.get('recommendation_time', 'N/A')}s")
            else:
                print("⚠️ No recommendations generated")
                
        except Exception as e:
            print(f"❌ Error in test case {i}: {e}")
    
    print("\n" + "=" * 50)
    print(f"📈 Performance Summary:")
    print(f"   • Total successful runs: {successful_runs}/{len(test_cases)}")
    print(f"   • Average execution time: {round(total_time/successful_runs, 2)}s" if successful_runs > 0 else "   • No successful runs")
    print(f"   • Total time: {round(total_time, 2)}s")
    
    # Test cache effectiveness
    print(f"\n🔍 Cache Effectiveness Test:")
    print("-" * 30)
    
    # Test repeated calls with same parameters
    test_state = {
        "user_tags": ["romance", "anime"],
        "recommendation_prompt": "User prefers romantic anime stories.",
        "iteration": 0
    }
    
    times = []
    for i in range(3):
        start_time = time.time()
        result = recommendation_agent_node(test_state)
        end_time = time.time()
        times.append(round(end_time - start_time, 2))
        print(f"   Run {i+1}: {times[-1]}s")
    
    if len(times) > 1:
        improvement = round((times[0] - times[-1]) / times[0] * 100, 1)
        print(f"   📉 Performance improvement: {improvement}% (first vs last run)")

if __name__ == "__main__":
    test_performance() 
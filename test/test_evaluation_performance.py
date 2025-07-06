#!/usr/bin/env python3
"""
Performance testing script for the optimized evaluation agent with parallel branching.
"""

import time
import sys
import os
from agents.evaluation_agent import evaluation_agent

def test_evaluation_performance():
    """Test the performance of the optimized evaluation agent"""
    
    # Test cases with different user profiles
    test_cases = [
        {
            "user_id": "606827",
            "recommendation_list": [123, 456, 789, 101, 202, 303, 404, 505, 606, 707],
            "recommendation_prompt": "User prefers romantic anime stories with slice-of-life elements.",
            "user_tags": ["romance", "anime", "school"],
            "full_interest_tags": ["original character", "genshin impact", "naruto", "romance", "slice of life"]
        },
        {
            "user_id": "1054188", 
            "recommendation_list": [111, 222, 333, 444, 555, 666, 777, 888, 999, 1000],
            "recommendation_prompt": "User enjoys action-packed superhero adventures with dramatic conflicts.",
            "user_tags": ["action", "superhero", "adventure"],
            "full_interest_tags": ["marvel comics", "supernatural", "romance", "confession"]
        },
        {
            "user_id": "1101445",
            "recommendation_list": [201, 301, 401, 501, 601, 701, 801, 901, 1001, 1101],
            "recommendation_prompt": "User loves fantasy worlds with magical elements and epic adventures.",
            "user_tags": ["fantasy", "magic", "adventure"],
            "full_interest_tags": ["k-pop idols", "blue lock", "stray kids", "emotional support"]
        }
    ]
    
    print("🚀 Performance Testing for Optimized Evaluation Agent")
    print("=" * 60)
    
    total_time = 0
    successful_runs = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: User {test_case['user_id']}")
        print("-" * 40)
        
        # Prepare state
        state = {
            "user_id": test_case["user_id"],
            "recommendation_list": test_case["recommendation_list"],
            "recommendation_prompt": test_case["recommendation_prompt"],
            "user_tags": test_case["user_tags"],
            "full_interest_tags": test_case["full_interest_tags"],
            "taste_profile": "User enjoys diverse content with focus on character development."
        }
        
        try:
            # Time the execution
            start_time = time.time()
            result = evaluation_agent(state)
            end_time = time.time()
            
            execution_time = round(end_time - start_time, 2)
            total_time += execution_time
            successful_runs += 1
            
            print(f"✅ Execution time: {execution_time}s")
            print(f"⏱️ Evaluation time: {result.get('evaluation_time', 'N/A')}s")
            
            # Display key metrics
            if result.get("precision") is not None:
                print(f"📊 Precision: {result['precision']:.3f}")
            if result.get("tag_overlap_ratio") is not None:
                print(f"🏷️ Tag overlap ratio: {result['tag_overlap_ratio']:.3f}")
            if result.get("semantic_overlap_ratio") is not None:
                print(f"🧠 Semantic overlap ratio: {result['semantic_overlap_ratio']:.3f}")
                
        except Exception as e:
            print(f"❌ Error in test case {i}: {e}")
    
    print("\n" + "=" * 60)
    print(f"📈 Performance Summary:")
    print(f"   • Total successful runs: {successful_runs}/{len(test_cases)}")
    print(f"   • Average execution time: {round(total_time/successful_runs, 2)}s" if successful_runs > 0 else "   • No successful runs")
    print(f"   • Total time: {round(total_time, 2)}s")
    
    # Test parallel execution effectiveness
    print(f"\n🔍 Parallel Execution Test:")
    print("-" * 40)
    
    # Test repeated calls with same parameters to see caching benefits
    test_state = {
        "user_id": "606827",
        "recommendation_list": [123, 456, 789, 101, 202, 303, 404, 505, 606, 707],
        "recommendation_prompt": "User prefers romantic anime stories.",
        "user_tags": ["romance", "anime"],
        "full_interest_tags": ["romance", "anime", "school"],
        "taste_profile": "User enjoys romantic content."
    }
    
    times = []
    for i in range(3):
        start_time = time.time()
        result = evaluation_agent(test_state)
        end_time = time.time()
        times.append(round(end_time - start_time, 2))
        print(f"   Run {i+1}: {times[-1]}s")
    
    if len(times) > 1:
        improvement = round((times[0] - times[-1]) / times[0] * 100, 1)
        print(f"   📉 Performance improvement: {improvement}% (first vs last run)")
    
    # Explain the optimizations
    print(f"\n💡 Optimization Details:")
    print("-" * 40)
    print("   • Parallel branching: precision, tag_overlap, and semantic_overlap")
    print("   • ChromaDB collection caching with @lru_cache")
    print("   • Fan-out/fan-in pattern for independent calculations")
    print("   • Reduced sequential dependencies")

if __name__ == "__main__":
    test_evaluation_performance() 
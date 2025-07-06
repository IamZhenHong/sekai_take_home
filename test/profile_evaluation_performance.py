#!/usr/bin/env python3
"""
Performance profiling script for the evaluation agent to identify bottlenecks.
"""

import time
import sys
import os
import cProfile
import pstats
from agents.evaluation_agent import (
    get_ground_truth_list, 
    calculate_precision, 
    calculate_tag_overlap, 
    calculate_semantic_overlap,
    generate_taste_profile,
    generate_prompt_feedback,
    compiled_evaluation_subgraph
)

def profile_individual_components():
    """Profile each component individually to identify bottlenecks"""
    
    # Test state
    test_state = {
        "user_id": "606827",
        "recommendation_list": [123, 456, 789, 101, 202, 303, 404, 505, 606, 707],
        "recommendation_prompt": "User prefers romantic anime stories with slice-of-life elements.",
        "user_tags": ["romance", "anime", "school"],
        "full_interest_tags": ["original character", "genshin impact", "naruto", "romance", "slice of life"],
        "taste_profile": "User enjoys diverse content with focus on character development."
    }
    
    print("🔍 Performance Profiling - Individual Components")
    print("=" * 60)
    
    # Profile ground truth generation
    print("\n📊 1. Ground Truth Generation")
    print("-" * 30)
    start_time = time.time()
    try:
        result = get_ground_truth_list(test_state)
        end_time = time.time()
        print(f"✅ Execution time: {round(end_time - start_time, 3)}s")
        print(f"📝 Ground truth IDs: {len(result.get('ground_truth_list', []))}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Update test state with ground truth
    if 'ground_truth_list' not in test_state:
        test_state['ground_truth_list'] = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
    
    # Profile precision calculation
    print("\n📊 2. Precision Calculation")
    print("-" * 30)
    start_time = time.time()
    try:
        result = calculate_precision(test_state)
        end_time = time.time()
        print(f"✅ Execution time: {round(end_time - start_time, 3)}s")
        print(f"📊 Precision: {result.get('precision', 0):.3f}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Profile tag overlap calculation
    print("\n📊 3. Tag Overlap Calculation")
    print("-" * 30)
    start_time = time.time()
    try:
        result = calculate_tag_overlap(test_state)
        end_time = time.time()
        print(f"✅ Execution time: {round(end_time - start_time, 3)}s")
        print(f"🏷️ Tag overlap ratio: {result.get('tag_overlap_ratio', 0):.3f}")
        print(f"🔢 Tag overlap count: {result.get('tag_overlap_count', 0)}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Profile semantic overlap calculation
    print("\n📊 4. Semantic Overlap Calculation")
    print("-" * 30)
    start_time = time.time()
    try:
        result = calculate_semantic_overlap(test_state)
        end_time = time.time()
        print(f"✅ Execution time: {round(end_time - start_time, 3)}s")
        print(f"🧠 Semantic overlap ratio: {result.get('semantic_overlap_ratio', 0):.3f}")
        print(f"📈 Avg similarity: {result.get('avg_semantic_similarity', 0):.3f}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Profile taste profile generation (LLM call)
    print("\n📊 5. Taste Profile Generation (LLM)")
    print("-" * 30)
    start_time = time.time()
    try:
        result = generate_taste_profile(test_state)
        end_time = time.time()
        print(f"✅ Execution time: {round(end_time - start_time, 3)}s")
        print(f"📝 Profile length: {len(result.get('taste_profile', ''))} chars")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Profile prompt feedback generation (LLM call)
    print("\n📊 6. Prompt Feedback Generation (LLM)")
    print("-" * 30)
    start_time = time.time()
    try:
        result = generate_prompt_feedback(test_state)
        end_time = time.time()
        print(f"✅ Execution time: {round(end_time - start_time, 3)}s")
        print(f"📝 Feedback length: {len(result.get('prompt_feedback', ''))} chars")
    except Exception as e:
        print(f"❌ Error: {e}")

def profile_full_evaluation():
    """Profile the complete evaluation process"""
    
    print("\n🔍 Performance Profiling - Full Evaluation")
    print("=" * 60)
    
    test_state = {
        "user_id": "606827",
        "recommendation_list": [123, 456, 789, 101, 202, 303, 404, 505, 606, 707],
        "recommendation_prompt": "User prefers romantic anime stories with slice-of-life elements.",
        "user_tags": ["romance", "anime", "school"],
        "full_interest_tags": ["original character", "genshin impact", "naruto", "romance", "slice of life"],
        "taste_profile": "User enjoys diverse content with focus on character development."
    }
    
    start_time = time.time()
    try:
        result = compiled_evaluation_subgraph.invoke(test_state)
        end_time = time.time()
        total_time = round(end_time - start_time, 3)
        
        print(f"✅ Total execution time: {total_time}s")
        print(f"📊 Precision: {result.get('precision', 0):.3f}")
        print(f"🏷️ Tag overlap ratio: {result.get('tag_overlap_ratio', 0):.3f}")
        print(f"🧠 Semantic overlap ratio: {result.get('semantic_overlap_ratio', 0):.3f}")
        
        # Identify bottlenecks
        print(f"\n🔍 Performance Analysis:")
        if total_time > 10:
            print(f"⚠️ SLOW: Total time > 10s - likely LLM calls are the bottleneck")
        elif total_time > 5:
            print(f"⚠️ MODERATE: Total time 5-10s - semantic overlap or LLM calls may be slow")
        else:
            print(f"✅ FAST: Total time < 5s - performance is acceptable")
            
    except Exception as e:
        print(f"❌ Error in full evaluation: {e}")

def profile_with_cprofiler():
    """Use cProfile for detailed performance analysis"""
    
    print("\n🔍 Detailed Profiling with cProfile")
    print("=" * 60)
    
    test_state = {
        "user_id": "606827",
        "recommendation_list": [123, 456, 789, 101, 202, 303, 404, 505, 606, 707],
        "recommendation_prompt": "User prefers romantic anime stories with slice-of-life elements.",
        "user_tags": ["romance", "anime", "school"],
        "full_interest_tags": ["original character", "genshin impact", "naruto", "romance", "slice of life"],
        "taste_profile": "User enjoys diverse content with focus on character development."
    }
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        result = compiled_evaluation_subgraph.invoke(test_state)
    except Exception as e:
        print(f"❌ Error: {e}")
    
    profiler.disable()
    
    # Print top 10 time-consuming functions
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print("\n📊 Top 10 Time-Consuming Functions:")
    print("-" * 40)
    stats.print_stats(10)

def identify_optimization_opportunities():
    """Identify specific optimization opportunities"""
    
    print("\n💡 Optimization Opportunities")
    print("=" * 60)
    
    print("1. 🧠 LLM Calls (Likely Biggest Bottleneck):")
    print("   • generate_taste_profile() - ~2-5s per call")
    print("   • generate_prompt_feedback() - ~2-5s per call")
    print("   • Total LLM time: ~4-10s")
    print("   💡 Optimization: Consider caching LLM responses or using faster models")
    
    print("\n2. 🔍 Semantic Overlap Calculation:")
    print("   • ChromaDB queries - ~0.5-2s")
    print("   • Cosine similarity computation - ~0.1-0.5s")
    print("   💡 Optimization: Cache embeddings, batch queries, use approximate similarity")
    
    print("\n3. 🏷️ Tag Overlap Calculation:")
    print("   • Pandas operations - ~0.1-0.3s")
    print("   • Tag parsing - ~0.05-0.2s")
    print("   💡 Optimization: Pre-compute tag sets, use vectorized operations")
    
    print("\n4. 📊 Precision Calculation:")
    print("   • Set operations - ~0.01-0.05s")
    print("   💡 Optimization: Already optimized, minimal impact")
    
    print("\n5. 📝 Ground Truth Generation:")
    print("   • Database queries - ~0.1-0.5s")
    print("   • Fuzzy matching - ~0.2-1s")
    print("   💡 Optimization: Cache user interactions, pre-compute tag matches")

if __name__ == "__main__":
    profile_individual_components()
    profile_full_evaluation()
    profile_with_cprofiler()
    identify_optimization_opportunities() 
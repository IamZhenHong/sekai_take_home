#!/usr/bin/env python3
"""
Test script for calculate_tag_overlap function
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.evaluation_agent import calculate_tag_overlap, contents_df

def test_calculate_tag_overlap():
    """Test the calculate_tag_overlap function"""
    
    # Create a test state with some content IDs
    test_state = {
        "recommendation_list": ["213977", "213987", "214002"],  # Some content IDs from the CSV
        "ground_truth_list": ["213977", "214131", "214217"],    # Overlapping and different content IDs
        "user_id": "test_user",
        "precision": 0.33  # 1/3 overlap
    }
    
    print("🧪 Testing calculate_tag_overlap function...")
    print("=" * 50)
    
    # Check if content IDs exist in the dataframe
    print(f"Available content IDs in CSV: {contents_df['content_id'].head().tolist()}")
    print(f"Test recommendation list: {test_state['recommendation_list']}")
    print(f"Test ground truth list: {test_state['ground_truth_list']}")
    
    # Check if our test IDs exist
    existing_rec = contents_df[contents_df['content_id'].astype(str).isin(test_state['recommendation_list'])]
    existing_gt = contents_df[contents_df['content_id'].astype(str).isin(test_state['ground_truth_list'])]
    
    print(f"Found {len(existing_rec)} recommended content in CSV")
    print(f"Found {len(existing_gt)} ground truth content in CSV")
    
    if len(existing_rec) == 0 or len(existing_gt) == 0:
        print("❌ Test content IDs not found in CSV. Using first few content IDs instead.")
        # Use first few content IDs from the CSV
        sample_ids = contents_df['content_id'].head(6).astype(str).tolist()
        test_state["recommendation_list"] = sample_ids[:3]
        test_state["ground_truth_list"] = sample_ids[3:]
        print(f"New recommendation list: {test_state['recommendation_list']}")
        print(f"New ground truth list: {test_state['ground_truth_list']}")
    
    # Run the function
    try:
        result = calculate_tag_overlap(test_state)
        
        print("\n✅ Function executed successfully!")
        print("=" * 50)
        print(f"Tag Overlap Ratio: {result.get('tag_overlap_ratio', 0):.3f}")
        print(f"Tag Overlap Count: {result.get('tag_overlap_count', 0)}")
        print(f"Tag Coverage Ratio: {result.get('tag_coverage_ratio', 0):.3f}")
        print(f"Total Unique Tags: {result.get('total_unique_tags', 0)}")
        
        print(f"\nRecommended Tags ({len(result.get('recommended_tags', []))}):")
        for tag in sorted(result.get('recommended_tags', []))[:10]:  # Show first 10
            print(f"  - {tag}")
        
        print(f"\nGround Truth Tags ({len(result.get('ground_truth_tags', []))}):")
        for tag in sorted(result.get('ground_truth_tags', []))[:10]:  # Show first 10
            print(f"  - {tag}")
        
        print(f"\nOverlap Tags ({len(result.get('overlap_tags', []))}):")
        for tag in sorted(result.get('overlap_tags', [])):
            print(f"  - {tag}")
        
        print("\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_calculate_tag_overlap() 
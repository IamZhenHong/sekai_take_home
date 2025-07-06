#!/usr/bin/env python3
"""
Simple test for calculate_tag_overlap function without LLM dependencies
"""

import pandas as pd
import os

# Load the data directly
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
contents_path = os.path.join(base_dir, "datasets/contents.csv")
contents_df = pd.read_csv(contents_path)
contents_df["content_id"] = contents_df["content_id"].astype(str)

def extract_tags(tags_str):
    """Safely extract tags from string representation"""
    try:
        if isinstance(tags_str, str):
            # Handle string representation of list
            tags = eval(tags_str) if tags_str.startswith('[') else []
        else:
            tags = []
        return [str(tag).lower().strip() for tag in tags if tag]
    except:
        return []

def calculate_tag_overlap_simple(recommendation_list, ground_truth_list):
    """
    Calculate tag overlap between recommended content and ground truth content.
    """
    recommended_ids = set(recommendation_list)
    ground_truth_ids = set(ground_truth_list)
    
    # Get tags for recommended content
    recommended_content = contents_df[
        contents_df["content_id"].astype(str).isin(recommended_ids)
    ]
    
    # Get tags for ground truth content
    ground_truth_content = contents_df[
        contents_df["content_id"].astype(str).isin(ground_truth_ids)
    ]
    
    # Get all tags from recommended content
    recommended_tags = set()
    for _, row in recommended_content.iterrows():
        tags = extract_tags(row.get("generated_tags", ""))
        recommended_tags.update(tags)
    
    # Get all tags from ground truth content
    ground_truth_tags = set()
    for _, row in ground_truth_content.iterrows():
        tags = extract_tags(row.get("generated_tags", ""))
        ground_truth_tags.update(tags)
    
    # Calculate overlap metrics
    if not ground_truth_tags:
        tag_overlap_ratio = 0.0
        tag_overlap_count = 0
    else:
        overlap_tags = recommended_tags & ground_truth_tags
        tag_overlap_count = len(overlap_tags)
        tag_overlap_ratio = tag_overlap_count / len(ground_truth_tags)
    
    # Calculate additional metrics
    total_unique_tags = len(recommended_tags | ground_truth_tags)
    tag_coverage_ratio = len(recommended_tags) / total_unique_tags if total_unique_tags > 0 else 0.0
    
    return {
        "tag_overlap_ratio": tag_overlap_ratio,
        "tag_overlap_count": tag_overlap_count,
        "recommended_tags": list(recommended_tags),
        "ground_truth_tags": list(ground_truth_tags),
        "overlap_tags": list(recommended_tags & ground_truth_tags),
        "tag_coverage_ratio": tag_coverage_ratio,
        "total_unique_tags": total_unique_tags
    }

def test_calculate_tag_overlap():
    """Test the calculate_tag_overlap function"""
    
    # Create a test state with some content IDs
    recommendation_list = ["213977", "213987", "214002"]  # Some content IDs from the CSV
    ground_truth_list = ["213977", "214131", "214217"]    # Overlapping and different content IDs
    
    print("🧪 Testing calculate_tag_overlap function...")
    print("=" * 50)
    
    # Check if content IDs exist in the dataframe
    print(f"Available content IDs in CSV: {contents_df['content_id'].head().tolist()}")
    print(f"Test recommendation list: {recommendation_list}")
    print(f"Test ground truth list: {ground_truth_list}")
    
    # Check if our test IDs exist
    existing_rec = contents_df[contents_df['content_id'].astype(str).isin(recommendation_list)]
    existing_gt = contents_df[contents_df['content_id'].astype(str).isin(ground_truth_list)]
    
    print(f"Found {len(existing_rec)} recommended content in CSV")
    print(f"Found {len(existing_gt)} ground truth content in CSV")
    
    if len(existing_rec) == 0 or len(existing_gt) == 0:
        print("❌ Test content IDs not found in CSV. Using first few content IDs instead.")
        # Use first few content IDs from the CSV
        sample_ids = contents_df['content_id'].head(6).astype(str).tolist()
        recommendation_list = sample_ids[:3]
        ground_truth_list = sample_ids[3:]
        print(f"New recommendation list: {recommendation_list}")
        print(f"New ground truth list: {ground_truth_list}")
    
    # Run the function
    try:
        result = calculate_tag_overlap_simple(recommendation_list, ground_truth_list)
        
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
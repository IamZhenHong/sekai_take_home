import pandas as pd
from batch_generate_tags import generate_tags_for_batch

def test_tag_generation():
    """Test the batch tag generation on a few sample items."""
    
    # Load a few sample items from the CSV
    df = pd.read_csv("datasets/contents.csv")
    
    # Test on first 3 items that don't have tags
    test_items = []
    for idx, row in df.iterrows():
        if pd.isna(row.get('generated_tags')) or row['generated_tags'] == '':
            test_items.append((idx, row))
            if len(test_items) >= 3:
                break
    
    print("Testing batch tag generation on sample items...")
    print("=" * 50)
    
    # Prepare batch data
    batch_data = []
    for idx, row in test_items:
        batch_data.append({
            'title': row['title'],
            'intro': row['intro'],
            'character_list': row.get('character_list', ''),
            'initial_record': row.get('initial_record', '')
        })
    
    # Generate tags for the batch
    tag_lists = generate_tags_for_batch(batch_data)
    
    # Display results
    for i, (idx, row) in enumerate(test_items):
        print(f"\nItem {idx}: {row['title']}")
        print(f"Intro: {row['intro'][:100]}...")
        print(f"Generated tags: {tag_lists[i]}")
        print("-" * 30)

if __name__ == "__main__":
    test_tag_generation() 
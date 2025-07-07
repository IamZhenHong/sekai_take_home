import os
import pandas as pd
import json
from typing import List
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the LLM
llm = init_chat_model("openai:gpt-4o-mini")

def generate_tags_for_batch(content_batch: List[dict]) -> List[List[str]]:
    """
    Generate tags for multiple content items in a single GPT call.
    
    Args:
        content_batch: List of dictionaries containing content info
                      Each dict should have: title, intro, character_list, initial_record
        
    Returns:
        List of tag lists, one for each content item
    """
    
    # Build the batch prompt
    batch_prompt = """
You are an expert content tagger for a roleplay story platform. Your job is to analyze multiple content items and generate 10-15 relevant tags for each one.

For each content item, generate tags that capture:
- Genre (romance, fantasy, action, comedy, etc.)
- Themes (school life, supernatural, adventure, etc.)
- Character dynamics (harem, love triangle, etc.)
- Setting (school, fantasy world, modern, etc.)
- Tone (dramatic, lighthearted, intense, etc.)
- Specific elements (superpowers, magic, ninja, etc.)

Generate 10-15 tags for each content that are:
- Relevant and specific to the content
- Useful for content discovery
- Cover different aspects (genre, theme, setting, etc.)
- Written in lowercase, separated by commas

Return the results in this exact format:
CONTENT 1: tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8, tag9, tag10, tag11, tag12, tag13, tag14, tag15
CONTENT 2: tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8, tag9, tag10, tag11, tag12, tag13, tag14, tag15
CONTENT 3: tag1, tag2, tag3, tag4, tag5, tag6, tag7, tag8, tag9, tag10, tag11, tag12, tag13, tag14, tag15
...and so on for each content item.

Content to analyze:
"""
    
    # Add each content item to the prompt
    for i, content in enumerate(content_batch, 1):
        batch_prompt += f"""
CONTENT {i}:
Title: {content['title']}
Introduction: {content['intro']}
Characters: {content.get('character_list', '')}
Initial Record: {content.get('initial_record', '')}
"""
    
    try:
        response = llm.invoke(batch_prompt)
        response_text = response.content.strip()
        
        # Parse the response to extract tags for each content
        tag_lists = []
        lines = response_text.split('\n')
        
        for line in lines:
            if line.startswith('CONTENT ') and ':' in line:
                # Extract tags from the line
                tags_part = line.split(':', 1)[1].strip()
                tags = [tag.strip().lower() for tag in tags_part.split(',') if tag.strip()]
                
                # Remove duplicates while preserving order
                seen = set()
                unique_tags = []
                for tag in tags:
                    if tag not in seen:
                        seen.add(tag)
                        unique_tags.append(tag)
                
                tag_lists.append(unique_tags)
        
        # Ensure we have the right number of tag lists
        while len(tag_lists) < len(content_batch):
            tag_lists.append(['general', 'roleplay'])
        
        return tag_lists[:len(content_batch)]  # Return only what we need
        
    except Exception as e:
        print(f"Error generating tags for batch: {e}")
        # Return fallback tags for all items in the batch
        fallback_tags = ['general', 'roleplay']
        return [fallback_tags for _ in content_batch]

def batch_generate_tags(csv_path: str, output_path: str = None, start_index: int = 0, batch_size: int = 50, override_existing: bool = False):
    """
    Batch generate tags for all content in the CSV file.
    
    Args:
        csv_path: Path to the CSV file
        output_path: Path to save the updated CSV (if None, overwrites original)
        start_index: Starting index for processing (useful for resuming)
        batch_size: Number of items to process in one batch
        override_existing: If True, regenerate tags for items that already have tags
    """
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    if output_path is None:
        output_path = csv_path
    
    total_items = len(df)
    print(f"Processing {total_items} content items...")
    print(f"Starting from index {start_index}")
    print(f"Batch size: {batch_size}")
    print(f"Override existing tags: {override_existing}")
    print(f"Estimated batches: {(total_items - start_index + batch_size - 1) // batch_size}")
    
    # Count items that need processing
    items_to_process = []
    for idx in range(start_index, len(df)):
        row = df.iloc[idx]
        existing_tags = row.get('generated_tags', '')
        
        # Check if item needs processing
        needs_processing = (
            pd.isna(existing_tags) or 
            existing_tags == '' or 
            existing_tags == '[]' or 
            existing_tags == "['']" or
            override_existing
        )
        
        if needs_processing:
            items_to_process.append(idx)
    
    print(f"📊 Items that need tag generation: {len(items_to_process)}/{total_items}")
    if not override_existing:
        skipped_count = total_items - len(items_to_process)
        if skipped_count > 0:
            print(f"⏭️  Skipping {skipped_count} items that already have tags")
    
    # Process content in batches
    processed_count = 0
    for i in range(0, len(items_to_process), batch_size):
        batch_end = min(i + batch_size, len(items_to_process))
        batch_indices = items_to_process[i:batch_end]
        batch_num = i//batch_size + 1
        total_batches = (len(items_to_process) + batch_size - 1) // batch_size
        
        print(f"\n🔄 Processing batch {batch_num}/{total_batches}: items {i} to {batch_end-1}")
        print(f"📊 Progress: {processed_count}/{len(items_to_process)} items processed ({processed_count/len(items_to_process)*100:.1f}%)")
        
        # Prepare batch data
        batch_data = []
        
        for idx in batch_indices:
            row = df.iloc[idx]
            batch_data.append({
                'title': row['title'],
                'intro': row['intro'],
                'character_list': row.get('character_list', ''),
                'initial_record': row.get('initial_record', '')
            })
        
        print(f"🏷️  Generating tags for {len(batch_data)} items in batch...")
        
        # Generate tags for the entire batch
        tag_lists = generate_tags_for_batch(batch_data)
        
        # Apply tags to each item in the batch
        for idx, tags in zip(batch_indices, tag_lists):
            row = df.iloc[idx]
            title_preview = row['title'][:50] + "..." if len(row['title']) > 50 else row['title']
            
            # Check if this item already had tags
            existing_tags = row.get('generated_tags', '')
            had_existing = not (pd.isna(existing_tags) or existing_tags == '' or existing_tags == '[]' or existing_tags == "['']")
            
            if had_existing and override_existing:
                print(f"  🔄 Item {idx}: {title_preview} (overriding existing tags)")
            else:
                print(f"  📝 Item {idx}: {title_preview}")
            
            print(f"  🏷️  Generated tags: {tags}")
            
            # Convert to the same format as existing tags (string representation of list)
            tags_str = str(tags)
            df.at[idx, 'generated_tags'] = tags_str
            processed_count += 1
        
        # Save progress after each batch
        df.to_csv(output_path, index=False)
        print(f"💾 Saved progress to {output_path}")
        print(f"✅ Batch {batch_num} completed!")
    
    # Double-check and regenerate tags for empty entries
    print("\n" + "="*50)
    print("DOUBLE-CHECKING: Looking for empty tag entries...")
    print("="*50)
    
    empty_count = 0
    max_retries = 3  # Maximum retry attempts for each empty entry
    
    for attempt in range(max_retries):
        # Find rows with empty or invalid tags
        empty_mask = (
            df['generated_tags'].isna() | 
            (df['generated_tags'] == '') | 
            (df['generated_tags'] == '[]') |
            (df['generated_tags'] == "['']")
        )
        
        empty_indices = df[empty_mask].index.tolist()
        
        if not empty_indices:
            print("✅ All content now has tags!")
            break
            
        print(f"\n🔄 Attempt {attempt + 1}/{max_retries}: Found {len(empty_indices)} items with empty tags")
        
        # Process empty items in batches
        batch_size_retry = min(10, len(empty_indices))  # Smaller batches for retry
        total_retry_batches = (len(empty_indices) + batch_size_retry - 1) // batch_size_retry
        
        for batch_start in range(0, len(empty_indices), batch_size_retry):
            batch_end = min(batch_start + batch_size_retry, len(empty_indices))
            batch_indices = empty_indices[batch_start:batch_end]
            retry_batch_num = batch_start // batch_size_retry + 1
            
            print(f"  📦 Retry batch {retry_batch_num}/{total_retry_batches}: processing {len(batch_indices)} items")
            
            # Prepare batch data for empty items
            batch_data = []
            for idx in batch_indices:
                row = df.iloc[idx]
                batch_data.append({
                    'title': row['title'],
                    'intro': row['intro'],
                    'character_list': row.get('character_list', ''),
                    'initial_record': row.get('initial_record', '')
                })
            
            print(f"  🏷️  Regenerating tags for {len(batch_data)} empty items...")
            
            # Generate tags for the batch
            tag_lists = generate_tags_for_batch(batch_data)
            
            # Apply tags to each item
            for idx, tags in zip(batch_indices, tag_lists):
                row = df.iloc[idx]
                title_preview = row['title'][:50] + "..." if len(row['title']) > 50 else row['title']
                print(f"    📝 Item {idx}: {title_preview}")
                
                # Ensure we have at least one tag
                if not tags:
                    # Fallback tags based on content analysis
                    fallback_tags = ['general', 'roleplay']
                    if 'school' in row['title'].lower() or 'class' in row['title'].lower():
                        fallback_tags.append('school life')
                    if 'romance' in row['title'].lower() or 'love' in row['title'].lower():
                        fallback_tags.append('romance')
                    if 'fantasy' in row['title'].lower() or 'magic' in row['title'].lower():
                        fallback_tags.append('fantasy')
                    tags = fallback_tags
                
                tags_str = str(tags)
                df.at[idx, 'generated_tags'] = tags_str
                empty_count += 1
                
                print(f"    🏷️  Generated tags: {tags}")
        
        # Save after each retry attempt
        df.to_csv(output_path, index=False)
        print(f"  💾 Saved progress after attempt {attempt + 1}")
    
    print(f"\nCompleted! Updated CSV saved to {output_path}")
    
    # Final summary
    total_with_tags = df['generated_tags'].notna().sum()
    empty_final = df['generated_tags'].isna().sum() + (df['generated_tags'] == '').sum() + (df['generated_tags'] == '[]').sum()
    
    print(f"Final Summary:")
    print(f"Total items with tags: {total_with_tags}/{len(df)}")
    print(f"Items still empty: {empty_final}")
    
    if empty_final > 0:
        print(f"⚠️  Warning: {empty_final} items still have empty tags after {max_retries} attempts")
    else:
        print("✅ Success: All content now has at least one tag!")

def main():
    """Main function to run the tag generation."""
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Batch generate tags for content')
    parser.add_argument('--input', '-i', default="datasets/contents.csv", 
                       help='Input CSV file path (default: datasets/contents.csv)')
    parser.add_argument('--output', '-o', default="datasets/contents_with_tags.csv", 
                       help='Output CSV file path (default: datasets/contents_with_tags.csv)')
    parser.add_argument('--start-index', '-s', type=int, default=0, 
                       help='Starting index for processing (default: 0)')
    parser.add_argument('--batch-size', '-b', type=int, default=20, 
                       help='Number of items to process in one batch (default: 20)')
    parser.add_argument('--override', action='store_true', 
                       help='Override existing tags (default: False)')
    
    args = parser.parse_args()
    
    # Configuration
    csv_path = args.input
    output_path = args.output
    
    # If output path is the default, put it in datasets directory
    if output_path == "contents_with_tags.csv":
        output_path = "datasets/contents_with_tags.csv"
    
    print("Starting batch tag generation...")
    print(f"Input file: {csv_path}")
    print(f"Output file: {output_path}")
    print(f"Override existing tags: {args.override}")
    
    batch_generate_tags(
        csv_path=csv_path,
        output_path=output_path,
        start_index=args.start_index,
        batch_size=args.batch_size,
        override_existing=args.override
    )

if __name__ == "__main__":
    main() 
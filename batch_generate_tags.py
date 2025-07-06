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

def batch_generate_tags(csv_path: str, output_path: str = None, start_index: int = 0, batch_size: int = 50):
    """
    Batch generate tags for all content in the CSV file.
    
    Args:
        csv_path: Path to the CSV file
        output_path: Path to save the updated CSV (if None, overwrites original)
        start_index: Starting index for processing (useful for resuming)
        batch_size: Number of items to process in one batch
    """
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    if output_path is None:
        output_path = csv_path
    
    print(f"Processing {len(df)} content items...")
    print(f"Starting from index {start_index}")
    
    # Process content in batches
    for i in range(start_index, len(df), batch_size):
        batch_end = min(i + batch_size, len(df))
        print(f"\nProcessing batch {i//batch_size + 1}: items {i} to {batch_end-1}")
        
        # Prepare batch data
        batch_data = []
        batch_indices = []
        
        for idx in range(i, batch_end):
            row = df.iloc[idx]
            batch_data.append({
                'title': row['title'],
                'intro': row['intro'],
                'character_list': row.get('character_list', ''),
                'initial_record': row.get('initial_record', '')
            })
            batch_indices.append(idx)
        
        print(f"Generating tags for {len(batch_data)} items in batch...")
        
        # Generate tags for the entire batch
        tag_lists = generate_tags_for_batch(batch_data)
        
        # Apply tags to each item in the batch
        for idx, tags in zip(batch_indices, tag_lists):
            row = df.iloc[idx]
            print(f"Item {idx}: {row['title'][:50]}...")
            print(f"Generated tags: {tags}")
            
            # Convert to the same format as existing tags (string representation of list)
            tags_str = str(tags)
            df.at[idx, 'generated_tags'] = tags_str
        
        # Save progress after each batch
        df.to_csv(output_path, index=False)
        print(f"Saved progress to {output_path}")
    
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
            
        print(f"\nAttempt {attempt + 1}: Found {len(empty_indices)} items with empty tags")
        
        # Process empty items in batches
        batch_size_retry = min(10, len(empty_indices))  # Smaller batches for retry
        
        for batch_start in range(0, len(empty_indices), batch_size_retry):
            batch_end = min(batch_start + batch_size_retry, len(empty_indices))
            batch_indices = empty_indices[batch_start:batch_end]
            
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
            
            print(f"Regenerating tags for {len(batch_data)} empty items in batch...")
            
            # Generate tags for the batch
            tag_lists = generate_tags_for_batch(batch_data)
            
            # Apply tags to each item
            for idx, tags in zip(batch_indices, tag_lists):
                row = df.iloc[idx]
                print(f"Item {idx}: {row['title'][:50]}...")
                
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
                
                print(f"Generated tags: {tags}")
        
        # Save after each retry attempt
        df.to_csv(output_path, index=False)
        print(f"Saved progress after attempt {attempt + 1}")
    
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
    
    # Configuration
    csv_path = "datasets/contents.csv"
    output_path = "contents_with_tags.csv"  # Save to new file to be safe
    
    print("Starting batch tag generation...")
    print(f"Input file: {csv_path}")
    print(f"Output file: {output_path}")
    
    # You can customize these parameters:
    start_index = 0  # Start from beginning
    batch_size = 20  # Process 20 items at a time
    
    batch_generate_tags(
        csv_path=csv_path,
        output_path=output_path,
        start_index=start_index,
        batch_size=batch_size
    )

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Script to embed contents from contents_with_tags.csv using chroma_utils.py
"""

import pandas as pd
import os
from .chroma_utils import embed_and_store_content

def main():
    """
    Main function to load contents_with_tags.csv and embed them using chroma_utils
    """
    print("🚀 Starting content embedding process...")
    
    # Check if contents_with_tags.csv exists in current directory or datasets directory
    csv_path = "contents_with_tags.csv"
    if not os.path.exists(csv_path):
        csv_path = "datasets/contents_with_tags.csv"
        if not os.path.exists(csv_path):
            print("❌ Error: contents_with_tags.csv not found in current directory or datasets/")
            return
    
    try:
        # Load the CSV file
        print(f"📖 Loading {csv_path}...")
        content_df = pd.read_csv(csv_path)
        
        print(f"✅ Successfully loaded {len(content_df)} content entries")
        print(f"📊 Columns found: {list(content_df.columns)}")
        
        # Display some basic statistics
        print(f"📈 Content statistics:")
        print(f"   - Total entries: {len(content_df)}")
        print(f"   - Titles with content: {content_df['title'].notna().sum()}")
        print(f"   - Intros with content: {content_df['intro'].notna().sum()}")
        print(f"   - Initial records with content: {content_df['initial_record'].notna().sum()}")
        
        # Check for required columns
        required_columns = ['content_id', 'title', 'intro', 'initial_record']
        missing_columns = [col for col in required_columns if col not in content_df.columns]
        
        if missing_columns:
            print(f"❌ Error: Missing required columns: {missing_columns}")
            return
        
        # Clean the data - replace NaN values with empty strings
        print("🧹 Cleaning data...")
        content_df = content_df.fillna('')
        
        # Embed and store the content
        print("🔧 Starting embedding process...")
        embed_and_store_content(
            content_df=content_df,
            collection_name="sekai_content",
            delete_existing=True  # Set to True to recreate the collection
        )
        
        print("✅ Content embedding process completed!")
        
    except Exception as e:
        print(f"❌ Error during embedding process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
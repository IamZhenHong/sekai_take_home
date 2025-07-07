#!/usr/bin/env python3
"""
Advanced script to embed combined contents from contents_with_tags.csv using chroma_utils.py
Creates one embedding per content item by concatenating title, intro, and initial_record
"""

import pandas as pd
import os
import argparse
import sys
from .chroma_utils import embed_and_store_combined_content, chroma_client, get_openai_embedding_function

def validate_csv_structure(content_df):
    """
    Validate that the CSV has the expected structure
    """
    required_columns = ['content_id', 'title', 'intro', 'initial_record']
    missing_columns = [col for col in required_columns if col not in content_df.columns]
    
    if missing_columns:
        print(f"❌ Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(content_df.columns)}")
        return False
    
    return True

def display_content_stats(content_df):
    """
    Display statistics about the content
    """
    print(f"📊 Content Statistics:")
    print(f"   - Total entries: {len(content_df)}")
    print(f"   - Titles with content: {content_df['title'].str.strip().ne('').sum()}")
    print(f"   - Intros with content: {content_df['intro'].str.strip().ne('').sum()}")
    print(f"   - Initial records with content: {content_df['initial_record'].str.strip().ne('').sum()}")
    
    # Show sample of content IDs
    sample_ids = content_df['content_id'].head(5).tolist()
    print(f"   - Sample content IDs: {sample_ids}")
    
    # Show combined content length statistics
    combined_lengths = []
    for _, row in content_df.iterrows():
        title = str(row.get('title', ''))
        intro = str(row.get('intro', ''))
        initial_record = str(row.get('initial_record', ''))
        
        # Clean up fields
        if title.lower() == 'nan':
            title = ""
        if intro.lower() == 'nan':
            intro = ""
        if initial_record.lower() == 'nan':
            initial_record = ""
        
        combined_text = ""
        if title:
            combined_text += f"Title: {title}\n\n"
        if intro:
            combined_text += f"Introduction: {intro}\n\n"
        if initial_record:
            combined_text += f"Initial Record: {initial_record}"
        
        combined_lengths.append(len(combined_text.strip()))
    
    if combined_lengths:
        print(f"   - Average combined text length: {sum(combined_lengths) // len(combined_lengths)} characters")
        print(f"   - Min combined text length: {min(combined_lengths)} characters")
        print(f"   - Max combined text length: {max(combined_lengths)} characters")

def clean_content_data(content_df):
    """
    Clean the content data for embedding
    """
    print("🧹 Cleaning content data...")
    
    # Replace NaN values with empty strings
    content_df = content_df.fillna('')
    
    # Strip whitespace from text fields
    text_columns = ['title', 'intro', 'initial_record']
    for col in text_columns:
        if col in content_df.columns:
            content_df[col] = content_df[col].astype(str).str.strip()
    
    # Remove rows where all text fields are empty
    text_content = content_df[text_columns].apply(lambda x: x.str.len() > 0).any(axis=1)
    valid_rows = content_df[text_content]
    
    if len(valid_rows) < len(content_df):
        removed_count = len(content_df) - len(valid_rows)
        print(f"⚠️  Removed {removed_count} rows with no text content")
    
    return valid_rows

def search_combined_content(query, collection_name="sekai_content_combined", n_results=5):
    """
    Perform semantic search on combined embedded content
    """
    try:
        # Get the collection
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=get_openai_embedding_function()
        )
        
        # Perform the search
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
        
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return None

def display_combined_results(results, query):
    """
    Display search results for combined content in a formatted way
    """
    if not results or not results['ids'] or not results['ids'][0]:
        print("❌ No results found")
        return
    
    print(f"\n🔍 Search Results for: '{query}'")
    print("=" * 60)
    
    for i, (doc_id, document, metadata, distance) in enumerate(zip(
        results['ids'][0], 
        results['documents'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    )):
        print(f"\n📄 Result {i+1} (Similarity: {1-distance:.3f})")
        print(f"   Content ID: {doc_id}")
        print(f"   Title: {metadata.get('title', 'N/A')}")
        
        # Parse the combined document to show structure
        lines = document.split('\n')
        current_section = ""
        
        for line in lines:
            if line.startswith('Title:'):
                current_section = "Title"
                title = line.replace('Title:', '').strip()
                if title:
                    print(f"   Title: {title}")
            elif line.startswith('Introduction:'):
                current_section = "Introduction"
                intro = line.replace('Introduction:', '').strip()
                if intro:
                    print(f"   Introduction: {intro[:100]}{'...' if len(intro) > 100 else ''}")
            elif line.startswith('Initial Record:'):
                current_section = "Initial Record"
                record = line.replace('Initial Record:', '').strip()
                if record:
                    print(f"   Initial Record: {record[:100]}{'...' if len(record) > 100 else ''}")
            elif line.strip() and current_section:
                # Continue the current section
                if current_section == "Introduction":
                    intro_part = line.strip()
                    if len(intro_part) > 100:
                        print(f"   ... {intro_part[:100]}...")
                    break
                elif current_section == "Initial Record":
                    record_part = line.strip()
                    if len(record_part) > 100:
                        print(f"   ... {record_part[:100]}...")
                    break
        
        print("-" * 40)

def interactive_search(collection_name="sekai_content_combined"):
    """
    Run interactive search mode for combined content
    """
    print("\n🔍 Interactive Combined Content Search Mode")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for commands")
    print("=" * 50)
    
    while True:
        try:
            query = input("\n🔍 Enter your search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if query.lower() == 'help':
                print("\n📖 Available commands:")
                print("  - Type any text to search across all content")
                print("  - 'limit:10 <query>' - limit results to 10")
                print("  - 'quit' or 'exit' - stop searching")
                print("  - Search will match across title, intro, and initial record combined")
                continue
                
            if not query:
                continue
            
            # Parse limit
            n_results = 5
            if query.startswith('limit:'):
                parts = query.split(' ', 1)
                if len(parts) == 2:
                    try:
                        limit = int(parts[0].split(':', 1)[1])
                        if 1 <= limit <= 50:
                            n_results = limit
                            query = parts[1]
                        else:
                            print("❌ Limit must be between 1 and 50")
                            continue
                    except ValueError:
                        print("❌ Invalid limit number")
                        continue
            
            # Perform search
            results = search_combined_content(query, collection_name, n_results)
            if results:
                display_combined_results(results, query)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def check_combined_embeddings_exist(content_df, collection_name="sekai_content_combined"):
    """
    Check if combined embeddings exist for the content
    """
    try:
        # Try to get the collection
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=get_openai_embedding_function()
        )
        
        # Check if collection has embeddings
        collection_count = collection.count()
        
        if collection_count > 0:
            # Sample check to ensure embeddings are accessible
            sample_id = str(content_df['content_id'].iloc[0])
            try:
                collection.get(ids=[sample_id])
                print(f"✅ Found existing combined embeddings in collection '{collection_name}' ({collection_count} embeddings)")
                return True
            except Exception as e:
                print(f"⚠️ Found corrupted collection. Recreating embeddings... Error: {e}")
        else:
            print(f"⚠️ Found empty collection. Creating embeddings...")
        
        return False
        
    except Exception:
        print(f"⚠️ No existing combined embeddings found in collection '{collection_name}'. Creating new embeddings...")
        return False

def main():
    """
    Main function with command line argument support
    """
    parser = argparse.ArgumentParser(description='Embed combined contents from CSV using chroma_utils')
    parser.add_argument('--csv-file', default='datasets/contents_with_tags.csv', 
                       help='Path to the CSV file (default: datasets/contents_with_tags.csv)')
    parser.add_argument('--collection-name', default='sekai_content_combined',
                       help='Name of the ChromaDB collection (default: sekai_content_combined)')
    parser.add_argument('--delete-existing', action='store_true',
                       help='Delete existing collection before creating new one')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check if embeddings exist, don\'t create new ones')
    parser.add_argument('--force', action='store_true',
                       help='Force recreation even if embeddings exist')
    parser.add_argument('--search-after', action='store_true',
                       help='Start interactive search after embedding')
    parser.add_argument('--test-query', type=str,
                       help='Test a search query after embedding')
    
    args = parser.parse_args()
    
    print("🚀 Starting combined content embedding process...")
    print(f"📁 CSV file: {args.csv_file}")
    print(f"🗂️  Collection: {args.collection_name}")
    print("📝 Note: This creates ONE embedding per content item (title + intro + initial_record combined)")
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"❌ Error: {args.csv_file} not found")
        sys.exit(1)
    
    try:
        # Load the CSV file
        print(f"📖 Loading {args.csv_file}...")
        content_df = pd.read_csv(args.csv_file)
        
        print(f"✅ Successfully loaded {len(content_df)} content entries")
        
        # Validate structure
        if not validate_csv_structure(content_df):
            sys.exit(1)
        
        # Display statistics
        display_content_stats(content_df)
        
        # Clean the data
        content_df = clean_content_data(content_df)
        
        if len(content_df) == 0:
            print("❌ Error: No valid content found after cleaning")
            sys.exit(1)
        
        print(f"✅ {len(content_df)} valid entries ready for combined embedding")
        
        if args.check_only:
            # Only check if embeddings exist
            print("🔍 Checking if combined embeddings exist...")
            embeddings_exist = check_combined_embeddings_exist(content_df, args.collection_name)
            if embeddings_exist:
                print("✅ Combined embeddings already exist and are accessible")
            else:
                print("❌ Combined embeddings don't exist or are corrupted")
            return
        
        # Check if embeddings already exist (unless force is specified)
        if not args.force and not args.delete_existing:
            print("🔍 Checking if combined embeddings already exist...")
            embeddings_exist = check_combined_embeddings_exist(content_df, args.collection_name)
            if embeddings_exist:
                print("✅ Combined embeddings already exist. Use --force or --delete-existing to recreate")
                return
        
        # Embed and store the combined content
        print("🔧 Starting combined embedding process...")
        embed_and_store_combined_content(
            content_df=content_df,
            collection_name=args.collection_name,
            delete_existing=args.delete_existing
        )
        
        print("✅ Combined content embedding process completed successfully!")
        print(f"📊 Created {len(content_df)} combined embeddings (one per content item)")
        
        # Handle post-embedding search options
        if args.test_query:
            print(f"\n🧪 Testing search with query: '{args.test_query}'")
            results = search_combined_content(args.test_query, args.collection_name)
            if results:
                display_combined_results(results, args.test_query)
        
        if args.search_after:
            interactive_search(args.collection_name)
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during combined embedding process: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 
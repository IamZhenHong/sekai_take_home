#!/usr/bin/env python3
"""
Advanced script to embed contents from contents_with_tags.csv using chroma_utils.py
Provides additional options and better error handling
"""

import pandas as pd
import os
import argparse
import sys
from .chroma_utils import embed_and_store_content, ensure_embeddings_exist, chroma_client, get_openai_embedding_function

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

def search_content(query, collection_name="sekai_content", n_results=5, field_filter=None):
    """
    Perform semantic search on embedded content
    """
    try:
        # Get the collection
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=get_openai_embedding_function()
        )
        
        # Prepare where clause for field filtering
        where_clause = None
        if field_filter:
            where_clause = {"field": field_filter}
        
        # Perform the search
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
        
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return None

def display_results(results, query):
    """
    Display search results in a formatted way
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
        print(f"   ID: {doc_id}")
        print(f"   Content ID: {metadata.get('content_id', 'N/A')}")
        print(f"   Field: {metadata.get('field', 'N/A')}")
        print(f"   Title: {metadata.get('title', 'N/A')}")
        print(f"   Content: {document[:200]}{'...' if len(document) > 200 else ''}")
        
        # Show more content if it's short
        if len(document) <= 200:
            print(f"   Full Content: {document}")
        
        print("-" * 40)

def interactive_search(collection_name="sekai_content"):
    """
    Run interactive search mode
    """
    print("\n🔍 Interactive Semantic Search Mode")
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
                print("  - Type any text to search")
                print("  - 'field:title <query>' - search only titles")
                print("  - 'field:intro <query>' - search only introductions")
                print("  - 'field:initial_record <query>' - search only initial records")
                print("  - 'limit:10 <query>' - limit results to 10")
                print("  - 'quit' or 'exit' - stop searching")
                continue
                
            if not query:
                continue
            
            # Parse field filter
            field_filter = None
            if query.startswith('field:'):
                parts = query.split(' ', 1)
                if len(parts) == 2:
                    field_type = parts[0].split(':', 1)[1]
                    if field_type in ['title', 'intro', 'initial_record']:
                        field_filter = field_type
                        query = parts[1]
                    else:
                        print(f"❌ Invalid field type: {field_type}")
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
            results = search_content(query, collection_name, n_results, field_filter)
            if results:
                display_results(results, query)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """
    Main function with command line argument support
    """
    parser = argparse.ArgumentParser(description='Embed contents from CSV using chroma_utils')
    parser.add_argument('--csv-file', default='datasets/contents_with_tags.csv', 
                       help='Path to the CSV file (default: datasets/contents_with_tags.csv)')
    parser.add_argument('--collection-name', default='sekai_content',
                       help='Name of the ChromaDB collection (default: sekai_content)')
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
    
    print("🚀 Starting content embedding process...")
    print(f"📁 CSV file: {args.csv_file}")
    print(f"🗂️  Collection: {args.collection_name}")
    
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
        
        print(f"✅ {len(content_df)} valid entries ready for embedding")
        
        if args.check_only:
            # Only check if embeddings exist
            print("🔍 Checking if embeddings exist...")
            embeddings_exist = ensure_embeddings_exist(content_df, args.collection_name)
            if embeddings_exist:
                print("✅ Embeddings already exist and are accessible")
            else:
                print("❌ Embeddings don't exist or are corrupted")
            return
        
        # Check if embeddings already exist (unless force is specified)
        if not args.force and not args.delete_existing:
            print("🔍 Checking if embeddings already exist...")
            embeddings_exist = ensure_embeddings_exist(content_df, args.collection_name)
            if embeddings_exist:
                print("✅ Embeddings already exist. Use --force or --delete-existing to recreate")
                return
        
        # Embed and store the content
        print("🔧 Starting embedding process...")
        embed_and_store_content(
            content_df=content_df,
            collection_name=args.collection_name,
            delete_existing=args.delete_existing
        )
        
        print("✅ Content embedding process completed successfully!")
        
        # Handle post-embedding search options
        if args.test_query:
            print(f"\n🧪 Testing search with query: '{args.test_query}'")
            results = search_content(args.test_query, args.collection_name)
            if results:
                display_results(results, args.test_query)
        
        if args.search_after:
            interactive_search(args.collection_name)
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during embedding process: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 
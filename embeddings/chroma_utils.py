import pandas as pd
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.config import Settings
from openai import OpenAI
from typing import List
from chromadb.api.types import EmbeddingFunction

class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=input,
            model=self.model
        )
        return [item.embedding for item in response.data]

# Load environment variables
load_dotenv()

# Initialize clients at module level with persistent storage
try:
    # Ensure the persistent directory exists
    os.makedirs("./chroma_db", exist_ok=True)

    # ✅ Updated Chroma client instantiation (v4+ compatible)
    chroma_client = chromadb.Client(Settings(
        persist_directory="./chroma_db",
        anonymized_telemetry=False,
        is_persistent=True
    ))

    # ✅ Initialize OpenAI client
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

except Exception as e:
    print(f"❌ Error initializing clients in chroma_utils: {e}")
    chroma_client = None
    openai_client = None

# Create a persistent embedding function
def get_openai_embedding_function():
    """Create and return an OpenAI embedding function."""
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

def ensure_embeddings_exist(content_df: pd.DataFrame, collection_name: str = "sekai_content") -> bool:
    """
    Check if embeddings exist for the content and create them if they don't.
    Returns True if embeddings are ready to use, False otherwise.
    """
    global chroma_client, openai_client
    
    if not chroma_client or not openai_client:
        try:
            if not openai_client:
                openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if not chroma_client:
                os.makedirs("./chroma_db", exist_ok=True)
                chroma_client = chromadb.Client(Settings(
                    persist_directory="./chroma_db",
                    chroma_db_impl="duckdb+parquet",
                ))
        except Exception as e:
            print(f"Error initializing clients: {e}")
            return False

    try:
        # Try to get the collection
        try:
            collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=get_openai_embedding_function()
            )
            
            # Check if collection has embeddings
            collection_count = collection.count()
            
            if collection_count > 0:
                # Sample check to ensure embeddings are accessible
                sample_id = f"{content_df['content_id'].iloc[0]}_intro"
                try:
                    collection.get(ids=[sample_id])
                    print(f"✅ Found existing embeddings in collection '{collection_name}' ({collection_count} embeddings)")
                    return True
                except Exception as e:
                    print(f"⚠️ Found corrupted collection. Recreating embeddings... Error: {e}")
            else:
                print(f"⚠️ Found empty collection. Creating embeddings...")
            
            # If we get here, we need to recreate the embeddings
            embed_and_store_content(content_df, collection_name)
            return True
            
        except Exception:
            print(f"⚠️ No existing embeddings found in collection '{collection_name}'. Creating new embeddings...")
            embed_and_store_content(content_df, collection_name)
            return True
            
    except Exception as e:
        print(f"❌ Error checking/creating embeddings: {e}")
        return False

def embed_and_store_content(content_df: pd.DataFrame, collection_name: str = "sekai_content", delete_existing: bool = False):
    """
    Embeds content fields (title, intro, initial_record) and stores them in ChromaDB.
    If delete_existing is True, will delete and recreate the collection to ensure clean state.
    Otherwise, will add to or create the collection if it doesn't exist.
    """
    global chroma_client, openai_client
    
    if not chroma_client or not openai_client:
        try:
            if not openai_client:
                openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if not chroma_client:
                os.makedirs("./chroma_db", exist_ok=True)
                chroma_client = chromadb.Client(Settings(
                    persist_directory="./chroma_db",
                    chroma_db_impl="duckdb+parquet",
                ))
        except Exception as e:
            print(f"Error initializing clients: {e}")
            return

    try:
        # Delete existing collection if requested
        if delete_existing:
            try:
                chroma_client.delete_collection(name=collection_name)
                print(f"🗑️  '{collection_name}' deleted")
            except Exception:
                pass  # Collection didn't exist, which is fine

        # Create or get collection with OpenAI embedding function
        try:
            collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=OpenAIEmbeddingFunction()
            )
            print(f"✨ Using existing collection '{collection_name}'")
        except Exception:
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=OpenAIEmbeddingFunction()
            )
            print(f"✨ Created new collection '{collection_name}'")

        ids, documents, metadatas = [], [], []
        print("📝 Preparing documents for embedding...")
        
        for _, row in content_df.iterrows():
            cid = str(row["content_id"])
            
            title = str(row.get("title", ""))
            intro = str(row.get("intro", ""))
            initial_record = str(row.get("initial_record", ""))

            base_metadata = {
                "content_id": cid,
                "title": title,
                "intro": intro,
                "initial_record": initial_record
            }

            if title and title.lower() != 'nan':
                ids.append(f"{cid}_title")
                documents.append(title)
                metadatas.append({**base_metadata, "field": "title"})

            if intro and intro.lower() != 'nan':
                ids.append(f"{cid}_intro")
                documents.append(intro)
                metadatas.append({**base_metadata, "field": "intro"})

            if initial_record and pd.notna(initial_record) and initial_record.lower() != 'nan':
                ids.append(f"{cid}_initial_record")
                documents.append(initial_record)
                metadatas.append({**base_metadata, "field": "initial_record"})

        if documents:
            print(f"📦 Adding {len(documents)} documents to ChromaDB...")
            # Add documents in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                collection.add(
                    ids=ids[i:end_idx],
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
                print(f"   ✅ Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            print(f"✨ Successfully embedded and stored {len(documents)} documents in collection '{collection_name}'")
        else:
            print("⚠️ No valid documents to embed")

    except Exception as e:
        print(f"❌ Error embedding and storing content: {e}")

def embed_and_store_combined_content(content_df: pd.DataFrame, collection_name: str = "sekai_content_combined", delete_existing: bool = False):
    """
    Embeds combined content (title + intro + initial_record concatenated) and stores them in ChromaDB.
    Creates one embedding per content item by concatenating all fields.
    If delete_existing is True, will delete and recreate the collection to ensure clean state.
    Otherwise, will add to or create the collection if it doesn't exist.
    """
    global chroma_client, openai_client
    
    if not chroma_client or not openai_client:
        try:
            if not openai_client:
                openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if not chroma_client:
                os.makedirs("./chroma_db", exist_ok=True)
                chroma_client = chromadb.Client(Settings(
                    persist_directory="./chroma_db",
                    chroma_db_impl="duckdb+parquet",
                ))
        except Exception as e:
            print(f"Error initializing clients: {e}")
            return

    try:
        # Delete existing collection if requested
        if delete_existing:
            try:
                chroma_client.delete_collection(name=collection_name)
                print(f"🗑️  '{collection_name}' deleted")
            except Exception:
                pass  # Collection didn't exist, which is fine

        # Create or get collection with OpenAI embedding function
        try:
            collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=OpenAIEmbeddingFunction()
            )
            print(f"✨ Using existing collection '{collection_name}'")
        except Exception:
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=OpenAIEmbeddingFunction()
            )
            print(f"✨ Created new collection '{collection_name}'")

        ids, documents, metadatas = [], [], []
        print("📝 Preparing combined documents for embedding...")
        
        for _, row in content_df.iterrows():
            cid = str(row["content_id"])
            
            # Get individual fields
            title = str(row.get("title", ""))
            intro = str(row.get("intro", ""))
            initial_record = str(row.get("initial_record", ""))

            # Clean up fields (remove 'nan' strings)
            if title.lower() == 'nan':
                title = ""
            if intro.lower() == 'nan':
                intro = ""
            if initial_record.lower() == 'nan':
                initial_record = ""

            # Combine all fields into one document
            combined_text = ""
            if title:
                combined_text += f"Title: {title}\n\n"
            if intro:
                combined_text += f"Introduction: {intro}\n\n"
            if initial_record:
                combined_text += f"Initial Record: {initial_record}"

            # Only add if we have some content
            if combined_text.strip():
                ids.append(cid)  # Use content_id as the embedding ID
                documents.append(combined_text.strip())
                metadatas.append({
                    "content_id": cid,
                    "title": title,
                    "intro": intro,
                    "initial_record": initial_record,
                    "field": "combined"
                })

        if documents:
            print(f"📦 Adding {len(documents)} combined documents to ChromaDB...")
            # Add documents in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                collection.add(
                    ids=ids[i:end_idx],
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
                print(f"   ✅ Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            print(f"✨ Successfully embedded and stored {len(documents)} combined documents in collection '{collection_name}'")
        else:
            print("⚠️ No valid documents to embed")

    except Exception as e:
        print(f"❌ Error embedding and storing combined content: {e}")

def get_embedding(texts):
    """
    Get embeddings for a list of texts using OpenAI's API.
    This is used as a fallback if the embedding function is not working.
    """
    global openai_client
    if not openai_client:
        try:
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            return []

    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [r.embedding for r in response.data]

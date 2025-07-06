import os
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Set, Optional
from difflib import SequenceMatcher
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain.tools import Tool, StructuredTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import re
import json
import pandas as pd
import os
import argparse
import sys
import time
from functools import lru_cache
from embeddings.chroma_utils import chroma_client, get_openai_embedding_function
from embeddings.chroma_utils import OpenAIEmbeddingFunction

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load dataset once and preprocess
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
contents_path = os.path.join(base_dir, "datasets", "contents.csv")
contents = pd.read_csv(contents_path)  # Assumes columns: content_id, title, intro, generated_tags

# Preprocess and cache tag data for faster matching
def preprocess_tags():
    """Preprocess tags for faster matching"""
    tag_cache = {}
    for idx, row in contents.iterrows():
        content_id = row['content_id']
        generated_tags = row['generated_tags']
        
        if not isinstance(generated_tags, list):
            try:
                generated_tags = eval(generated_tags)
            except:
                generated_tags = []
        
        # Normalize tags to lowercase for faster matching
        normalized_tags = [tag.lower().strip() for tag in generated_tags if tag]
        tag_cache[content_id] = normalized_tags
    
    return tag_cache

# Initialize tag cache
TAG_CACHE = preprocess_tags()

# Cache ChromaDB collection
@lru_cache(maxsize=1)
def get_chroma_collection():
    """Get cached ChromaDB collection"""
    return chroma_client.get_collection(
        name="sekai_content",
        embedding_function=OpenAIEmbeddingFunction()
    )

# --- LangChain LLM Setup ---
llm = init_chat_model("openai:gpt-4.1-mini")

# --- Tool: Optimized Fuzzy Tag Filter ---
from rapidfuzz import fuzz, process

from difflib import SequenceMatcher

def fuzzy_tag_match(user_tags: List[str], threshold=0.75, min_results=50, max_results=70) -> List[int]:
    """Optimized fuzzy tag matching with fallback floor and tail extension"""
    print(f"🔍 Running optimized fuzzy_tag_match with tags: {user_tags}")
    
    # Normalize user tags
    user_tag_set = {tag.lower().strip() for tag in user_tags}

    def fast_score(row):
        content_tags = row.get("generated_tags", "")
        if isinstance(content_tags, str) and content_tags.startswith("["):
            try:
                tag_set = {t.lower().strip() for t in eval(content_tags) if isinstance(t, str)}
            except:
                return 0
        else:
            return 0

        # Fast path: exact matches
        exact_matches = len(user_tag_set & tag_set)
        if exact_matches > 0:
            return exact_matches * 2
        
        # Fuzzy fallback
        fuzzy_matches = 0
        for utag in user_tag_set:
            for ctag in tag_set:
                if SequenceMatcher(None, utag, ctag).ratio() >= threshold:
                    fuzzy_matches += 1
                    break
        return fuzzy_matches

    # Score all content
    contents["match_score"] = contents.apply(fast_score, axis=1)

    # Sort by score
    sorted_df = contents.sort_values("match_score", ascending=False)
    top_df = sorted_df[sorted_df["match_score"] > 0]

    if len(top_df) >= max_results:
        result = top_df.head(max_results)
    elif len(top_df) >= min_results:
        result = top_df
    else:
        # Pad with non-matching content if needed
        padding_needed = min_results - len(top_df)
        padding_df = sorted_df[~sorted_df.index.isin(top_df.index)].head(padding_needed)
        result = pd.concat([top_df, padding_df])

    print(f"📊 Returning {len(result)} items (matches: {len(top_df)})")
    return result["content_id"].tolist()

def semantic_search(query: str, filtered_content_ids: List[int], num_results=30) -> List[int]:
    """
    Optimized semantic search using ChromaDB and OpenAI embeddings.
    Returns a ranked list of content_ids from the provided filtered list.
    """
    if not query.strip():
        raise ValueError("❌ Query string is empty.")
    
    if not filtered_content_ids:
        raise ValueError("❌ No content IDs provided for semantic filtering.")

    try:
        # Use cached collection
        collection = get_chroma_collection()
        print("✅ Chroma collection loaded successfully")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load Chroma collection: {e}")

    # Convert all filtered content IDs to strings once
    filtered_str_ids = [str(cid) for cid in filtered_content_ids]

    try:
        results = collection.query(
            query_texts=[query],
            n_results=num_results,
            where={"content_id": {"$in": filtered_str_ids}},
            include=["metadatas", "distances"]
        )
    except Exception as e:
        raise RuntimeError(f"❌ Error during semantic search query: {e}")

    metadatas = results.get("metadatas", [[]])[0]
    if not metadatas:
        print("⚠️ No semantic matches found.")
        return []

    # Extract and filter returned content_ids using set for O(1) lookup
    filtered_set = set(filtered_str_ids)
    matched_ids = [
        meta.get("content_id")
        for meta in metadatas
        if meta.get("content_id") in filtered_set
    ]

    print(f"✅ Returning {len(matched_ids)} matched content IDs (filtered and ranked)")
    return [int(cid) for cid in matched_ids]

def tag_filter_tool_fn(tags_input: str) -> List[int]:
    tags = [t.strip() for t in tags_input.split(",")]
    return fuzzy_tag_match(tags)

tag_filter_tool = Tool(
    name="filter_by_tags",
    func=tag_filter_tool_fn,
    description="Filters content by a comma-separated list of tags."
)

class SemanticSearchArgs(BaseModel):
    query: str
    filtered_content_ids: List[int]
    threshold: float = 0.75
    num_results: int = 20

semantic_search_tool = StructuredTool.from_function(   
    name="semantic_search",
    description="Semantic search for content based on a query.",
    func=semantic_search,
    args_schema=SemanticSearchArgs
)

# --- Output model from LangChain agent ---
class RecommendationOutput(BaseModel):
    recommendation_list: List[int]

# Cache for agent creation
_agent_cache = {}

def create_dynamic_react_agent(system_prompt: str):
    """Create a ReAct agent with caching for better performance"""
    # Use hash of system prompt as cache key
    prompt_hash = hash(system_prompt)
    
    if prompt_hash not in _agent_cache:
        _agent_cache[prompt_hash] = create_react_agent(
            model=llm,
            tools=[tag_filter_tool, semantic_search_tool],
            prompt=system_prompt,
            response_format=RecommendationOutput,
        )
    
    return _agent_cache[prompt_hash]

# --- LangGraph Node Wrapper ---
def recommendation_agent_node(state: dict) -> dict:
    if not state.get("user_tags") and not state.get("recommendation_prompt"):
        return state

    prompt = state.get("recommendation_prompt")
    if not prompt:
        tags = state.get("user_tags", [])
        if not tags:
            return state
        tags_str = ", ".join(tags)
        prompt = f"User taste profile: {tags_str} driven preferences. User tags: {tags_str}"

    # Build system prompt more efficiently
    system_prompt_parts = [
        """You are a **recommendation agent** working within a multi-agent loop to deliver the 10 most relevant Sekai roleplay stories for a specific user.

Your results will be compared against a high-quality ground-truth recommendation list generated from the user's full history and profile. Accuracy matters—your goal is **precision, personal resonance, and thematic fit**.

Follow the steps carefully:
Below is the learned taste profile of the user and guidelines to follow :""",
        state.get("recommendation_prompt", ""),
        f"User's interest tags: {state.get('user_tags')}",
        """Instructions:
1. From the user's taste profile and initial interest tags, extract a comprehensive list of 15-25 comma-separated tags, (e.g. romance, space, anime).
2. Use the "filter_by_tags" tool to get top 100 matched content items' content_id list
3. Then, use the taste profile to craft a comprehensive query, and the filtered_list from the previous step into the "semantic_search" tool to get top 30 most relevant content items.
4. Once you have the filtered items, review them and pick the 10 most relevant based on the user's taste profile.
5. Return your reasoning followed by a JSON array of 10 content IDs.

ONLY return the JSON array at the end, like: [123, 456, 789, ...]"""
    ]
    
    system_prompt = "\n".join(system_prompt_parts)

    # Create agent with dynamic system prompt (now cached)
    dynamic_agent = create_dynamic_react_agent(system_prompt)
    
    user_message = "Recommend me great stories based on my taste profile!"
    messages = [{"role": "user", "content": user_message}]

    try:
        start_time = time.time()
        result = dynamic_agent.invoke(
            {"messages": messages},
            {"recursion_limit": 10}
        )
        end_time = time.time()
        recommendation_time = round(end_time - start_time, 2)

        rec_list = result['structured_response'].recommendation_list

        if rec_list is None:
            return state

        print("RECOMMENDATION PROMPT: ", prompt)
        return {
            **state,
            "iteration": state.get("iteration", 0) + 1,
            "recommendation_prompt": prompt,
            "recommendation_list": rec_list,
            "structured_response": result,
            "recommendation_time": recommendation_time
        }

    except Exception as e:
        return state

# --- Build LangGraph ---
graph = StateGraph(dict, output=dict)
graph.add_node("recommendation", recommendation_agent_node)
graph.add_edge(START, "recommendation")
graph.add_edge("recommendation", END)
recommendation_graph = graph.compile()

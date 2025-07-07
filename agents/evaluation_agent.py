from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
import pandas as pd
import random
import os
import numpy as np
import time
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
import chromadb
from embeddings.chroma_utils import chroma_client
from embeddings.chroma_utils import OpenAIEmbeddingFunction
from functools import lru_cache
import hashlib
from difflib import SequenceMatcher

llm = init_chat_model("openai:o4-mini")
# llm = init_chat_model("openai:gpt-4.1-mini")
# --- Load datasets lazily to avoid import-time file access ---
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Check both root directory and datasets directory for contents_with_tags.csv
contents_path = os.path.join(base_dir, "contents_with_tags.csv")
if not os.path.exists(contents_path):
    contents_path = os.path.join(base_dir, "datasets", "contents_with_tags.csv")
interactions_path = os.path.join(base_dir, "datasets/interactions.csv")

def get_contents_df():
    """Get contents DataFrame, loading it if not already loaded"""
    if not hasattr(get_contents_df, '_contents_df'):
        try:
            get_contents_df._contents_df = pd.read_csv(contents_path)
            get_contents_df._contents_df["content_id"] = get_contents_df._contents_df["content_id"].astype(str)
        except FileNotFoundError:
            print(f"⚠️  Warning: {contents_path} not found. Creating empty DataFrame.")
            get_contents_df._contents_df = pd.DataFrame(columns=['content_id', 'title', 'intro', 'generated_tags'])
    return get_contents_df._contents_df

def get_interactions_df():
    """Get interactions DataFrame, loading it if not already loaded"""
    if not hasattr(get_interactions_df, '_interactions_df'):
        try:
            get_interactions_df._interactions_df = pd.read_csv(interactions_path)
            get_interactions_df._interactions_df["content_id"] = get_interactions_df._interactions_df["content_id"].astype(str)
        except FileNotFoundError:
            print(f"⚠️  Warning: {interactions_path} not found. Creating empty DataFrame.")
            get_interactions_df._interactions_df = pd.DataFrame(columns=['user_id', 'content_id', 'interaction_count'])
    return get_interactions_df._interactions_df

# Pre-compute and cache tag sets for faster processing
def precompute_tag_sets():
    """Pre-compute tag sets for faster overlap calculations"""
    contents_df = get_contents_df()
    print("🔄 Pre-computing tag sets for evaluation optimization...")
    tag_sets = {}
    for idx, row in contents_df.iterrows():
        content_id = row['content_id']
        generated_tags = row['generated_tags']
        
        if isinstance(generated_tags, str) and generated_tags.startswith("["):
            try:
                tags = [t.strip().lower() for t in eval(generated_tags) if isinstance(t, str)]
                tag_sets[content_id] = set(tags)
            except:
                tag_sets[content_id] = set()
        else:
            tag_sets[content_id] = set()
    
    print(f"✅ Pre-computed tag sets for {len(tag_sets)} content items")
    return tag_sets

# Initialize pre-computed tag sets lazily
def get_tag_sets():
    """Get tag sets, initializing them if not already initialized"""
    if not hasattr(get_tag_sets, '_tag_sets'):
        get_tag_sets._tag_sets = precompute_tag_sets()
    return get_tag_sets._tag_sets

# Cache ChromaDB collection
@lru_cache(maxsize=1)
def get_chroma_collection():
    """Get cached ChromaDB collection"""
    return chroma_client.get_collection(
        name="sekai_content_combined",
        embedding_function=OpenAIEmbeddingFunction()
    )

# Cache for LLM responses
LLM_CACHE = {}

def get_cache_key(prompt: str, function_name: str) -> str:
    """Generate cache key for LLM responses"""
    content = f"{function_name}:{prompt}"
    return hashlib.md5(content.encode()).hexdigest()

def cached_llm_call(prompt: str, function_name: str):
    """Cached LLM call to avoid repeated expensive API calls"""
    cache_key = get_cache_key(prompt, function_name)
    
    if cache_key in LLM_CACHE:
        print(f"🔄 Using cached response for {function_name}")
        return LLM_CACHE[cache_key]
    
    print(f"🤖 Making LLM call for {function_name}")
    response = llm.invoke(prompt)
    LLM_CACHE[cache_key] = response.content
    return response.content

# --- Define state structure with annotations ---
class EvaluationState(TypedDict):
    # Core state (read-only for parallel nodes)
    user_id: str
    recommendation_list: list
    recommendation_prompt: str
    user_tags: list
    full_interest_tags: list
    taste_profile: str
    
    # Ground truth (read-only for parallel nodes)
    ground_truth_list: list
    
    # Parallel calculation results (annotated for proper merging)
    precision: Annotated[float, "reducer"]
    tag_overlap_ratio: Annotated[float, "reducer"]
    tag_overlap_count: Annotated[int, "reducer"]
    semantic_overlap_ratio: Annotated[float, "reducer"]
    avg_semantic_similarity: Annotated[float, "reducer"]
    max_semantic_similarity: Annotated[float, "reducer"]
    
    # LLM outputs
    prompt_feedback: str
    evaluation_time: float

# --- Define LangGraph nodes ---
def get_ground_truth_list(state: EvaluationState) -> EvaluationState:
    """
    Optimized ground truth generation with caching
    """
    # ---------- sanity checks ----------
    rec_list = state.get("recommendation_list", [])
    if not rec_list:
        raise ValueError("Missing 'recommendation_list' in state.")
    if state.get("ground_truth_list"):
        return state

    # ---------- fetch user interactions ----------
    user_id_str = str(state.get("user_id")).strip()
    if not user_id_str:
        raise ValueError("State must contain 'user_id'.")

    # Use vectorized operations for faster filtering
    interactions_df = get_interactions_df()
    user_interactions = interactions_df[
        interactions_df["user_id"].astype(str).str.strip() == user_id_str
    ]

    ground_truth_ids: list[str] = (
        user_interactions.sort_values("interaction_count", ascending=False)
                         .head(10)["content_id"]
                         .astype(str)
                         .tolist()
    )

    # ---------- need to top-up? ----------
    if len(ground_truth_ids) < 10:
        needed = 10 - len(ground_truth_ids)

        # full set of tags we know about the user
        full_tags: list[str] = (
            state.get("full_interest_tags") or
            state.get("user_tags") or []
        )
        tag_set = set([t.lower().strip() for t in full_tags])

        # Optimized fuzzy overlap scorer using pre-computed tag sets
        tag_sets = get_tag_sets()
        def tag_overlap_fast(content_id: str) -> int:
            content_tags = tag_sets.get(content_id, set())
            score = 0
            for utag in tag_set:
                for ctag in content_tags:
                    if SequenceMatcher(None, utag, ctag).ratio() >= 0.75:
                        score += 1
                        break
            return score

        # Use vectorized operations for faster scoring
        contents_df = get_contents_df()
        candidate_df = contents_df.copy()
        candidate_df['score'] = candidate_df['content_id'].apply(tag_overlap_fast)
        candidate_df = candidate_df.sort_values("score", ascending=False)

        for cid in candidate_df["content_id"].astype(str):
            if cid not in ground_truth_ids:
                ground_truth_ids.append(cid)
                if len(ground_truth_ids) == 10:
                    break

    return {
        **state,
        "ground_truth_list": ground_truth_ids,
    }

def calculate_precision(state: EvaluationState) -> dict:
    # Optimized precision calculation using sets
    recommended = set(map(str, state["recommendation_list"]))
    ground_truth = set(map(str, state["ground_truth_list"]))

    if not ground_truth:
        precision = 0.0
    else:
        precision = len(recommended & ground_truth) / len(ground_truth)

    return {
        "precision": precision
    }

def calculate_tag_overlap(state: EvaluationState) -> dict:
    """
    Optimized tag overlap calculation using pre-computed tag sets
    """
    recommended_ids = set(map(str, state["recommendation_list"]))
    ground_truth_ids = set(map(str, state["ground_truth_list"]))

    # Use pre-computed tag sets for faster processing
    tag_sets = get_tag_sets()
    rec_tags = set()
    gt_tags = set()

    # Collect tags from recommended content
    for content_id in recommended_ids:
        if content_id in tag_sets:
            rec_tags.update(tag_sets[content_id])

    # Collect tags from ground truth content
    for content_id in ground_truth_ids:
        if content_id in tag_sets:
            gt_tags.update(tag_sets[content_id])

    if not gt_tags:
        return {
            "tag_overlap_ratio": 0.0,
            "tag_overlap_count": 0
        }

    overlap = rec_tags & gt_tags
    return {
        "tag_overlap_ratio": len(overlap) / len(gt_tags),
        "tag_overlap_count": len(overlap)
    }

def calculate_semantic_overlap(state: EvaluationState) -> dict:
    """
    Optimized semantic overlap with batched queries and caching
    """
    recommended_ids = set(state["recommendation_list"])
    ground_truth_ids = set(state["ground_truth_list"])

    if not recommended_ids or not ground_truth_ids:
        return {
            "semantic_overlap_ratio": 0.0,
            "avg_semantic_similarity": 0.0,
            "max_semantic_similarity": 0.0,
        }

    try:
        # Use cached collection
        collection = get_chroma_collection()

        rec_ids = [str(cid) for cid in recommended_ids]
        gt_ids = [str(cid) for cid in ground_truth_ids]

        # Handle duplicate IDs by deduplicating and tracking positions
        all_unique_ids = list(set(rec_ids + gt_ids))
        
        # Get embeddings for unique IDs only
        unique_embeddings = collection.get(ids=all_unique_ids, include=["embeddings"])["embeddings"]
        
        # Create mapping from ID to embedding index
        id_to_embedding = {id_str: emb for id_str, emb in zip(all_unique_ids, unique_embeddings)}
        
        # Extract embeddings for recommended and ground truth, handling duplicates
        rec_embs = []
        gt_embs = []
        
        for rec_id in rec_ids:
            if rec_id in id_to_embedding:
                rec_embs.append(id_to_embedding[rec_id])
        
        for gt_id in gt_ids:
            if gt_id in id_to_embedding:
                gt_embs.append(id_to_embedding[gt_id])
        
        rec_embs = np.array(rec_embs)
        gt_embs = np.array(gt_embs)

        # Optimized cosine similarity computation
        # Normalize embeddings to unit vectors
        rec_norm = rec_embs / (np.linalg.norm(rec_embs, axis=1, keepdims=True) + 1e-8)
        gt_norm = gt_embs / (np.linalg.norm(gt_embs, axis=1, keepdims=True) + 1e-8)

        # Compute cosine similarity matrix
        similarity_matrix = np.dot(rec_norm, gt_norm.T)
        similarities = similarity_matrix.flatten()

        if similarities.size > 0:
            avg_similarity = float(np.mean(similarities))
            max_similarity = float(np.max(similarities))
            threshold = 0.7
            semantic_overlap_ratio = np.sum(similarities >= threshold) / len(similarities)
        else:
            avg_similarity = 0.0
            max_similarity = 0.0
            semantic_overlap_ratio = 0.0

    except Exception as e:
        print(f"⚠️ Error in semantic overlap calculation: {e}")
        avg_similarity = 0.0
        max_similarity = 0.0
        semantic_overlap_ratio = 0.0

    return {
        "semantic_overlap_ratio": semantic_overlap_ratio,
        "avg_semantic_similarity": avg_similarity,
        "max_semantic_similarity": max_similarity,
    }

def generate_taste_profile(state: EvaluationState) -> EvaluationState:
    prompt = f"""
You are a **Taste Profile Synthesiser** working for a content recommendation system.

Your job is to improve a user's **taste profile** by learning from the relationship between:
- The latest recommendations generated
- Ground truth content they genuinely like
- Their previously stated preferences

---

🧾 **Inputs**
• Existing taste profile:  
{state.get("taste_profile", "∅")}

• Full interest tags:  
{', '.join(state.get('full_interest_tags', []))}

• Ground-truth liked items:  
{state.get("ground_truth_list")}

• Explicit/initial tags:  
{', '.join(state.get('user_tags', []))}

---

🛠 **Your Task**

- Compare the new recommendations to the ground truth and initial tags.
- Identify **recurring patterns** and **meaningful differences**.
- If the existing profile already captures something well, do not repeat it.
- Add only **clear, concrete insights**: preferences for themes (e.g., revenge, school life), character dynamics (e.g., obsessive love), or tone (e.g., lighthearted vs. intense).
- Use simple, readable language—**no abstract or poetic phrasing**.

---

Rules:
- Taste profiles summarize what the user likes: genres, moods, themes, character types, storytelling styles, or content they avoid.
- The profile should evolve and accumulate when **new, clearer** signals are found—not repeat what's already known.
- If there are new patterns that are not captured in the existing profile, add them to the profile.
- Do not remove any existing patterns that are already in the profile.
- Try to summarise the profile in a way that is easy to understand and use for the recommendation syste without missing any key patterns.
- Extrapolate the profile to include all the patterns that are not captured in the existing profile.

Think: "What do we now understand about what this user really likes or dislikes?"

Return only the new taste profile.
    """

    response_content = cached_llm_call(prompt, "generate_taste_profile")
    return {
        **state,
        "taste_profile": response_content
    }

def generate_prompt_feedback(state: EvaluationState) -> EvaluationState:
    prompt = f"""
You are an expert evaluator of prompt effectiveness in recommendation systems.

Your job is to analyze how well a given recommendation **prompt** succeeds in generating content recommendations that align with the user's **taste profile** and **ground truth preferences**.

---

🎯 **Purpose of the Prompt Being Evaluated**

The recommendation prompt is designed to guide an AI agent to select story recommendations that match a specific user's evolving taste profile. These tastes may include stylistic, thematic, emotional, or narrative preferences (e.g. "romance with psychological complexity", "action-packed school settings", etc.).

The system is expected to filter, prioritize, and synthesize content accordingly.

The agent can only perform the following actions:
- Filter by tags
- Semantic search
- Recommendation

Hence, any feedback that is not related/applicable to the above actions should be ignored.

---

🧾 **You Will Receive the Following Inputs:**

- ✍️ **Prompt used**:  
  {state.get("recommendation_prompt")}

- 📥 **AI's output recommendation list** (titles or summaries):  
  {state.get("recommendation_list")}

- 📊 **Ground truth** (ideal recommendations based on prior behavior or explicit signals):  
  {state.get("ground_truth_list")}

- 🎯 **User taste profile** (a rich summary of the user's preferences):  
  {state.get("taste_profile")}

---

🔍 **Your Evaluation Task:**

Critically assess the **recommendation prompt** above in the following dimensions:

1. **Pattern Capture**:
   - Which parts of the user's taste profile does the prompt capture well?
   - Which *specific* nuances, motifs, or themes does it fail to represent or emphasize?

2. **Effectiveness**:
   - To what extent does the generated recommendation list align with the ground truth?
   - Are there mismatches in tone, genre, theme, or depth?

3. **Prompt Quality**:
   - Is the language of the prompt clear, targeted, and directive?
   - Are there vague or under-specified areas?

4. **What's Working**:
   - Highlight the elements of the prompt that contributed to effective recommendations.
   - Mention any phrasing, structure, or emphasis that improved output quality.

5. **Suggestions for Improvement**:
   - Recommend concrete changes or strategies to better align the prompt with the user's intent and the ground truth.
   - Consider phrasing tweaks, specificity, or reordering of elements.

---

Important: The feedback should only be related to the content, not other things like functions, tools, etc.

🎯 **Format your answer as structured feedback**, including clear sections like:

- ✅ Strengths
- ⚠️ Weaknesses / Gaps
- 🔧 Suggestions for Improvement

Be concise but detailed, as if advising a prompt engineer in a production system.

---

Only return the feedback content — no preamble.
    """

    response_content = cached_llm_call(prompt, "generate_prompt_feedback")
    return {
        **state,
        "prompt_feedback": response_content
    }

# --- Build optimized evaluation graph with proper state annotations ---
evaluation_subgraph = StateGraph(EvaluationState)

# Add nodes
evaluation_subgraph.add_node("get_ground_truth_list", get_ground_truth_list)
evaluation_subgraph.add_node("calculate_precision", calculate_precision)
evaluation_subgraph.add_node("calculate_tag_overlap", calculate_tag_overlap)
evaluation_subgraph.add_node("calculate_semantic_overlap", calculate_semantic_overlap)
evaluation_subgraph.add_node("generate_taste_profile", generate_taste_profile)
evaluation_subgraph.add_node("generate_prompt_feedback", generate_prompt_feedback)

# Set entry point
evaluation_subgraph.set_entry_point("get_ground_truth_list")

# Add parallel branching: all three calculation nodes run in parallel after ground truth
evaluation_subgraph.add_edge("get_ground_truth_list", "calculate_precision")
evaluation_subgraph.add_edge("get_ground_truth_list", "calculate_tag_overlap")
evaluation_subgraph.add_edge("get_ground_truth_list", "calculate_semantic_overlap")

# Fan-in: all parallel calculations converge to taste profile generation
evaluation_subgraph.add_edge("calculate_precision", "generate_taste_profile")
evaluation_subgraph.add_edge("calculate_tag_overlap", "generate_taste_profile")
evaluation_subgraph.add_edge("calculate_semantic_overlap", "generate_taste_profile")

# Continue with sequential processing after aggregation
evaluation_subgraph.add_edge("generate_taste_profile", "generate_prompt_feedback")

# Set finish point
evaluation_subgraph.set_finish_point("generate_prompt_feedback")

compiled_evaluation_subgraph = evaluation_subgraph.compile()

# Add timing wrapper to the evaluation agent
def evaluation_agent_with_timing(state: dict) -> dict:
    start_time = time.time()
    result = compiled_evaluation_subgraph.invoke(state)
    end_time = time.time()
    evaluation_time = round(end_time - start_time, 2)
    
    return {
        **result,
        "evaluation_time": evaluation_time
    }

evaluation_agent = evaluation_agent_with_timing

# # --- Visualize the graph (optional) ---
# display(Image(evaluation_graph.get_graph().draw_mermaid_png()))

# # --- Run the graph ---
# state = evaluation_graph.invoke({
#     "recommendation_list": recommendation_list
# })

# print("🧪 Evaluation Results:")
# print("======================")
# print("Recommendation List:", state["recommendation_list"])
# print("Ground Truth List:", state["ground_truth_list"])
# print("Precision Score:", state["precision"])

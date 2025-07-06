from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command
from agents.recommendation_agent import recommendation_agent_node
from agents.evaluation_agent import evaluation_agent
from typing_extensions import TypedDict
from agents.optimiser_agent import optimiser_agent_node

# Unified state schema for all agents
class UnifiedState(TypedDict):
    # Core user data
    user_id: str
    user_tags: list[str]
    full_interest_tags: list[str]  # Optional, used for evaluation fallback
    
    # Recommendation system
    recommendation_prompt: str
    recommendation_list: list[str]
    structured_response: dict  # From recommendation agent output
    
    # Evaluation system
    ground_truth_list: list[str]
    precision: float
    prompt_feedback: str
    tag_overlap_ratio: float
    tag_overlap_count: int
    semantic_overlap_ratio: float
    avg_semantic_similarity: float
    max_semantic_similarity: float
    
    # Taste profile
    taste_profile: str
    
    # Iteration control
    iteration: int
    max_iterations: int
    remaining_steps: int
    
    # Timing states
    recommendation_time: float  # Time taken by recommendation agent
    evaluation_time: float      # Time taken by evaluation agent  
    optimiser_time: float       # Time taken by optimiser agent
    total_time: float          # Total time across all agents

def routing_function(state: UnifiedState) -> Literal["optimiser", END]:
    # print("ROUTING FUNCTION CALLED")
    # print("ITERATION: ", state.get("iteration"))
    # print("MAX ITERATIONS: ", state.get("max_iterations"))
    # print("PRECISION: ", state.get("precision"))
    
    # Calculate total time so far
    total_time = (
        state.get("recommendation_time", 0.0) + 
        state.get("evaluation_time", 0.0) + 
        state.get("optimiser_time", 0.0)
    )
    
    if state.get("precision") < 0.5 and state.get("iteration") <= state.get("max_iterations"):
        return "optimiser"
    else:
        return END

main_graph = StateGraph(UnifiedState)
main_graph.add_node("recommendation", recommendation_agent_node)
main_graph.add_node("evaluation", evaluation_agent)
main_graph.add_node("optimiser", optimiser_agent_node)
main_graph.add_edge(START, "recommendation")
main_graph.add_edge("recommendation", "evaluation")
# main_graph.add_edge("evaluation", "optimiser")
main_graph.add_edge("optimiser", "recommendation")
# main_graph.add_edge("optimiser", END)
main_graph.add_conditional_edges("evaluation", routing_function)

# ✅ Compile the graph
compiled_graph = main_graph.compile()

# # ✅ Now invoke
# state = compiled_graph.invoke({
#     "user_tags": ["romance", "space", "anime", "adventure"],
#     "recommendation_prompt": "User taste profile: romantic space adventure, emotionally driven anime."
# })

# print(state)

# --------------  👆 everything you already wrote stays as-is  -----------------

# -----------------------------------------------------------------------------#
# 🆕 ADD-ON : run the same graph for any user in users.csv with a random
#             subset of their tags.  Nothing below changes your existing logic!
# -----------------------------------------------------------------------------#
import random, os, argparse
import pandas as pd

# ---- 1.  load users table once ------------------------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
USERS_CSV = os.path.join(base_dir, "datasets", "users.csv")
users_df   = pd.read_csv(USERS_CSV)      # expects columns: user_id, user_interest_tags

def sample_user_tags(user_id: str, k: int | None = None, seed: int = 42) -> list[str]:
    """
    Return a random subset (size=k) of interest tags for the chosen user_id.
    If k is None → use max(3, len(tags)//3).
    """
    random.seed(seed)
    row = users_df.loc[users_df["user_id"].astype(str) == str(user_id)]
    if row.empty:
        raise ValueError(f"user_id {user_id} not found in ")

    all_tags = [t.strip() for t in row.iloc[0]["user_interest_tags"].split(",") if t.strip()]
    all_tags = list(dict.fromkeys(all_tags))          # dedupe while preserving order
    if not all_tags:
        raise ValueError(f"user_id {user_id} has 0 tags in the file.")

    if k is None:
        k = max(3, len(all_tags) // 3)
    k = min(k, len(all_tags))
    return random.sample(all_tags, k)

def update_total_time(state: dict) -> dict:
    """Calculate and update total execution time"""
    total_time = (
        state.get("recommendation_time", 0.0) + 
        state.get("evaluation_time", 0.0) + 
        state.get("optimiser_time", 0.0)
    )
    return {
        **state,
        "total_time": round(total_time, 2)
    }

def run_one_user(user_id: str, k: int | None = None, seed: int = 42):
    """
    Build inputs for the LangGraph using a partial tag list,
    call the compiled graph, and print results.
    """
    subset = sample_user_tags(user_id, k=k, seed=seed)

    prompt = (
        "User taste profile (partial): " + ", ".join(subset) +
        ". Recommend me great stories!"
    )

    inputs = {
        "remaining_steps": 0,
        "structured_response": None,
        "recommendation_list": [],
        "user_id": user_id,
        "user_tags": subset,
        "full_interest_tags": subset,  # Initialize with same tags
        "recommendation_prompt": prompt,
        "iteration": 0,
        "max_iterations": 3,
        "ground_truth_list": [],
        "precision": 0.0,
        "prompt_feedback": "",
        "taste_profile": "",
        # Initialize timing states
        "recommendation_time": 0.0,
        "evaluation_time": 0.0,
        "optimiser_time": 0.0,
        "total_time": 0.0
    }

    result_state = compiled_graph.stream(inputs,stream_mode="updates")

    # Stream the agents output
    for chunk in result_state: 
        print("STREAMING AGENTS OUTPUT")
        # print(chunk)
        if( "recommendation" in chunk):
            print("RECOMMENDATION: ", chunk)
            rec_chunk = chunk["recommendation"]
            print("RECOMMENDATION LIST: ", rec_chunk.get("recommendation_list"))
            print("PRECISION: ", rec_chunk.get("precision"))
            print("TASTE PROFILE: ", chunk.get("taste_profile"))
            print("ITERATION: ", chunk.get("iteration"))
            print("MAX ITERATIONS: ", chunk.get("max_iterations"))
            print("FEEDBACK: ", chunk.get("prompt_feedback"))
            print("RECOMMENDATION TIME: ", rec_chunk.get("recommendation_time", 0.0))
            print("TOTAL TIME SO FAR: ", update_total_time(rec_chunk).get("total_time", 0.0))
        elif("evaluation" in chunk):
            print("EVALUATION: ", chunk)
            eval_chunk = chunk["evaluation"]
            print("PRECISION: ", eval_chunk.get("precision"))
            print("FEEDBACK: ", eval_chunk.get("prompt_feedback"))
            print("TASTE PROFILE: ", eval_chunk.get("taste_profile"))
            print("ITERATION: ", eval_chunk.get("iteration"))
            print("MAX ITERATIONS: ", eval_chunk.get("max_iterations"))
            print("GROUND TRUTH LIST: ", chunk.get("ground_truth_list"))
            print("EVALUATION TIME: ", eval_chunk.get("evaluation_time", 0.0))
            print("TOTAL TIME SO FAR: ", update_total_time(eval_chunk).get("total_time", 0.0))
        else:
            print("OPTIMISER: ", chunk )
            opt_chunk = chunk["optimiser"]
            print("OPTIMISED PROMPT: ", opt_chunk.get("optimised_prompt"))
            print("ITERATION: ", opt_chunk.get("iteration"))
            print("MAX ITERATIONS: ", opt_chunk.get("max_iterations"))
            print("FEEDBACK: ", opt_chunk.get("prompt_feedback"))
            print("TASTE PROFILE: ", opt_chunk.get("taste_profile"))
            print("GROUND TRUTH LIST: ", chunk.get("ground_truth_list"))
            print("OPTIMISER TIME: ", opt_chunk.get("optimiser_time", 0.0))
            print("TOTAL TIME SO FAR: ", update_total_time(opt_chunk).get("total_time", 0.0))
        # print("----------------------------------------")
        # print("Reommendation Prompt: ", chunk.get("recommendation_prompt"))
        # print("Recommendation List: ", chunk.get("recommendation_list"))
        # print("Ground Truth List: ", chunk.get("ground_truth_list"))
        # print("Precision: ", chunk.get("precision"))
        # print("Taste Profile: ", chunk.get("taste_profile"))
        # print("Iteration: ", chunk.get("iteration"))
        # print("Max Iterations: ", chunk.get("max_iterations"))
        # print("Feedback: ", chunk.get("prompt_feedback"))

        # print("----------------------------------------")

    # print(f"Taste profile: {result_state.get('taste_profile')}")

    # print("\n============= RUN COMPLETE =============")
    # print(f"user_id            : {user_id}")
    # print(f"tags sent to graph : {subset}")
    # print("----------------------------------------")
    # print(f"recommendation_list: {result_state.get('recommendation_list')}")
    # print(f"precision@10       : {result_state.get('precision')}")
    # print("========================================\n")


# ---- 2.  basic CLI ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the recommendation-evaluation graph for one user.")
    parser.add_argument("--user_id", type=str, help="user_id to run (default: random user from users.csv)")
    parser.add_argument("--k",       type=int, help="how many tags to sample for the cold-start subset")
    parser.add_argument("--seed",    type=int, default=42, help="random seed")
    args = parser.parse_args()

    chosen_id = args.user_id or random.choice(users_df["user_id"].astype(str).tolist())
    run_one_user(chosen_id, k=args.k, seed=args.seed)

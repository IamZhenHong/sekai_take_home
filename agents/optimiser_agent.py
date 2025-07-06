import os
from dotenv import load_dotenv
import pandas as pd
from typing import List
from difflib import SequenceMatcher
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import time


# --- Setup ------------------------------------------------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = init_chat_model("openai:o4-mini")

# --- Output Schema ----------------------------------------------------------
class OptimiserOutput(BaseModel):
    recommendation_prompt: str


def extract_recommendation_prompt(optimiser_result):
    """
    Extracts the recommendation prompt from the optimiser result object.
    
    Args:
        optimiser_result (dict): The raw result object from the optimiser agent.

    Returns:
        str: The recommendation prompt string.
    """
    try:
        return optimiser_result['structured_response'].recommendation_prompt
    except AttributeError:
        # Handle case if 'structured_response' is a dict instead of a namedtuple or Pydantic object
        return optimiser_result['structured_response'].get('recommendation_prompt')
    except Exception as e:
        #print(f"Error extracting recommendation prompt: {e}")
        return None


# --- LangChain Agent --------------------------------------------------------
optimiser_agent = create_react_agent(
    model=init_chat_model("openai:o3-mini"),
    tools=[],
    prompt="""
You are a prompt optimiser for a recommendation system.

You are given:

- Original prompt: 
- Taste profile: 
- Prompt feedback: 

You must optimise the recommendation prompt to be more accurate, emotionally rich, and aligned with the user's evolving taste profile.

Guidelines:
-You should include the taste profile in the prompt.
- While adding another section of instructions that is based on the previous prompt's feedback.
- Do not remove any existing patterns that are already in the profile.
- Try to summarise the profile in a way that is easy to understand and use for the recommendation system without missing any key patterns.
- Extrapolate the profile to include all the patterns that are not captured in the existing profile.

Return ONLY the improved recommendation prompt, not the analysis or profile.
""",
    response_format=OptimiserOutput
)

def optimise_recommendation_prompt(state: dict) -> dict:
    if not state.get("recommendation_prompt") or not state.get("taste_profile"):
        return state
    prompt = """
You are a prompt optimiser for a recommendation system.

You are given:

- Original prompt: 
- Taste profile: 
- Prompt feedback: 

You must optimise the recommendation prompt to be more accurate, emotionally rich, and aligned with the user's evolving taste profile.

Guidelines:
-You should include the taste profile in the prompt.
- While adding another section of instructions that is based on the previous prompt's feedback.
- Do not remove any existing patterns that are already in the profile.
- Try to summarise the profile in a way that is easy to understand and use for the recommendation system without missing any key patterns.
- Extrapolate the profile to include all the patterns that are not captured in the existing profile.

Return ONLY the improved recommendation prompt, not the analysis or profile.
"""
    prompt += f"Prompt: {state['recommendation_prompt']}\nTaste Profile: {state['taste_profile']}\nPrompt Feedback: {state['prompt_feedback']}"

    start_time = time.time()

    response = llm.invoke(prompt)
    end_time = time.time()
    optimiser_time = round(end_time - start_time, 2)

    optimised_prompt = response.content

    return {
        **state,
        "recommendation_prompt": optimised_prompt,
        "optimiser_time": optimiser_time
    }


# --- LangGraph Node ---------------------------------------------------------
def optimiser_agent_node(state: dict) -> dict:
    #print("📥 Optimiser node received state keys:", list(state.keys()))

    if not state.get("recommendation_prompt") or not state.get("taste_profile"):
        #print("⚠️ Missing fields in state. Skipping.")
        return state

    print("TASTE")

    prompt = f"Prompt: {state['recommendation_prompt']}\nTaste Profile: {state['taste_profile']}\nPrompt Feedback: {state['prompt_feedback']}"

    #print("PROMPT SENT TO OPTIMISER AGENT: ", prompt)
    start_time = time.time()
    result = optimiser_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    end_time = time.time()
    optimiser_time = round(end_time - start_time, 2)

    # #print("\n✅ Optimiser agent raw result:\n", result)

    optimised_prompt = extract_recommendation_prompt(result)
    # #print("\n🔧 Extracted optimised prompt:\n", optimised_prompt)

    return {
        **state,
        "recommendation_prompt": optimised_prompt,
        "optimiser_time": optimiser_time
    }


# --- LangGraph Setup --------------------------------------------------------
# Note: This is now just for standalone testing - the main graph is in graph.py
graph = StateGraph(dict, output=dict)
graph.add_node("optimiser", optimise_recommendation_prompt) 
graph.add_edge(START, "optimiser")
graph.add_edge("optimiser", END)

optimiser_graph = graph.compile()

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from agents.graph import compiled_graph, sample_user_tags
import random
import pandas as pd
import os
import json
import asyncio
import logging

app = FastAPI(
    title="Recommendation Flow API",
    description="API for running the recommendation-evaluation-optimisation graph",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load users data
USERS_CSV = os.path.join(os.path.dirname(__file__), "datasets/users.csv")
users_df = pd.read_csv(USERS_CSV)

# Load content data
CONTENT_CSV = os.path.join(os.path.dirname(__file__), "contents_with_tags.csv")
# Check both root directory and datasets directory
if not os.path.exists(CONTENT_CSV):
    CONTENT_CSV = os.path.join(os.path.dirname(__file__), "datasets", "contents_with_tags.csv")

try:
    content_df = pd.read_csv(CONTENT_CSV)
except FileNotFoundError:
    logger.warning(f"⚠️  Warning: {CONTENT_CSV} not found. Creating empty DataFrame.")
    content_df = pd.DataFrame(columns=['content_id', 'title', 'intro', 'generated_tags'])

class RecommendationRequest(BaseModel):
    user_id: str
    k: Optional[int] = None
    seed: int = 42
    max_iterations: int = 2

class IterationStep(BaseModel):
    recommendation_list: List[str] = []
    precision: float = 0.0
    taste_profile: str = ""
    iteration: int = 0
    max_iterations: int = 0
    prompt_feedback: str = ""
    recommendation_time: float = 0.0  # Time taken by recommendation agent

class EvaluationStep(BaseModel):
    precision: float = 0.0
    prompt_feedback: str = ""
    taste_profile: str = ""
    iteration: int = 0
    max_iterations: int = 0
    ground_truth_list: List[str] = []
    evaluation_time: float = 0.0  # Time taken by evaluation agent

class OptimiserStep(BaseModel):
    optimised_prompt: str = ""
    iteration: int = 0
    max_iterations: int = 0
    prompt_feedback: str = ""
    taste_profile: str = ""
    ground_truth_list: List[str] = []
    optimiser_time: float = 0.0  # Time taken by optimiser agent

class IterationData(BaseModel):
    iteration_number: int
    recommendation: IterationStep
    evaluation: EvaluationStep
    optimiser: OptimiserStep
    total_iteration_time: float = 0.0  # Total time for this iteration

class FinalState(BaseModel):
    recommendation_list: List[str] = []
    precision: float = 0.0
    taste_profile: str = ""
    iteration: int = 0
    total_iterations: int = 0
    total_execution_time: float = 0.0  # Total time across all iterations

class RecommendationResponse(BaseModel):
    user_id: str
    user_tags: List[str]
    full_interest_tags: List[str]
    iterations: List[IterationData]
    final_state: FinalState

@app.get("/")
async def root():
    return {"message": "Recommendation Flow API is running"}

@app.get("/users")
async def get_users():
    """Get list of available user IDs"""
    user_ids = users_df["user_id"].astype(str).tolist()
    return {"user_ids": user_ids, "total_users": len(user_ids)}

def run_recommendation_with_iterations(user_id: str, user_tags: List[str], full_interest_tags: List[str], 
                                    recommendation_prompt: str, max_iterations: int) -> Dict[str, Any]:
    """Run the recommendation graph and return structured iteration data"""
    
    # Create initial state
    test_state = {
        "user_id": user_id,
        "user_tags": user_tags,
        "full_interest_tags": full_interest_tags,
        "recommendation_prompt": recommendation_prompt,
        "recommendation_list": [],
        "ground_truth_list": [],
        "precision": 0.0,
        "prompt_feedback": "",
        "taste_profile": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "remaining_steps": 0,
        "structured_response": None,
        # Initialize timing states
        "recommendation_time": 0.0,
        "evaluation_time": 0.0,
        "optimiser_time": 0.0,
        "total_time": 0.0
    }
    
    # Run the graph with streaming
    result = compiled_graph.stream(test_state, stream_mode="updates")
    
    # Group output by iterations (every 3 chunks)
    iteration_count = 0
    current_iteration_chunks = []
    api_output = {
        "user_id": test_state["user_id"],
        "user_tags": test_state["user_tags"],
        "full_interest_tags": test_state["full_interest_tags"],
        "iterations": [],
        "final_state": {}
    }
    
    for chunk in result:
        current_iteration_chunks.append(chunk)
        
        # When we have 3 chunks, we've completed one iteration
        if len(current_iteration_chunks) == 3:
            iteration_count += 1
            
            # Extract timing data
            rec_time = current_iteration_chunks[0].get("recommendation", {}).get("recommendation_time", 0.0)
            eval_time = current_iteration_chunks[1].get("evaluation", {}).get("evaluation_time", 0.0)
            opt_time = current_iteration_chunks[2].get("optimiser", {}).get("optimiser_time", 0.0)
            total_iteration_time = rec_time + eval_time + opt_time
            
            # Create API-friendly iteration data
            iteration_data = {
                "iteration_number": iteration_count,
                "recommendation": {
                    "recommendation_prompt": current_iteration_chunks[0].get("recommendation", {}).get("recommendation_prompt", ""),
                    "recommendation_list": current_iteration_chunks[0].get("recommendation", {}).get("recommendation_list", []),
                    "precision": current_iteration_chunks[0].get("recommendation", {}).get("precision", 0.0),
                    "taste_profile": current_iteration_chunks[0].get("recommendation", {}).get("taste_profile", ""),
                    "iteration": current_iteration_chunks[0].get("recommendation", {}).get("iteration", 0),
                    "max_iterations": current_iteration_chunks[0].get("recommendation", {}).get("max_iterations", 0),
                    "prompt_feedback": current_iteration_chunks[0].get("recommendation", {}).get("prompt_feedback", ""),
                    "recommendation_time": rec_time
                },
                "evaluation": {
                    "tag_overlap_ratio": current_iteration_chunks[1].get("evaluation", {}).get("tag_overlap_ratio", 0.0),
                    "tag_overlap_count": current_iteration_chunks[1].get("evaluation", {}).get("tag_overlap_count", 0),
                    "semantic_overlap_ratio": current_iteration_chunks[1].get("evaluation", {}).get("semantic_overlap_ratio", 0.0),
                    "avg_semantic_similarity": current_iteration_chunks[1].get("evaluation", {}).get("avg_semantic_similarity", 0.0),
                    "max_semantic_similarity": current_iteration_chunks[1].get("evaluation", {}).get("max_semantic_similarity", 0.0),
                    "precision": current_iteration_chunks[1].get("evaluation", {}).get("precision", 0.0),
                    "prompt_feedback": current_iteration_chunks[1].get("evaluation", {}).get("prompt_feedback", ""),
                    "taste_profile": current_iteration_chunks[1].get("evaluation", {}).get("taste_profile", ""),
                    "iteration": current_iteration_chunks[1].get("evaluation", {}).get("iteration", 0),
                    "max_iterations": current_iteration_chunks[1].get("evaluation", {}).get("max_iterations", 0),
                    "ground_truth_list": current_iteration_chunks[1].get("evaluation", {}).get("ground_truth_list", []),
                    "evaluation_time": eval_time
                },
                "optimiser": {
                    "optimised_prompt": current_iteration_chunks[2].get("optimiser", {}).get("recommendation_prompt", ""),
                    "iteration": current_iteration_chunks[2].get("optimiser", {}).get("iteration", 0),
                    "max_iterations": current_iteration_chunks[2].get("optimiser", {}).get("max_iterations", 0),
                    "prompt_feedback": current_iteration_chunks[2].get("optimiser", {}).get("prompt_feedback", ""),
                    "taste_profile": current_iteration_chunks[2].get("optimiser", {}).get("taste_profile", ""),
                    "ground_truth_list": current_iteration_chunks[2].get("optimiser", {}).get("ground_truth_list", []),
                    "optimiser_time": opt_time
                },
                "total_iteration_time": round(total_iteration_time, 2)
            }
            
            api_output["iterations"].append(iteration_data)
            current_iteration_chunks = []
    
    # Calculate total execution time
    total_execution_time = sum(
        iteration["total_iteration_time"] for iteration in api_output["iterations"]
    )
    
    # Get final state
    api_output["final_state"] = {
        "recommendation_list": result.get('recommendation_list', []),
        "precision": result.get('precision', 0.0),
        "taste_profile": result.get('taste_profile', ''),
        "iteration": result.get('iteration', 0),
        "total_iterations": len(api_output["iterations"]),
        "total_execution_time": round(total_execution_time, 2)
    }
    
    return api_output

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get recommendations for a user with detailed iteration data
    
    - user_id: Required - the user ID to get recommendations for
    - k: Optional - number of tags to sample (default: max(3, len(tags)//3))
    - seed: Optional - random seed for reproducible sampling
    - max_iterations: Optional - maximum iterations to run
    """
    try:
        # Get user data from CSV
        user_id = request.user_id
        row = users_df.loc[users_df["user_id"].astype(str) == str(user_id)]
        
        if row.empty:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found in users.csv")
        
        # Get full interest tags from the user
        full_tags = [t.strip() for t in row.iloc[0]["user_interest_tags"].split(",") if t.strip()]
        
        if not full_tags:
            raise HTTPException(status_code=400, detail=f"User {user_id} has no interest tags")
        
        # Sample a subset of tags
        if request.k is None:
            k = max(3, len(full_tags) // 3)
        else:
            k = min(request.k, len(full_tags))
        
        # Set random seed for reproducible sampling
        random.seed(request.seed)
        subset = random.sample(full_tags, k)

        # Create recommendation prompt
        prompt = (
            "User taste profile (partial): " + ", ".join(subset) +
            ". Recommend me great stories!"
        )

        # Run the graph with iterations
        result = run_recommendation_with_iterations(
            user_id=user_id,
            user_tags=subset,
            full_interest_tags=full_tags,
            recommendation_prompt=prompt,
            max_iterations=request.max_iterations
        )

        return RecommendationResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.get("/content/batch")
async def get_content_batch(content_ids: str):
    """Get multiple content items by IDs (comma-separated)"""
    try:
        logger.debug(f"Received content_ids: {content_ids}")
        logger.debug(f"Content CSV shape: {content_df.shape}")
        logger.debug(f"Content CSV columns: {content_df.columns.tolist()}")
        logger.debug(f"Sample content_ids in CSV: {content_df['content_id'].head().tolist()}")
        
        ids = [id.strip() for id in content_ids.split(",")]
        logger.debug(f"Looking for IDs: {ids}")
        
        results = []
        
        for content_id in ids:
            logger.debug(f"Searching for content_id: {content_id}")
            # Try different ways to match
            row = content_df.loc[content_df["content_id"].astype(str) == str(content_id)]
            if row.empty:
                # Try as integer if it's a number
                try:
                    content_id_int = int(content_id)
                    row = content_df.loc[content_df["content_id"] == content_id_int]
                except ValueError:
                    pass
            
            if not row.empty:
                result = {
                    "content_id": content_id,
                    "title": row.iloc[0].get("title", ""),
                    "intro": row.iloc[0].get("intro", ""),
                    "character_list": row.iloc[0].get("character_list", ""),
                    "initial_record": row.iloc[0].get("initial_record", ""),
                    "generated_tags": row.iloc[0].get("generated_tags", "")
                }
                results.append(result)
                logger.debug(f"Found content: {result['title']}")
            else:
                logger.debug(f"Content ID {content_id} not found")
        
        return {"content": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/content/{content_id}")
async def get_content(content_id: str):
    """Get content details by ID"""
    try:
        row = content_df.loc[content_df["content_id"].astype(str) == str(content_id)]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"Content {content_id} not found")
        
        return {
            "content_id": content_id,
            "title": row.iloc[0].get("title", ""),
            "intro": row.iloc[0].get("intro", ""),
            "character_list": row.iloc[0].get("character_list", ""),
            "initial_record": row.iloc[0].get("initial_record", ""),
            "generated_tags": row.iloc[0].get("generated_tags", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    """Simple test WebSocket endpoint"""
    logger.debug("Test WebSocket connection attempt")
    logger.debug(f"Headers: {dict(websocket.headers)}")
    await websocket.accept()
    await websocket.send_text("Test connection successful!")
    await websocket.close()

@app.websocket("/ws/recommend")
async def websocket_recommend(websocket: WebSocket):
    # Debug: Print all headers
    logger.debug(f"WebSocket headers: {dict(websocket.headers)}")
    
    # Check origin for CORS
    origin = websocket.headers.get("origin")
    logger.debug(f"Origin: {origin}")
    
    # For development, allow all connections
    # In production, you'd want to check the origin properly
    await websocket.accept()
    logger.debug("WebSocket accepted!")
    
    try:
        logger.debug("Waiting for message...")
        data = await websocket.receive_text()
        logger.debug(f"Received data: {data}")
        request = json.loads(data)
        
        # Validate required user_id
        user_id = request.get("user_id")
        if not user_id:
            logger.error("No user_id provided")
            await websocket.send_text(json.dumps({"error": "user_id is required"}))
            await websocket.close()
            return
        
        logger.debug(f"Processing request for user_id: {user_id}")
        
        # Get user data from CSV
        row = users_df.loc[users_df["user_id"].astype(str) == str(user_id)]
        if row.empty:
            logger.error(f"User {user_id} not found")
            await websocket.send_text(json.dumps({"error": f"User {user_id} not found in users.csv"}))
            await websocket.close()
            return
        
        logger.debug("User found, processing...")
        
        # Get full interest tags from the user
        full_tags = [t.strip() for t in row.iloc[0]["user_interest_tags"].split(",") if t.strip()]
        if not full_tags:
            logger.error(f"User {user_id} has no tags")
            await websocket.send_text(json.dumps({"error": f"User {user_id} has no interest tags"}))
            await websocket.close()
            return
        
        # Sample a subset of tags
        k = request.get("k")
        if k is None:
            k = max(3, len(full_tags) // 3)
        else:
            k = min(k, len(full_tags))
        
        # Set random seed for reproducible sampling
        seed = request.get("seed", 42)
        random.seed(seed)
        subset = random.sample(full_tags, k)

        # Create recommendation prompt
        prompt = (
            "User taste profile (partial): " + ", ".join(subset) +
            ". Recommend me great stories!"
        )

        max_iterations = request.get("max_iterations", 2)

        # Create initial state
        test_state = {
            "user_id": user_id,
            "user_tags": subset,
            "full_interest_tags": full_tags,
            "recommendation_prompt": prompt,
            "recommendation_list": [],
            "ground_truth_list": [],
            "precision": 0.0,
            "prompt_feedback": "",
            "taste_profile": "",
            "iteration": 0,
            "max_iterations": max_iterations,
            "remaining_steps": 0,
            "structured_response": None,
            # Initialize timing states
            "recommendation_time": 0.0,
            "evaluation_time": 0.0,
            "optimiser_time": 0.0,
            "total_time": 0.0
        }
        
        logger.debug("Starting recommendation process...")
        
        # Send initial setup message
        await websocket.send_text(json.dumps({
            "type": "setup",
            "user_id": user_id,
            "user_tags": subset,
            "full_interest_tags": full_tags,
            "max_iterations": max_iterations
        }))
        
        # Run the graph with streaming
        result = compiled_graph.stream(test_state, stream_mode="updates")
        
        # Group output by iterations (every 3 chunks) - STREAM IN REAL-TIME
        iteration_count = 0
        current_iteration_chunks = []
        total_execution_time = 0.0
        
        for chunk in result:
            current_iteration_chunks.append(chunk)
            
            # Send progress message for each step
            step_name = ["recommendation", "evaluation", "optimiser"][len(current_iteration_chunks) - 1]
            await websocket.send_text(json.dumps({
                "type": "progress",
                "step": step_name,
                "iteration": iteration_count + 1,
                "step_number": len(current_iteration_chunks)
            }))
            
            # When we have 3 chunks, we've completed one iteration - SEND IMMEDIATELY
            if len(current_iteration_chunks) == 3:
                iteration_count += 1
                
                # Extract timing data
                rec_time = current_iteration_chunks[0].get("recommendation", {}).get("recommendation_time", 0.0)
                eval_time = current_iteration_chunks[1].get("evaluation", {}).get("evaluation_time", 0.0)
                opt_time = current_iteration_chunks[2].get("optimiser", {}).get("optimiser_time", 0.0)
                total_iteration_time = rec_time + eval_time + opt_time
                total_execution_time += total_iteration_time
                
                # Create API-friendly iteration data
                iteration_data = {
                    "type": "iteration",
                    "iteration_number": iteration_count,
                    "recommendation": {
                        "recommendation_prompt": current_iteration_chunks[0].get("recommendation", {}).get("recommendation_prompt", ""),
                        "recommendation_list": current_iteration_chunks[0].get("recommendation", {}).get("recommendation_list", []),
                        "precision": current_iteration_chunks[0].get("recommendation", {}).get("precision", 0.0),
                        "taste_profile": current_iteration_chunks[0].get("recommendation", {}).get("taste_profile", ""),
                        "iteration": current_iteration_chunks[0].get("recommendation", {}).get("iteration", 0),
                        "max_iterations": current_iteration_chunks[0].get("recommendation", {}).get("max_iterations", 0),
                        "prompt_feedback": current_iteration_chunks[0].get("recommendation", {}).get("prompt_feedback", ""),
                        "recommendation_time": rec_time
                    },
                    "evaluation": {
                        "tag_overlap_ratio": current_iteration_chunks[1].get("evaluation", {}).get("tag_overlap_ratio", 0.0),
                        "tag_overlap_count": current_iteration_chunks[1].get("evaluation", {}).get("tag_overlap_count", 0),
                        "semantic_overlap_ratio": current_iteration_chunks[1].get("evaluation", {}).get("semantic_overlap_ratio", 0.0),
                        "avg_semantic_similarity": current_iteration_chunks[1].get("evaluation", {}).get("avg_semantic_similarity", 0.0),
                        "max_semantic_similarity": current_iteration_chunks[1].get("evaluation", {}).get("max_semantic_similarity", 0.0),
                        "precision": current_iteration_chunks[1].get("evaluation", {}).get("precision", 0.0),
                        "prompt_feedback": current_iteration_chunks[1].get("evaluation", {}).get("prompt_feedback", ""),
                        "taste_profile": current_iteration_chunks[1].get("evaluation", {}).get("taste_profile", ""),
                        "iteration": current_iteration_chunks[1].get("evaluation", {}).get("iteration", 0),
                        "max_iterations": current_iteration_chunks[1].get("evaluation", {}).get("max_iterations", 0),
                        "ground_truth_list": current_iteration_chunks[1].get("evaluation", {}).get("ground_truth_list", []),
                        "evaluation_time": eval_time
                    },
                    "optimiser": {
                        "optimised_prompt": current_iteration_chunks[2].get("optimiser", {}).get("recommendation_prompt", ""),
                        "iteration": current_iteration_chunks[2].get("optimiser", {}).get("iteration", 0),
                        "max_iterations": current_iteration_chunks[2].get("optimiser", {}).get("max_iterations", 0),
                        "prompt_feedback": current_iteration_chunks[2].get("optimiser", {}).get("prompt_feedback", ""),
                        "taste_profile": current_iteration_chunks[2].get("optimiser", {}).get("taste_profile", ""),
                        "ground_truth_list": current_iteration_chunks[2].get("optimiser", {}).get("ground_truth_list", []),
                        "optimiser_time": opt_time
                    },
                    "total_iteration_time": round(total_iteration_time, 2)
                }
                
                # SEND EACH ITERATION IMMEDIATELY AS IT COMPLETES
                await websocket.send_text(json.dumps(iteration_data))
                
                # Small delay to make streaming more visible (optional)
                await asyncio.sleep(0.1)
                
                current_iteration_chunks = []
        
        # Send final state
        final_state = {
            "type": "final",
            "recommendation_list": result.get('recommendation_list', []),
            "precision": result.get('precision', 0.0),
            "taste_profile": result.get('taste_profile', ''),
            "iteration": result.get('iteration', 0),
            "total_iterations": iteration_count,
            "total_execution_time": round(total_execution_time, 2)
        }
        await websocket.send_text(json.dumps(final_state))
        logger.debug("WebSocket process completed successfully")
        await websocket.close()
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_text(json.dumps({"type": "error", "error": str(e)}))
        await websocket.close()

if __name__ == "__main__":
    # Configure logging
    VERBOSE = os.getenv("VERBOSE", "0") == "1"
    logging.basicConfig(
        level=logging.DEBUG if VERBOSE else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)
    uvicorn.run(app, host="0.0.0.0", port=8000) 
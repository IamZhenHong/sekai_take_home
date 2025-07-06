#!/usr/bin/env python3
"""
Test script to verify timing functionality in main.py API endpoints.
"""

import requests
import json
import time

def test_http_endpoint():
    """Test the HTTP /recommend endpoint with timing"""
    print("🧪 Testing HTTP endpoint timing...")
    print("=" * 50)
    
    url = "http://localhost:8000/recommend"
    
    payload = {
        "user_id": "606827",
        "k": 5,
        "seed": 42,
        "max_iterations": 2
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print("✅ HTTP endpoint test successful!")
            print(f"Total API response time: {end_time - start_time:.2f}s")
            
            # Print timing information
            print("\n📊 Timing Information:")
            print("-" * 30)
            
            for i, iteration in enumerate(data.get("iterations", []), 1):
                print(f"\nIteration {i}:")
                print(f"  Recommendation time: {iteration['recommendation']['recommendation_time']}s")
                print(f"  Evaluation time: {iteration['evaluation']['evaluation_time']}s")
                print(f"  Optimiser time: {iteration['optimiser']['optimiser_time']}s")
                print(f"  Total iteration time: {iteration['total_iteration_time']}s")
            
            print(f"\nTotal execution time: {data['final_state']['total_execution_time']}s")
            
        else:
            print(f"❌ HTTP endpoint test failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ HTTP endpoint test failed: {e}")

def test_websocket_endpoint():
    """Test the WebSocket endpoint with timing"""
    print("\n🧪 Testing WebSocket endpoint timing...")
    print("=" * 50)
    
    import websockets
    import asyncio
    
    async def test_websocket():
        uri = "ws://localhost:8000/ws/recommend"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Send request
                request = {
                    "user_id": "606827",
                    "k": 5,
                    "seed": 42,
                    "max_iterations": 2
                }
                
                await websocket.send(json.dumps(request))
                
                # Receive messages
                iteration_count = 0
                total_execution_time = 0.0
                
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        if data.get("type") == "setup":
                            print("✅ WebSocket connection established")
                            
                        elif data.get("type") == "progress":
                            print(f"📈 Progress: {data['step']} step {data['step_number']}/3")
                            
                        elif data.get("type") == "iteration":
                            iteration_count += 1
                            iteration_time = data.get("total_iteration_time", 0.0)
                            total_execution_time += iteration_time
                            
                            print(f"\n🔄 Iteration {iteration_count}:")
                            print(f"  Recommendation: {data['recommendation']['recommendation_time']}s")
                            print(f"  Evaluation: {data['evaluation']['evaluation_time']}s")
                            print(f"  Optimiser: {data['optimiser']['optimiser_time']}s")
                            print(f"  Total iteration: {iteration_time}s")
                            
                        elif data.get("type") == "final":
                            print(f"\n✅ Final state:")
                            print(f"  Total iterations: {data['total_iterations']}")
                            print(f"  Total execution time: {data['total_execution_time']}s")
                            break
                            
                    except websockets.exceptions.ConnectionClosed:
                        print("❌ WebSocket connection closed")
                        break
                        
        except Exception as e:
            print(f"❌ WebSocket test failed: {e}")
    
    # Run the async test
    asyncio.run(test_websocket())

if __name__ == "__main__":
    print("🚀 Testing main.py timing functionality")
    print("Make sure the FastAPI server is running on localhost:8000")
    print("=" * 60)
    
    # Test HTTP endpoint
    test_http_endpoint()
    
    # Test WebSocket endpoint
    test_websocket_endpoint()
    
    print("\n✅ All timing tests completed!") 
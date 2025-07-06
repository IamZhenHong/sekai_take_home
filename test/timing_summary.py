#!/usr/bin/env python3
"""
Timing summary and analysis for the agent system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph import run_one_user
import pandas as pd
from typing import Dict, List
import time

class TimingTracker:
    """Track timing statistics across multiple runs"""
    
    def __init__(self):
        self.runs = []
        self.current_run = {}
    
    def start_run(self, user_id: str):
        """Start tracking a new run"""
        self.current_run = {
            "user_id": user_id,
            "start_time": time.time(),
            "agent_times": {}
        }
    
    def record_agent_time(self, agent_name: str, execution_time: float):
        """Record execution time for an agent"""
        if "agent_times" not in self.current_run:
            self.current_run["agent_times"] = {}
        self.current_run["agent_times"][agent_name] = execution_time
    
    def end_run(self):
        """End the current run and add to statistics"""
        if self.current_run:
            self.current_run["total_time"] = time.time() - self.current_run["start_time"]
            self.runs.append(self.current_run.copy())
            self.current_run = {}
    
    def get_statistics(self) -> Dict:
        """Get timing statistics across all runs"""
        if not self.runs:
            return {}
        
        stats = {
            "total_runs": len(self.runs),
            "average_total_time": 0.0,
            "agent_averages": {},
            "slowest_agent": None,
            "fastest_agent": None
        }
        
        # Calculate averages
        total_times = [run["total_time"] for run in self.runs]
        stats["average_total_time"] = sum(total_times) / len(total_times)
        
        # Calculate agent averages
        agent_times = {}
        for run in self.runs:
            for agent, time_taken in run["agent_times"].items():
                if agent not in agent_times:
                    agent_times[agent] = []
                agent_times[agent].append(time_taken)
        
        for agent, times in agent_times.items():
            stats["agent_averages"][agent] = sum(times) / len(times)
        
        # Find slowest and fastest agents
        if stats["agent_averages"]:
            stats["slowest_agent"] = max(stats["agent_averages"], key=stats["agent_averages"].get)
            stats["fastest_agent"] = min(stats["agent_averages"], key=stats["agent_averages"].get)
        
        return stats
    
    def print_summary(self):
        """Print a summary of timing statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("📊 TIMING SUMMARY")
        print("="*60)
        print(f"Total runs analyzed: {stats.get('total_runs', 0)}")
        print(f"Average total execution time: {stats.get('average_total_time', 0):.2f}s")
        
        print("\nAgent Performance:")
        print("-" * 40)
        for agent, avg_time in stats.get("agent_averages", {}).items():
            print(f"{agent:20s}: {avg_time:.2f}s")
        
        if stats.get("slowest_agent"):
            print(f"\n🐌 Slowest agent: {stats['slowest_agent']}")
        if stats.get("fastest_agent"):
            print(f"⚡ Fastest agent: {stats['fastest_agent']}")
        
        print("="*60)

def run_with_timing_tracking(user_id: str, k: int = 5, seed: int = 42):
    """Run the system with timing tracking"""
    tracker = TimingTracker()
    tracker.start_run(user_id)
    
    # This would need to be modified to capture timing from the streaming output
    # For now, this is a placeholder for the timing tracking functionality
    print(f"Running timing analysis for user {user_id}...")
    
    # You would need to modify the run_one_user function to accept a tracker
    # and record timing information as it streams
    
    tracker.end_run()
    return tracker

if __name__ == "__main__":
    # Example usage
    test_user_id = "606827"
    tracker = run_with_timing_tracking(test_user_id)
    tracker.print_summary() 
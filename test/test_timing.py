#!/usr/bin/env python3
"""
Test script to verify timing functionality in the agent system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph import run_one_user
import pandas as pd

def test_timing():
    """Test the timing functionality with a sample user"""
    print("🧪 Testing timing functionality...")
    print("=" * 50)
    
    # Test with a specific user ID
    test_user_id = "606827"  # From the users.csv file
    
    try:
        run_one_user(test_user_id, k=5, seed=42)
        print("\n✅ Timing test completed successfully!")
    except Exception as e:
        print(f"\n❌ Timing test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_timing() 
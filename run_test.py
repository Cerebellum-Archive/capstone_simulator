#!/usr/bin/env python3
"""
Test script to run multi_target_simulator and capture output.
"""

import os
import sys
import subprocess
import traceback

def run_simulation_test():
    """Run the multi_target_simulator and capture output."""
    
    print("🧪 Running multi_target_simulator test...")
    print("📁 Current directory:", os.getcwd())
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    try:
        # Import the module
        print("📦 Importing multi_target_simulator...")
        from multi_target_simulator import main
        
        # Run the main function
        print("🚀 Running main function...")
        main()
        
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        
        # Try to import plotting_utils separately
        print("\n🔍 Testing plotting_utils import...")
        try:
            from plotting_utils import create_tear_sheet
            print("✅ plotting_utils import successful")
        except Exception as plot_e:
            print(f"❌ plotting_utils import failed: {plot_e}")
            traceback.print_exc()

if __name__ == "__main__":
    run_simulation_test() 
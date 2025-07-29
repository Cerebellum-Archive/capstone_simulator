#!/usr/bin/env python3
"""
Simple test to verify matplotlib file saving works.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def test_simple_plot():
    """Test that matplotlib can save files to reports directory."""
    
    print("🧪 Testing simple matplotlib save...")
    
    # Create a simple plot
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, label='sin(x)')
    ax.set_title('Simple Test Plot')
    ax.legend()
    
    # Get current directory and create reports path
    current_dir = os.getcwd()
    reports_dir = os.path.join(current_dir, 'reports')
    
    print(f"📁 Current directory: {current_dir}")
    print(f"📁 Reports directory: {reports_dir}")
    print(f"📁 Reports directory exists: {os.path.exists(reports_dir)}")
    
    # Create reports directory if it doesn't exist
    os.makedirs(reports_dir, exist_ok=True)
    print(f"📁 Reports directory created/exists: {os.path.exists(reports_dir)}")
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save test files
    pdf_filename = os.path.join(reports_dir, f'test_plot_{timestamp}.pdf')
    png_filename = os.path.join(reports_dir, f'test_plot_{timestamp}.png')
    
    print(f"📄 PDF filename: {pdf_filename}")
    print(f"🖼️  PNG filename: {png_filename}")
    
    try:
        # Save PDF
        print("💾 Saving PDF...")
        plt.savefig(pdf_filename, dpi=300, bbox_inches='tight', format='pdf')
        print("✅ PDF saved successfully")
        
        # Save PNG
        print("💾 Saving PNG...")
        plt.savefig(png_filename, dpi=300, bbox_inches='tight', format='png')
        print("✅ PNG saved successfully")
        
        # Check if files were created
        if os.path.exists(pdf_filename):
            print(f"✅ PDF file exists: {pdf_filename}")
            print(f"📊 PDF file size: {os.path.getsize(pdf_filename)} bytes")
        else:
            print(f"❌ PDF file was not created: {pdf_filename}")
            
        if os.path.exists(png_filename):
            print(f"✅ PNG file exists: {png_filename}")
            print(f"📊 PNG file size: {os.path.getsize(png_filename)} bytes")
        else:
            print(f"❌ PNG file was not created: {png_filename}")
            
    except Exception as e:
        print(f"❌ Error saving files: {e}")
        import traceback
        traceback.print_exc()
    
    plt.close()
    
    # List files in reports directory
    print("\n📁 Files in reports directory:")
    try:
        files = os.listdir(reports_dir)
        for file in files:
            file_path = os.path.join(reports_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  📄 {file} ({size} bytes)")
    except Exception as e:
        print(f"❌ Error listing files: {e}")

if __name__ == "__main__":
    test_simple_plot() 
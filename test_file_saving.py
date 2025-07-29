#!/usr/bin/env python3
"""
Test script to verify file saving works correctly.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def test_file_saving():
    """Test that files can be saved to the reports directory."""
    
    # Create a simple plot
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_title('Test Plot')
    
    # Get current directory and create reports path
    current_dir = os.getcwd()
    reports_dir = os.path.join(current_dir, 'reports')
    
    # Create reports directory if it doesn't exist
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save test files
    pdf_filename = os.path.join(reports_dir, f'test_tear_sheet_{timestamp}.pdf')
    png_filename = os.path.join(reports_dir, f'test_tear_sheet_{timestamp}.png')
    
    print(f"Current working directory: {current_dir}")
    print(f"Reports directory: {reports_dir}")
    print(f"Reports directory exists: {os.path.exists(reports_dir)}")
    print(f"Saving PDF to: {os.path.abspath(pdf_filename)}")
    print(f"Saving PNG to: {os.path.abspath(png_filename)}")
    
    # Save the files
    plt.savefig(pdf_filename, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(png_filename, dpi=300, bbox_inches='tight', format='png')
    
    print(f"✅ Test files saved:")
    print(f"   PDF: {pdf_filename}")
    print(f"   PNG: {png_filename}")
    
    # Check if files were actually created
    print(f"PDF file exists: {os.path.exists(pdf_filename)}")
    print(f"PNG file exists: {os.path.exists(png_filename)}")
    
    plt.close()
    
    return pdf_filename, png_filename

if __name__ == "__main__":
    test_file_saving() 
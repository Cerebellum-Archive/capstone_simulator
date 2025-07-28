#!/usr/bin/env python3
"""
Simple PDF Generator using HTML rendering

This script creates a PDF using Python's built-in capabilities and the HTML tutorial.
It uses weasyprint which is a more modern and maintained alternative to wkhtmltopdf.

Usage:
    python scripts/generate_pdf_simple.py
"""

import os
import subprocess
import sys

def check_weasyprint():
    """Check if weasyprint is available"""
    try:
        import weasyprint
        return True
    except ImportError:
        return False

def install_weasyprint():
    """Install weasyprint"""
    print("📦 Installing weasyprint for PDF generation...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'weasyprint'])
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install weasyprint automatically")
        print("Please try manually: pip install weasyprint")
        return False

def generate_pdf_from_html():
    """Generate PDF from existing HTML tutorial"""
    
    # Ensure weasyprint is available
    if not check_weasyprint():
        print("📋 weasyprint not found. Installing...")
        if not install_weasyprint():
            return False
        
        # Import after installation
        try:
            import weasyprint
        except ImportError:
            print("❌ Still couldn't import weasyprint after installation")
            return False
    else:
        import weasyprint
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_file = os.path.join(base_dir, 'docs', 'Blue_Water_Macro_Tutorial.html')
    pdf_file = os.path.join(base_dir, 'docs', 'Blue_Water_Macro_Tutorial.pdf')
    
    # Check if HTML exists
    if not os.path.exists(html_file):
        print("📄 HTML tutorial not found. Generating...")
        # Generate HTML first
        try:
            subprocess.check_call([sys.executable, 'scripts/generate_html_tutorial.py'])
        except subprocess.CalledProcessError:
            print("❌ Failed to generate HTML tutorial")
            return False
    
    try:
        print("🔄 Converting HTML to PDF...")
        
        # Generate PDF with professional settings
        html_doc = weasyprint.HTML(filename=html_file)
        pdf_doc = html_doc.write_pdf()
        
        # Write to file
        with open(pdf_file, 'wb') as f:
            f.write(pdf_doc)
        
        # Get file size
        file_size = os.path.getsize(pdf_file) / (1024 * 1024)  # MB
        
        print(f"✅ PDF generated successfully!")
        print(f"📁 Location: {pdf_file}")
        print(f"📄 File size: {file_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return False

def generate_simple_pdf():
    """Fallback method using basic HTML to PDF"""
    try:
        import pdfkit
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        html_file = os.path.join(base_dir, 'docs', 'Blue_Water_Macro_Tutorial.html')
        pdf_file = os.path.join(base_dir, 'docs', 'Blue_Water_Macro_Tutorial.pdf')
        
        if not os.path.exists(html_file):
            print("📄 HTML tutorial not found. Please run: python scripts/generate_html_tutorial.py")
            return False
            
        # Simple options that don't require wkhtmltopdf
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in', 
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None
        }
        
        print("🔄 Attempting PDF generation with pdfkit...")
        pdfkit.from_file(html_file, pdf_file, options=options)
        
        file_size = os.path.getsize(pdf_file) / (1024 * 1024)
        print(f"✅ PDF generated successfully!")
        print(f"📁 Location: {pdf_file}")
        print(f"📄 File size: {file_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ pdfkit method failed: {e}")
        return False

if __name__ == "__main__":
    print("📖 Blue Water Macro PDF Generator")
    print("=" * 40)
    
    # Try modern weasyprint first
    success = generate_pdf_from_html()
    
    # Fallback to pdfkit if available
    if not success:
        print("\n🔄 Trying alternative method...")
        success = generate_simple_pdf()
    
    if success:
        print("\n🎉 PDF tutorial ready!")
        print("\n📚 Use cases:")
        print("   • Offline study and reference")
        print("   • Student distribution") 
        print("   • Academic portfolio material")
        print("   • Professional development")
        print("\n💡 Tip: The PDF includes:")
        print("   • Complete framework guide")
        print("   • Code examples and templates")
        print("   • Career development guidance")
        print("   • Mathematical foundations")
    else:
        print("\n❌ PDF generation failed.")
        print("\n🌐 Alternative: Use the HTML version!")
        print("   1. Open docs/Blue_Water_Macro_Tutorial.html in browser")
        print("   2. Press Ctrl+P (Cmd+P on Mac)")
        print("   3. Choose 'Save as PDF'")
        print("   4. Result: Professional PDF tutorial")
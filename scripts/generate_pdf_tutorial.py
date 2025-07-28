#!/usr/bin/env python3
"""
PDF Tutorial Generator for Blue Water Macro Quantitative Trading Framework

This script converts the comprehensive markdown tutorial into a PDF
suitable for offline reading and distribution to students.

Usage:
    python scripts/generate_pdf_tutorial.py

Requirements:
    pip install markdown pdfkit beautifulsoup4 pygments

Note: This script requires wkhtmltopdf to be installed on your system.
On macOS: brew install wkhtmltopdf
On Ubuntu: sudo apt-get install wkhtmltopdf
On Windows: Download from https://wkhtmltopdf.org/downloads.html
"""

import os
import sys
import markdown
from datetime import datetime
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import pdfkit
        import bs4
        import pygments
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install markdown pdfkit beautifulsoup4 pygments")
        return False

def check_wkhtmltopdf():
    """Check if wkhtmltopdf is installed"""
    try:
        subprocess.run(['wkhtmltopdf', '--version'], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("wkhtmltopdf not found. Please install:")
        print("macOS: brew install wkhtmltopdf")
        print("Ubuntu: sudo apt-get install wkhtmltopdf")
        print("Windows: Download from https://wkhtmltopdf.org/downloads.html")
        return False

def create_html_template():
    """Create professional HTML template with Blue Water Macro styling"""
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Blue Water Macro Quantitative Trading Tutorial</title>
    <style>
        @page {
            size: A4;
            margin: 1in;
            @top-center {
                content: "Blue Water Macro Quantitative Trading Tutorial";
                font-family: 'Arial', sans-serif;
                font-size: 10pt;
                color: #666;
            }
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-family: 'Arial', sans-serif;
                font-size: 10pt;
                color: #666;
            }
        }
        
        body {
            font-family: 'Georgia', serif;
            line-height: 1.6;
            color: #333;
            max-width: none;
            margin: 0;
            padding: 0;
        }
        
        h1 {
            color: #1e4d72;
            border-bottom: 3px solid #1e4d72;
            padding-bottom: 10px;
            page-break-before: always;
            font-size: 24pt;
        }
        
        h1:first-of-type {
            page-break-before: avoid;
        }
        
        h2 {
            color: #2e5984;
            border-bottom: 2px solid #2e5984;
            padding-bottom: 5px;
            margin-top: 30px;
            font-size: 18pt;
        }
        
        h3 {
            color: #4a7ba7;
            margin-top: 25px;
            font-size: 14pt;
        }
        
        h4 {
            color: #5d8bb5;
            margin-top: 20px;
            font-size: 12pt;
        }
        
        pre {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            overflow-x: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 9pt;
            line-height: 1.4;
        }
        
        code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 9pt;
            color: #d73a49;
        }
        
        pre code {
            background-color: transparent;
            color: #333;
            padding: 0;
        }
        
        blockquote {
            border-left: 4px solid #1e4d72;
            padding-left: 20px;
            margin-left: 0;
            font-style: italic;
            color: #666;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        
        th {
            background-color: #1e4d72;
            color: white;
            font-weight: bold;
        }
        
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        .title-page {
            text-align: center;
            page-break-after: always;
            padding-top: 200px;
        }
        
        .title-page h1 {
            font-size: 36pt;
            color: #1e4d72;
            border: none;
            page-break-before: avoid;
        }
        
        .subtitle {
            font-size: 18pt;
            color: #666;
            margin: 20px 0;
        }
        
        .author {
            font-size: 14pt;
            color: #333;
            margin-top: 50px;
        }
        
        .toc {
            page-break-after: always;
        }
        
        .toc ul {
            list-style-type: none;
            padding-left: 0;
        }
        
        .toc li {
            margin: 8px 0;
            padding-left: 20px;
        }
        
        .toc a {
            text-decoration: none;
            color: #1e4d72;
        }
        
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 10pt;
            color: #666;
            text-align: center;
        }
        
        .page-break {
            page-break-before: always;
        }
        
        .no-break {
            page-break-inside: avoid;
        }
        
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
        }
        
        .center {
            text-align: center;
        }
    </style>
</head>
<body>
{content}
</body>
</html>
"""

def generate_pdf_tutorial():
    """Generate professional PDF tutorial"""
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    if not check_wkhtmltopdf():
        return False
    
    import pdfkit
    from bs4 import BeautifulSoup
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    md_file = os.path.join(base_dir, 'docs', 'Blue_Water_Macro_Quantitative_Trading_Tutorial.md')
    output_file = os.path.join(base_dir, 'docs', 'Blue_Water_Macro_Tutorial.pdf')
    
    # Read markdown file
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
    except FileNotFoundError:
        print(f"Markdown file not found: {md_file}")
        return False
    
    # Convert markdown to HTML
    md = markdown.Markdown(extensions=[
        'codehilite',
        'tables',
        'toc',
        'fenced_code',
        'attr_list'
    ])
    
    html_content = md.convert(md_content)
    
    # Parse with BeautifulSoup for post-processing
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Add page breaks before each h1 (except first)
    h1_tags = soup.find_all('h1')
    for i, h1 in enumerate(h1_tags):
        if i > 0:  # Skip first h1
            h1['class'] = h1.get('class', []) + ['page-break']
    
    # Process code blocks for better PDF formatting
    for pre in soup.find_all('pre'):
        pre['class'] = pre.get('class', []) + ['no-break']
    
    # Create title page
    title_page = f"""
    <div class="title-page">
        <h1>Blue Water Macro<br>Quantitative Trading Framework</h1>
        <p class="subtitle">Complete Educational Tutorial</p>
        <p class="subtitle">Institutional-Grade Quantitative Finance for Students</p>
        <p class="author">Developed by Conrad Gann<br>Blue Water Macro Corp.</p>
        <p class="author">© 2025 Blue Water Macro Corp.<br>All Rights Reserved</p>
        <p class="author">Generated: {datetime.now().strftime('%B %d, %Y')}</p>
    </div>
    """
    
    # Combine everything
    full_html = create_html_template().format(
        content=title_page + str(soup)
    )
    
    # PDF generation options
    options = {
        'page-size': 'A4',
        'margin-top': '1in',
        'margin-right': '1in',
        'margin-bottom': '1in',
        'margin-left': '1in',
        'encoding': "UTF-8",
        'no-outline': None,
        'enable-local-file-access': None,
        'print-media-type': None,
        'disable-smart-shrinking': None,
        'header-center': 'Blue Water Macro Quantitative Trading Tutorial',
        'header-font-size': '9',
        'header-spacing': '5',
        'footer-center': 'Page [page] of [topage]',
        'footer-font-size': '9',
        'footer-spacing': '5'
    }
    
    try:
        # Generate PDF
        print("Generating PDF tutorial...")
        pdfkit.from_string(full_html, output_file, options=options)
        print(f"✅ PDF tutorial generated successfully: {output_file}")
        
        # Get file size for confirmation
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"📄 File size: {file_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Blue Water Macro PDF Tutorial Generator")
    print("=" * 50)
    
    success = generate_pdf_tutorial()
    
    if success:
        print("\n🎉 Tutorial PDF generation completed successfully!")
        print("\n📖 The PDF tutorial includes:")
        print("   • Complete framework overview")
        print("   • Step-by-step tutorials") 
        print("   • Advanced concepts and techniques")
        print("   • Career development guidance")
        print("   • Professional code templates")
        print("   • Mathematical foundations")
        print("\n💼 Perfect for:")
        print("   • Offline study and reference")
        print("   • Student distribution")
        print("   • Academic course materials")
        print("   • Professional training")
    else:
        print("\n❌ PDF generation failed. Please check dependencies and try again.")
        sys.exit(1)
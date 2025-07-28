#!/usr/bin/env python3
"""
HTML Tutorial Generator for Blue Water Macro Quantitative Trading Framework

This script creates a standalone HTML version of the tutorial that can be:
1. Viewed in any web browser
2. Easily printed to PDF using browser print function
3. Shared via email or web hosting
4. Used offline without dependencies

Usage:
    python scripts/generate_html_tutorial.py
"""

import os
import re
from datetime import datetime

def create_html_template():
    """Create professional HTML template with Blue Water Macro styling"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blue Water Macro Quantitative Trading Tutorial</title>
    <style>
        @media print {{
            @page {{
                size: A4;
                margin: 1in;
            }}
            .no-print {{ display: none; }}
            h1 {{ page-break-before: always; }}
            h1:first-of-type {{ page-break-before: avoid; }}
            pre, .code-block {{ page-break-inside: avoid; }}
        }}
        
        body {{
            font-family: 'Georgia', serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 50px;
            padding: 30px;
            background: linear-gradient(135deg, #1e4d72, #2e5984);
            color: white;
            border-radius: 10px;
        }}
        
        .logo {{
            max-width: 200px;
            margin-bottom: 20px;
        }}
        
        h1 {{
            color: #1e4d72;
            border-bottom: 3px solid #1e4d72;
            padding-bottom: 10px;
            margin-top: 40px;
            font-size: 2.2em;
        }}
        
        h2 {{
            color: #2e5984;
            border-bottom: 2px solid #2e5984;
            padding-bottom: 5px;
            margin-top: 35px;
            font-size: 1.8em;
        }}
        
        h3 {{
            color: #4a7ba7;
            margin-top: 30px;
            font-size: 1.4em;
        }}
        
        h4 {{
            color: #5d8bb5;
            margin-top: 25px;
            font-size: 1.2em;
        }}
        
        .toc {{
            background-color: #f8f9fa;
            padding: 30px;
            border-radius: 8px;
            margin: 30px 0;
            border-left: 5px solid #1e4d72;
        }}
        
        .toc h2 {{
            margin-top: 0;
            color: #1e4d72;
            border: none;
        }}
        
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        
        .toc li {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        
        .toc a {{
            text-decoration: none;
            color: #1e4d72;
            font-weight: 500;
        }}
        
        .toc a:hover {{
            color: #2e5984;
            text-decoration: underline;
        }}
        
        pre {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 20px;
            overflow-x: auto;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.4;
            margin: 20px 0;
        }}
        
        code {{
            background-color: #f8f9fa;
            padding: 3px 6px;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
            color: #d73a49;
            border: 1px solid #e1e4e8;
        }}
        
        pre code {{
            background-color: transparent;
            color: #333;
            padding: 0;
            border: none;
        }}
        
        blockquote {{
            border-left: 4px solid #1e4d72;
            padding-left: 20px;
            margin-left: 0;
            font-style: italic;
            color: #666;
            background-color: #f9f9f9;
            padding: 15px 20px;
            border-radius: 0 6px 6px 0;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 25px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 15px;
            text-align: left;
        }}
        
        th {{
            background-color: #1e4d72;
            color: white;
            font-weight: bold;
        }}
        
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        
        tr:hover {{
            background-color: #f1f5f9;
        }}
        
        .highlight {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
        
        .info-box {{
            background-color: #d1ecf1;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #0dcaf0;
            margin: 20px 0;
        }}
        
        .warning-box {{
            background-color: #f8d7da;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #dc3545;
            margin: 20px 0;
        }}
        
        .footer {{
            margin-top: 50px;
            padding: 30px;
            background-color: #f8f9fa;
            border-radius: 8px;
            text-align: center;
            color: #666;
            border-top: 3px solid #1e4d72;
        }}
        
        .print-button {{
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #1e4d72;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            z-index: 1000;
        }}
        
        .print-button:hover {{
            background-color: #2e5984;
        }}
        
        .section-nav {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin: 20px 0;
            text-align: center;
        }}
        
        .section-nav a {{
            color: #1e4d72;
            text-decoration: none;
            margin: 0 15px;
            font-weight: 500;
        }}
        
        .section-nav a:hover {{
            text-decoration: underline;
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .center {{
            text-align: center;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            background-color: #1e4d72;
            color: white;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            margin: 2px;
        }}
        
        ul, ol {{
            padding-left: 30px;
        }}
        
        li {{
            margin: 8px 0;
        }}
        
        .metadata {{
            background-color: #e7f3ff;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            border: 1px solid #b3d7ff;
        }}
    </style>
</head>
<body>
    <button class="print-button no-print" onclick="window.print()">📄 Print to PDF</button>
    
    <div class="header">
        <h1 style="color: white; border: none; margin: 0; font-size: 2.5em;">Blue Water Macro</h1>
        <h2 style="color: #e6f2ff; border: none; margin: 10px 0; font-size: 1.8em;">Quantitative Trading Framework</h2>
        <p style="font-size: 1.2em; margin: 20px 0;">Complete Educational Tutorial</p>
        <div class="metadata">
            <strong>Developed by:</strong> Conrad Gann<br>
            <strong>Organization:</strong> Blue Water Macro Corp.<br>
            <strong>Generated:</strong> {generation_date}<br>
            <strong>License:</strong> Blue Water Macro Educational License (BWMEL)
        </div>
    </div>

{content}

    <div class="footer">
        <h3>Blue Water Macro Corp.</h3>
        <p><strong>Advancing Quantitative Finance Education</strong></p>
        <p>© 2025 Blue Water Macro Corp. All Rights Reserved</p>
        <p>Licensed under the Blue Water Macro Educational License (BWMEL)</p>
        <p>For commercial licensing: <a href="mailto:licensing@bluewatermacro.com">licensing@bluewatermacro.com</a></p>
        <div style="margin-top: 20px;">
            <span class="badge">Professional</span>
            <span class="badge">Educational</span>
            <span class="badge">Open Source</span>
            <span class="badge">Institutional Grade</span>
        </div>
    </div>

    <script>
        // Add smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({{
                    behavior: 'smooth'
                }});
            }});
        }});
        
        // Add copy code functionality
        document.querySelectorAll('pre').forEach(pre => {{
            const button = document.createElement('button');
            button.textContent = 'Copy';
            button.style.cssText = 'position: absolute; top: 10px; right: 10px; padding: 5px 10px; background: #1e4d72; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;';
            button.className = 'no-print';
            pre.style.position = 'relative';
            pre.appendChild(button);
            
            button.addEventListener('click', () => {{
                navigator.clipboard.writeText(pre.textContent).then(() => {{
                    button.textContent = 'Copied!';
                    setTimeout(() => button.textContent = 'Copy', 2000);
                }});
            }});
        }});
    </script>
</body>
</html>"""

def markdown_to_html(md_text):
    """Convert markdown to HTML (simple implementation)"""
    html = md_text
    
    # Headers
    html = re.sub(r'^# (.*$)', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.*$)', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.*$)', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.*$)', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    
    # Code blocks
    html = re.sub(r'```python\n(.*?)\n```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'```bash\n(.*?)\n```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'```\n(.*?)\n```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    
    # Inline code
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
    
    # Bold and italic
    html = re.sub(r'\*\*([^\*]+)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*([^\*]+)\*', r'<em>\1</em>', html)
    
    # Links
    html = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', html)
    
    # Lists
    lines = html.split('\n')
    in_list = False
    result_lines = []
    
    for line in lines:
        if line.strip().startswith('- ') or line.strip().startswith('* '):
            if not in_list:
                result_lines.append('<ul>')
                in_list = True
            item = line.strip()[2:]
            result_lines.append(f'<li>{item}</li>')
        elif line.strip().startswith(('1. ', '2. ', '3. ', '4. ', '5. ')):
            if not in_list:
                result_lines.append('<ol>')
                in_list = True
            item = line.strip()[3:]
            result_lines.append(f'<li>{item}</li>')
        else:
            if in_list:
                result_lines.append('</ul>' if result_lines[-2].startswith('<ul>') or '<li>' in result_lines[-1] else '</ol>')
                in_list = False
            if line.strip():
                result_lines.append(f'<p>{line}</p>')
            else:
                result_lines.append('<br>')
    
    if in_list:
        result_lines.append('</ul>')
    
    html = '\n'.join(result_lines)
    
    # Tables (simple implementation)
    html = re.sub(r'\|([^\|]+)\|', lambda m: '<tr>' + ''.join(f'<td>{cell.strip()}</td>' for cell in m.group(1).split('|')) + '</tr>', html)
    
    # Paragraphs
    html = re.sub(r'\n\n', '</p><p>', html)
    
    return html

def generate_html_tutorial():
    """Generate HTML tutorial"""
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    md_file = os.path.join(base_dir, 'docs', 'Blue_Water_Macro_Quantitative_Trading_Tutorial.md')
    output_file = os.path.join(base_dir, 'docs', 'Blue_Water_Macro_Tutorial.html')
    
    # Read markdown file
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
    except FileNotFoundError:
        print(f"❌ Markdown file not found: {md_file}")
        return False
    
    # Convert markdown to HTML (basic implementation)
    html_content = markdown_to_html(md_content)
    
    # Generate current date
    generation_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    
    # Create full HTML
    template = create_html_template()
    full_html = template.format(
        content=html_content,
        generation_date=generation_date
    )
    
    # Write HTML file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"✅ HTML tutorial generated successfully!")
        print(f"📁 Location: {output_file}")
        
        # Get file size
        file_size = os.path.getsize(output_file) / 1024  # KB
        print(f"📄 File size: {file_size:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating HTML: {e}")
        return False

if __name__ == "__main__":
    print("🌐 Blue Water Macro HTML Tutorial Generator")
    print("=" * 50)
    
    success = generate_html_tutorial()
    
    if success:
        print("\n🎉 HTML tutorial generation completed!")
        print("\n📖 Features included:")
        print("   • Professional Blue Water Macro styling")
        print("   • Print-to-PDF functionality (Ctrl+P)")
        print("   • Interactive navigation")
        print("   • Code copy buttons")
        print("   • Responsive design")
        print("   • Offline viewing capability")
        print("\n💻 How to use:")
        print("   • Open the HTML file in any web browser")
        print("   • Use browser's Print function to save as PDF")
        print("   • Share via email or web hosting")
        print("   • Works completely offline")
    else:
        print("\n❌ HTML generation failed.")
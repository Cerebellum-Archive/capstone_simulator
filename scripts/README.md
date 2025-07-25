# PDF Tutorial Generation

This directory contains scripts to generate professional PDF versions of the Blue Water Macro tutorials.

## Quick Setup

1. **Install PDF dependencies:**
```bash
pip install markdown pdfkit beautifulsoup4 pygments
```

2. **Install wkhtmltopdf:**
```bash
# macOS
brew install wkhtmltopdf

# Ubuntu/Debian
sudo apt-get install wkhtmltopdf

# Windows - Download from: https://wkhtmltopdf.org/downloads.html
```

3. **Generate PDF:**
```bash
python scripts/generate_pdf_tutorial.py
```

## Output

The script generates `docs/Blue_Water_Macro_Tutorial.pdf` - a comprehensive, professionally formatted tutorial suitable for:

- **Offline Reading**: Complete framework guide without internet
- **Student Distribution**: Professional materials for courses
- **Career Portfolios**: Demonstrate quantitative finance expertise
- **Reference Guide**: Quick lookup for concepts and code templates

## Features

- **Professional Layout**: Blue Water Macro corporate styling
- **Table of Contents**: Easy navigation
- **Code Highlighting**: Syntax-highlighted Python examples  
- **Mathematical Formulas**: Properly formatted equations
- **Page Breaks**: Logical section organization
- **Headers/Footers**: Professional document structure
"""
Convert README.md to PDF for study purposes.

Uses markdown2 for parsing and weasyprint for PDF generation.
If weasyprint is not available, falls back to creating an HTML file.
"""

import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install required packages."""
    packages = ["markdown2", "weasyprint"]
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def convert_with_weasyprint(markdown_path: Path, output_path: Path):
    """Convert markdown to PDF using weasyprint."""
    import markdown2
    from weasyprint import HTML, CSS
    
    # Read markdown
    md_content = markdown_path.read_text(encoding="utf-8")
    
    # Convert to HTML
    html_content = markdown2.markdown(
        md_content,
        extras=[
            "fenced-code-blocks",
            "tables",
            "header-ids",
            "code-friendly",
            "cuddled-lists",
        ]
    )
    
    # Add styling for better PDF output
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 11pt;
                line-height: 1.6;
                color: #333;
                max-width: 100%;
            }}
            h1 {{
                color: #2196F3;
                border-bottom: 2px solid #2196F3;
                padding-bottom: 10px;
                page-break-after: avoid;
            }}
            h2 {{
                color: #1976D2;
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
                margin-top: 30px;
                page-break-after: avoid;
            }}
            h3 {{
                color: #0D47A1;
                margin-top: 25px;
                page-break-after: avoid;
            }}
            h4 {{
                color: #333;
                margin-top: 20px;
            }}
            code {{
                background-color: #f5f5f5;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10pt;
            }}
            pre {{
                background-color: #2d2d2d;
                color: #f8f8f2;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                font-size: 9pt;
                page-break-inside: avoid;
            }}
            pre code {{
                background-color: transparent;
                color: inherit;
                padding: 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
                page-break-inside: avoid;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 10px;
                text-align: left;
            }}
            th {{
                background-color: #2196F3;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            blockquote {{
                border-left: 4px solid #2196F3;
                margin: 15px 0;
                padding-left: 15px;
                color: #666;
            }}
            hr {{
                border: none;
                border-top: 1px solid #ddd;
                margin: 30px 0;
            }}
            a {{
                color: #1976D2;
                text-decoration: none;
            }}
            ul, ol {{
                margin: 10px 0;
                padding-left: 25px;
            }}
            li {{
                margin: 5px 0;
            }}
            .toc {{
                background-color: #f5f5f5;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 30px;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Generate PDF
    HTML(string=styled_html).write_pdf(str(output_path))
    print(f"✓ PDF created: {output_path}")


def convert_with_pandoc(markdown_path: Path, output_path: Path):
    """Convert markdown to PDF using pandoc (if installed)."""
    try:
        subprocess.run(
            [
                "pandoc",
                str(markdown_path),
                "-o", str(output_path),
                "--pdf-engine=xelatex",
                "-V", "geometry:margin=1in",
                "-V", "fontsize=11pt",
            ],
            check=True
        )
        print(f"✓ PDF created with pandoc: {output_path}")
    except FileNotFoundError:
        return False
    return True


def convert_to_html_fallback(markdown_path: Path, output_path: Path):
    """Fallback: create HTML file if PDF generation fails."""
    import markdown2
    
    md_content = markdown_path.read_text(encoding="utf-8")
    html_content = markdown2.markdown(
        md_content,
        extras=["fenced-code-blocks", "tables", "header-ids"]
    )
    
    html_path = output_path.with_suffix(".html")
    
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>SoundPlot README</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; 
                max-width: 900px; 
                margin: 40px auto; 
                padding: 20px; 
                line-height: 1.6; 
                color: #333;
                background: #fff;
            }}
            h1 {{ color: #2196F3; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }}
            h2 {{ color: #1976D2; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
            h3 {{ color: #0D47A1; margin-top: 25px; }}
            
            /* Inline code */
            code {{ 
                background: #e8e8e8; 
                color: #c7254e;
                padding: 2px 6px; 
                border-radius: 3px; 
                font-family: Consolas, Monaco, monospace;
                font-size: 0.9em;
            }}
            
            /* Code blocks - LIGHT THEME for readability */
            pre {{ 
                background: #f6f8fa; 
                border: 1px solid #e1e4e8;
                padding: 16px; 
                border-radius: 6px; 
                overflow-x: auto; 
                font-size: 13px;
                line-height: 1.5;
            }}
            pre code {{ 
                background: transparent; 
                color: #24292e;
                padding: 0;
                font-size: inherit;
            }}
            
            /* Tables */
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background: #2196F3; color: white; }}
            tr:nth-child(even) {{ background: #f9f9f9; }}
            
            /* Other elements */
            blockquote {{ 
                border-left: 4px solid #2196F3; 
                margin: 15px 0; 
                padding-left: 15px; 
                color: #666; 
                background: #f9f9f9;
            }}
            hr {{ border: none; border-top: 1px solid #ddd; margin: 30px 0; }}
            a {{ color: #1976D2; }}
        </style>
    </head>
    <body>{html_content}</body>
    </html>
    """
    
    html_path.write_text(styled_html, encoding="utf-8")
    print(f"✓ HTML created (PDF fallback): {html_path}")
    print("  Open in browser and use Print > Save as PDF")
    return html_path


def convert_with_xhtml2pdf(html_content: str, output_path: Path):
    """Convert HTML to PDF using xhtml2pdf."""
    from xhtml2pdf import pisa
    
    # Replace Unicode box-drawing characters with ASCII for PDF compatibility
    # xhtml2pdf default fonts often lack these glyphs
    html_content = html_content.replace('├──', '|--')
    html_content = html_content.replace('└──', '+--')
    html_content = html_content.replace('│', '|')
    html_content = html_content.replace('&#9474;', '|') # HTML entity for │
    
    # Add simple CSS for xhtml2pdf
    # It requires internal CSS to be part of the HTML
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{ 
                font-family: Helvetica, sans-serif; 
                font-size: 10pt; 
                line-height: 1.5;
            }}
            h1 {{ color: #2196F3; border-bottom: 2px solid #2196F3; padding-bottom: 5px; }}
            h2 {{ color: #1976D2; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 20px; }}
            h3 {{ color: #0D47A1; margin-top: 15px; }}
            code {{ background-color: #f5f5f5; font-family: "Courier New", Courier, monospace; }}
            pre {{ 
                background-color: #f5f5f5; 
                padding: 10px; 
                font-family: "Courier New", Courier, monospace; 
                border: 1px solid #ddd;
                white-space: pre;
            }}
            blockquote {{ border-left: 4px solid #2196F3; padding-left: 10px; color: #666; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    with open(output_path, "wb") as f:
        pisa_status = pisa.CreatePDF(full_html, dest=f)
    
    if pisa_status.err:
        raise Exception("xhtml2pdf failed")
    
    print(f"✓ PDF created with xhtml2pdf: {output_path}")


def main():
    """Convert README.md to PDF."""
    readme_path = Path(__file__).parent / "README copy.md"
    pdf_path = Path(__file__).parent / "README copy.pdf"
    
    if not readme_path.exists():
        print(f"Error: {readme_path} not found")
        return
    
    print(f"Converting {readme_path.name} to PDF...")
    
    # 1. Read and parse Markdown (using markdown2)
    try:
        import markdown2
    except ImportError:
        install_dependencies()
        import markdown2

    md_content = readme_path.read_text(encoding="utf-8")
    html_content = markdown2.markdown(
        md_content,
        extras=["fenced-code-blocks", "tables", "header-ids"]
    )
    
    # 2. Try WeasyPrint (Best quality, hard requirements)
    try:
        print("Trying weasyprint...")
        convert_with_weasyprint(readme_path, pdf_path)
        return
    except Exception:
        print("WeasyPrint missing or failed (needs GTK3).")

    # 3. Try xhtml2pdf (Pure Python, good backup)
    try:
        print("Trying xhtml2pdf...")
        # Ensure installed
        try:
            import xhtml2pdf
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "xhtml2pdf"])
        
        convert_with_xhtml2pdf(html_content, pdf_path)
        return
    except Exception as e:
        print(f"xhtml2pdf failed: {e}")

    # 4. Try Pandoc (External Tool)
    if convert_with_pandoc(readme_path, pdf_path):
        return
    
    # 5. Fallback to HTML
    print("All PDF methods failed. Generating HTML view...")
    convert_to_html_fallback(readme_path, pdf_path)


if __name__ == "__main__":
    main()

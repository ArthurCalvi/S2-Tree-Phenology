#!/usr/bin/env python3
"""
Enhanced PDF annotation extractor with highlighted text and section detection.

Features:
- Extracts highlighted text
- Identifies section/heading for each annotation
- Color-coded categorization
- Multiple output formats
"""

import fitz  # PyMuPDF
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re


def rgb_to_color_name(rgb: Tuple[float, float, float]) -> str:
    """Convert RGB values (0-1 scale) to color name."""
    if not rgb or len(rgb) < 3:
        return "Unknown"

    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])

    # Orange/Yellow (high R and high G, low B) - general comments
    if r > 0.8 and g > 0.6 and b < 0.5:
        return "Yellow"

    # Pink/Red (high R, low-medium G, medium-high B) - critical issues
    if r > 0.8 and g < 0.6 and b > 0.3 and b < 0.8:
        return "Red"

    # Pure Red
    if r > 0.8 and g < 0.3 and b < 0.3:
        return "Red"

    # Green (medium-low R, high G, low-medium B) - positive feedback
    if r < 0.7 and g > 0.6 and b < 0.6:
        return "Green"

    # Blue (low R, low-medium G, high B) - citations
    if r < 0.5 and g < 0.7 and b > 0.7:
        return "Blue"

    # Cyan/Light Blue
    if r < 0.5 and g > 0.7 and b > 0.7:
        return "Blue"

    return f"Other-RGB({r:.2f}, {g:.2f}, {b:.2f})"


def extract_sections_from_page(page: fitz.Page) -> List[Dict]:
    """Extract section headings from a page based on font size and style."""
    sections = []
    blocks = page.get_text("dict")["blocks"]

    for block in blocks:
        if "lines" not in block:
            continue

        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                font_size = span["size"]
                font_flags = span["flags"]

                # Heuristic: headings are typically larger and/or bold
                # Also check for common section patterns
                is_heading = (
                    font_size > 11 or  # Larger than body text
                    font_flags & 2**4 or  # Bold
                    re.match(r'^\d+\.?\s+[A-Z]', text) or  # Numbered sections like "1. INTRO"
                    re.match(r'^[A-Z][A-Z\s]{3,}$', text)  # ALL CAPS headings
                )

                if is_heading and len(text) > 3 and len(text) < 200:
                    sections.append({
                        'text': text,
                        'y_position': span["bbox"][1],  # Top Y coordinate
                        'font_size': font_size
                    })

    return sections


def find_section_for_annotation(page: fitz.Page, annot_rect: fitz.Rect,
                                sections: List[Dict]) -> Optional[str]:
    """Find which section an annotation belongs to based on position."""
    annot_y = annot_rect.y0  # Top of annotation

    # Find the closest section heading above the annotation
    relevant_sections = [s for s in sections if s['y_position'] < annot_y]

    if relevant_sections:
        # Get the closest one (highest y_position below annotation)
        closest = max(relevant_sections, key=lambda s: s['y_position'])
        return closest['text']

    return None


def extract_highlighted_text(page: fitz.Page, annot) -> str:
    """Extract the text that was highlighted."""
    try:
        # Get the quadrilaterals (areas) covered by the highlight
        quads = annot.vertices

        if not quads or len(quads) < 4:
            return ""

        # Convert vertices to rectangles
        # Vertices come in groups of 4 (quad points)
        highlighted_texts = []

        for i in range(0, len(quads), 4):
            if i + 3 < len(quads):
                quad_points = quads[i:i+4]
                # Create rectangle from quad points
                x_coords = [p.x for p in quad_points]
                y_coords = [p.y for p in quad_points]
                rect = fitz.Rect(min(x_coords), min(y_coords),
                               max(x_coords), max(y_coords))

                # Extract text in this rectangle
                text = page.get_textbox(rect)
                if text and text.strip():
                    highlighted_texts.append(text.strip())

        return " ".join(highlighted_texts)

    except Exception as e:
        # Fallback: try to get text from annotation rect
        try:
            return page.get_textbox(annot.rect).strip()
        except:
            return ""


def extract_annotations(pdf_path: str) -> Dict[str, List[Dict]]:
    """Extract all annotations with highlighted text and section information."""

    doc = fitz.open(pdf_path)

    annotations_by_color = {
        "Yellow": [],
        "Blue": [],
        "Red": [],
        "Green": [],
        "Orange": [],
        "Other": []
    }

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extract sections from this page
        sections = extract_sections_from_page(page)

        # Process annotations
        for annot in page.annots():
            # Get annotation info
            annot_info = annot.info
            note = annot_info.get("content", "").strip()

            if not note:  # Skip annotations without notes
                continue

            author = annot_info.get("title", "Unknown")
            annot_type = annot.type[1] if annot.type else "Unknown"

            # Get color
            colors = annot.colors
            color_rgb = colors.get("stroke") if colors else None
            color_name = rgb_to_color_name(color_rgb) if color_rgb else "Unknown"

            # Extract highlighted text
            highlighted_text = extract_highlighted_text(page, annot)

            # Find section
            section = find_section_for_annotation(page, annot.rect, sections)

            # Get page label (might be different from page number)
            page_label = doc[page_num].get_label()

            annotation_data = {
                'page': page_num + 1,
                'page_label': page_label,
                'section': section or "Unknown section",
                'type': annot_type,
                'note': note,
                'highlighted_text': highlighted_text or "(no text extracted)",
                'author': author,
                'color': color_name,
                'rgb': list(color_rgb) if color_rgb else None,
                'position': {
                    'x0': annot.rect.x0,
                    'y0': annot.rect.y0,
                    'x1': annot.rect.x1,
                    'y1': annot.rect.y1
                }
            }

            # Categorize by color
            if color_name in annotations_by_color:
                annotations_by_color[color_name].append(annotation_data)
            else:
                annotations_by_color["Other"].append(annotation_data)

    doc.close()
    return annotations_by_color


def generate_markdown_report(annotations_by_color: Dict[str, List[Dict]],
                             pdf_name: str,
                             output_file: str = "annotation_report.md"):
    """Generate an enhanced markdown report."""

    color_categories = {
        "Yellow": "📝 General Comments/Suggestions",
        "Blue": "📚 Citation Checks",
        "Red": "🚨 Critical Issues",
        "Green": "✅ Positive Feedback",
        "Orange": "⚠️ Warnings",
        "Other": "❓ Other"
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# PDF Annotation Report (Enhanced)\n\n")
        f.write(f"**Document:** {pdf_name}\n\n")
        f.write("---\n\n")

        # Count total
        total = sum(len(annots) for annots in annotations_by_color.values() if annots)
        f.write(f"**Total Annotations:** {total}\n\n")

        # Write each category
        for color, category_name in color_categories.items():
            annots = annotations_by_color.get(color, [])

            if not annots:
                continue

            f.write(f"## {category_name}\n\n")
            f.write(f"*Count: {len(annots)}*\n\n")

            for i, annot in enumerate(annots, 1):
                f.write(f"### {i}. Page {annot['page']}")
                if annot['section'] != "Unknown section":
                    f.write(f" — {annot['section']}")
                f.write("\n\n")

                f.write(f"**Section:** {annot['section']}\n\n")
                f.write(f"**Highlighted Text:**\n")
                f.write(f"> {annot['highlighted_text']}\n\n")

                f.write(f"**Type:** {annot['type']}\n\n")
                f.write(f"**Author:** {annot['author']}\n\n")
                f.write(f"**Color:** {annot['color']}")

                if annot['rgb']:
                    rgb_str = ', '.join([f"{x:.2f}" for x in annot['rgb']])
                    f.write(f" (RGB: {rgb_str})")

                f.write(f"\n\n**Note/Comment:**\n\n")
                f.write(f"> {annot['note']}\n\n")
                f.write("---\n\n")


def generate_json_report(annotations_by_color: Dict[str, List[Dict]],
                        output_file: str = "annotations.json"):
    """Generate JSON export."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations_by_color, f, indent=2, ensure_ascii=False)


def print_summary(annotations_by_color: Dict[str, List[Dict]]):
    """Print summary to console."""
    print("\n" + "="*60)
    print("PDF ANNOTATION EXTRACTION SUMMARY (Enhanced)")
    print("="*60 + "\n")

    total = 0
    for color, annots in annotations_by_color.items():
        count = len(annots)
        if count > 0:
            total += count
            print(f"{color:15s}: {count:3d} annotations")

    print(f"\n{'TOTAL':15s}: {total:3d} annotations\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract PDF annotations with highlighted text and section detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Color Categories:
  Yellow  - General comments/suggestions
  Blue    - Citation checks
  Red     - Critical issues
  Green   - Positive feedback
  Orange  - Warnings

Features:
  ✓ Extracts highlighted text
  ✓ Identifies document sections
  ✓ Color-coded categorization
  ✓ Multiple output formats

Examples:
  python3 extract_pdf_annotations_enhanced.py paper.pdf
  python3 extract_pdf_annotations_enhanced.py --pdf paper.pdf --output review
        """
    )

    parser.add_argument(
        'pdf',
        nargs='?',
        default="Frontier_Phenology_Arthur_Calvi_Sarah_Brood__Alex_Copy_-8.pdf",
        help="Path to PDF file"
    )

    parser.add_argument(
        '--pdf', '-p',
        dest='pdf_path',
        help="Alternative way to specify PDF path"
    )

    parser.add_argument(
        '--output', '-o',
        default="annotation_report_enhanced",
        help="Output file basename (default: annotation_report_enhanced)"
    )

    args = parser.parse_args()

    # Use --pdf if specified, otherwise use positional argument
    pdf_path = args.pdf_path if args.pdf_path else args.pdf

    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found: {pdf_path}")
        print(f"\nUse --help for usage information")
        parser.exit(1)

    print(f"Extracting annotations from: {pdf_path}")
    print("Features: Highlighted text + Section detection\n")

    # Extract annotations
    annotations_by_color = extract_annotations(pdf_path)

    # Print summary
    print_summary(annotations_by_color)

    # Generate reports
    markdown_file = f"{args.output}.md"
    json_file = f"{args.output}.json"

    generate_markdown_report(annotations_by_color, Path(pdf_path).name, markdown_file)
    print(f"✓ Markdown report saved to: {markdown_file}")

    generate_json_report(annotations_by_color, json_file)
    print(f"✓ JSON data saved to: {json_file}")

    print("\nDone! 🎉\n")

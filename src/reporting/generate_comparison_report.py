#!/usr/bin/env python3
"""
generate_comparison_report.py
-----------------------------
Generates a formatted HTML report from a comparison CSV file created by compare_maps.py.
Detects the reference map type (DLT or BDForet) based on CSV headers and displays
metrics accordingly.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
import numpy as np # For potential weighted average calculation

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("generate_report")

# --- Constants ---
CUSTOM_CLASSES = ['Deciduous', 'Evergreen'] # Fixed order based on compare_maps.py (index 0, 1)

# --- HTML Generation Helper Functions ---

def generate_html_header(title):
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 80%; margin-bottom: 20px; margin-left: auto; margin-right: auto; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .summary-section, .region-section {{ margin-bottom: 30px; padding: 15px; border: 1px solid #eee; border-radius: 5px; background-color: #fff; }}
        .key-value {{ margin-bottom: 5px; }}
        .key-value span:first-child {{ display: inline-block; width: 250px; font-weight: bold; color: #555; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
"""

def generate_html_footer():
    return """
</body>
</html>
"""

def generate_table_html(headers, data):
    html = "<table>\n<thead>\n<tr>" + ''.join([f'<th>{h}</th>' for h in headers]) + "</tr>\n</thead>\n<tbody>\n"
    for row in data:
        html += "<tr>" + ''.join([f'<td>{item}</td>' for item in row]) + "</tr>\n"
    html += "</tbody>\n</table>\n"
    return html

def detect_reference_type(headers):
    """Detects the reference map type and classes from CSV headers."""
    logger.debug(f"Detecting reference type from headers: {headers}")
    # Check for DLT-specific confusion matrix column names
    if any('Broadleaved' in h or 'Coniferous' in h for h in headers if h.startswith('cm_')):
        ref_type = 'DLT'
        ref_classes = ['Broadleaved', 'Coniferous'] # Order corresponds to index 0, 1 in CM columns
        logger.info(f"Detected reference type: {ref_type} with classes {ref_classes}")
        return ref_type, ref_classes
    # Check for BDForet-specific confusion matrix column names (Deciduous/Evergreen)
    elif any(h.startswith('cm_Deciduous_vs_Deciduous') or h.startswith('cm_Evergreen_vs_Evergreen') for h in headers):
        ref_type = 'BDForet'
        # Assuming BDForet also uses 1: Deciduous, 2: Evergreen in its source
        ref_classes = ['Deciduous', 'Evergreen'] # Order corresponds to index 0, 1 in CM columns
        logger.info(f"Detected reference type: {ref_type} with classes {ref_classes}")
        return ref_type, ref_classes
    else:
        logger.warning("Could not reliably detect reference type from CM headers. Assuming DLT.")
        # Defaulting or raising an error might be better? For now, default to DLT
        return 'DLT', ['Broadleaved', 'Coniferous']

def parse_csv_data(csv_path):
    """Reads the CSV file and converts numeric values."""
    data = []
    headers = []
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            headers = reader.fieldnames
            if not headers:
                logger.error(f"CSV file is empty or header is missing: {csv_path}")
                return [], []

            for row in reader:
                parsed_row = {}
                for key, value in row.items():
                    try:
                        # Attempt to convert to int, then float, else keep as string
                        if value.isdigit():
                            parsed_row[key] = int(value)
                        else:
                            parsed_row[key] = float(value)
                    except (ValueError, TypeError):
                        parsed_row[key] = value # Keep as string if conversion fails (like region name)
                data.append(parsed_row)
        logger.info(f"Successfully read {len(data)} rows from {csv_path}")
        return data, headers
    except FileNotFoundError:
        logger.error(f"Input CSV file not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_path}: {e}")
        sys.exit(1)

# --- HTML Generation Function ---
def generate_html_report(data, headers, custom_classes, ref_classes, ref_type, output_path):
    """Generates the comparison report in HTML format."""
    report_title = f"Forest Comparison Report: Custom Map vs. {ref_type}"
    html_content = generate_html_header(report_title)

    # --- Overall Summary ---
    total_pixels_all_regions = sum(row.get('total_pixels', 0) for row in data)
    weighted_oa_sum = sum(row.get('overall_accuracy', 0.0) * row.get('total_pixels', 0) for row in data)
    weighted_kappa_sum = sum(row.get('kappa', 0.0) * row.get('total_pixels', 0) for row in data)

    html_content += "<div class='summary-section'>\n"
    html_content += "<h2>Overall Weighted Summary</h2>\n"
    if total_pixels_all_regions > 0:
        avg_oa = weighted_oa_sum / total_pixels_all_regions
        avg_kappa = weighted_kappa_sum / total_pixels_all_regions
        html_content += f"<p class='key-value'><span>Total Pixels Across All Regions:</span> <span>{total_pixels_all_regions:,}</span></p>\n"
        html_content += f"<p class='key-value'><span>Weighted Average OA:</span> <span>{avg_oa:.4f}</span></p>\n"
        html_content += f"<p class='key-value'><span>Weighted Average Kappa:</span> <span>{avg_kappa:.4f}</span></p>\n"
    else:
        html_content += "<p>No valid pixels found for comparison across all regions.</p>\n"
    html_content += "</div>\n"

    # --- Per-Region Details ---
    html_content += "<h2>Per-Region Details</h2>\n"

    for row in data:
        region_id = row.get('eco_region_id', 'N/A')
        region_name = row.get('eco_region_name', 'N/A')
        total_pixels = row.get('total_pixels', 0)
        oa = row.get('overall_accuracy', 0.0)
        kappa = row.get('kappa', 0.0)

        html_content += f"<div class='region-section'>\n"
        html_content += f"<h3>Eco-Region: {region_name} (ID: {region_id})</h3>\n"

        html_content += f"<p class='key-value'><span>Total Forest Pixels Compared:</span> <span>{total_pixels:,}</span></p>\n"
        html_content += f"<p class='key-value'><span>Overall Accuracy (OA):</span> <span>{oa:.4f}</span></p>\n"
        html_content += f"<p class='key-value'><span>Cohen's Kappa:</span> <span>{kappa:.4f}</span></p>\n"

        # --- Confusion Matrix ---
        html_content += "<h4>Confusion Matrix (Rows: Custom, Cols: Reference)</h4>\n"
        cm_headers = ["Custom \\ Reference"] + ref_classes # Header for the row labels column
        cm_data = []
        for i, custom_c in enumerate(custom_classes):
            row_data = [custom_c] # First column is the custom class label
            for j, ref_c in enumerate(ref_classes):
                cm_key = f'cm_{custom_c}_vs_{ref_c}'
                cm_value = row.get(cm_key, 0)
                row_data.append(f"{int(cm_value):,}") # Format with comma
            cm_data.append(row_data)
        html_content += generate_table_html(cm_headers, cm_data)

        # --- Class Metrics ---
        html_content += "<h4>Performance Metrics (for Custom Map Classes)</h4>\n"
        metric_headers = ["Class", "Precision", "Recall", "F1-Score"]
        metric_data = []
        for custom_c in custom_classes:
            precision = row.get(f'precision_{custom_c}', 0.0)
            recall = row.get(f'recall_{custom_c}', 0.0)
            f1 = row.get(f'f1_{custom_c}', 0.0)
            metric_data.append([custom_c, f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])
        html_content += generate_table_html(metric_headers, metric_data)

        html_content += "</div>\n" # End region-section

    html_content += generate_html_footer()

    # --- Save HTML File ---
    try:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(html_content)
        logger.info(f"HTML report saved successfully to: {output_path}")
    except IOError as e:
        logger.error(f"Failed to write HTML file to {output_path}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while writing HTML: {e}")
        sys.exit(1)

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(
        description="Generate a formatted HTML report from a forest type comparison CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--csv-file", type=str, required=True,
                        help="Path to the input CSV file generated by compare_maps.py.")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to save the output HTML report (e.g., report.html).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    csv_path = Path(args.csv_file)
    output_path = Path(args.output_file)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Read Data and Detect Type ---
    data, headers = parse_csv_data(csv_path)
    if not data:
        logger.info("No data found in CSV. Exiting.")
        sys.exit(0)

    ref_type, ref_classes = detect_reference_type(headers)

    # --- Generate HTML Report ---
    generate_html_report(data, headers, CUSTOM_CLASSES, ref_classes, ref_type, output_path)

if __name__ == "__main__":
    main() 
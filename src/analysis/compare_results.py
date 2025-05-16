import json
import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define file paths
baseline_overall_json = "results/final_model/phenology_metrics_selected_features.json"
baseline_eco_csv = "results/final_model/phenology_eco_metrics_selected_features.csv"
monge_overall_json = "results/domain_adaptation/skada_monge_self_adapt_test_cv5/skada_monge_selected_features_metrics_test_cv5.json"
monge_eco_csv = "results/domain_adaptation/skada_monge_self_adapt_test_cv5/skada_monge_selected_features_eco_metrics_test_cv5.csv"
output_csv = "comparison_f1_macro.csv"
progress_file = "progress.md"

def update_progress(message):
    with open(progress_file, "a") as f:
        f.write(f"- {message}\n")
    logging.info(f"Progress: {message}")

def get_f1_macro(file_path, file_type, eco_region_col='eco_region', f1_col='f1_macro_mean', overall_key='mean_f1_macro'):
    logging.info(f"Reading F1 macro from {file_path}")
    if file_type == 'json':
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return {"Overall": data.get(overall_key)}
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return {"Overall": None}
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {file_path}")
            return {"Overall": None}
        except Exception as e:
            logging.error(f"An unexpected error occurred while reading {file_path}: {e}")
            return {"Overall": None}
    elif file_type == 'csv':
        try:
            df = pd.read_csv(file_path)
            # Standardize column names for f1_macro if needed
            if 'f1_macro_mean' in df.columns:
                 f1_col_actual = 'f1_macro_mean'
            elif 'f1_macro' in df.columns: # Fallback if 'f1_macro_mean' is not present
                 f1_col_actual = 'f1_macro'
            else:
                logging.error(f"F1 macro column not found in {file_path} using expected names 'f1_macro_mean' or 'f1_macro'.")
                return {}

            if eco_region_col not in df.columns:
                logging.error(f"Eco region column '{eco_region_col}' not found in {file_path}.")
                return {}
            
            return df.set_index(eco_region_col)[f1_col_actual].to_dict()
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return {}
        except pd.errors.EmptyDataError:
            logging.error(f"No data: {file_path} is empty.")
            return {}
        except Exception as e:
            logging.error(f"An unexpected error occurred while reading {file_path}: {e}")
            return {}
    return {}

# Initialize progress file
if not os.path.exists(progress_file):
    with open(progress_file, "w") as f:
        f.write("# Project Progress\n\n")
    logging.info(f"Initialized {progress_file}")

update_progress("Started results comparison script.")

# Get F1 scores
baseline_overall_f1 = get_f1_macro(baseline_overall_json, 'json')
baseline_eco_f1 = get_f1_macro(baseline_eco_csv, 'csv')

monge_overall_f1 = get_f1_macro(monge_overall_json, 'json')
monge_eco_f1 = get_f1_macro(monge_eco_csv, 'csv', f1_col='f1_macro_mean') # skada_monge_selected_features_eco_metrics_test_cv5.csv uses 'f1_macro_mean'

# Combine overall and eco-region scores
baseline_f1 = {**baseline_overall_f1, **baseline_eco_f1}
monge_f1 = {**monge_overall_f1, **monge_eco_f1}

# Create DataFrame for comparison
comparison_data = []
all_eco_regions = sorted(list(set(baseline_f1.keys()) | set(monge_f1.keys())))

for region in all_eco_regions:
    b_f1 = baseline_f1.get(region)
    m_f1 = monge_f1.get(region)
    relative_diff = None
    if b_f1 is not None and m_f1 is not None and b_f1 != 0:
        relative_diff = (m_f1 - b_f1) / b_f1
    elif b_f1 is None or m_f1 is None:
        logging.warning(f"Missing F1 score for region '{region}'. Baseline: {b_f1}, Monge: {m_f1}")


    comparison_data.append({
        "eco_region": region,
        "baseline_f1_macro": b_f1,
        "monge_self_training_f1_macro": m_f1,
        "relative_difference_f1_macro": relative_diff
    })

comparison_df = pd.DataFrame(comparison_data)
# Ensure 'Overall' is the first row if it exists
if "Overall" in comparison_df['eco_region'].values:
    overall_row = comparison_df[comparison_df['eco_region'] == "Overall"]
    other_rows = comparison_df[comparison_df['eco_region'] != "Overall"]
    comparison_df = pd.concat([overall_row, other_rows]).reset_index(drop=True)


# Save to CSV
try:
    comparison_df.to_csv(output_csv, index=False)
    logging.info(f"Comparison saved to {output_csv}")
    update_progress(f"Successfully created comparison CSV: {output_csv}")
except Exception as e:
    logging.error(f"Failed to save CSV to {output_csv}: {e}")
    update_progress(f"Failed to create comparison CSV: {output_csv} due to {e}")

update_progress("Finished results comparison script.")

print(f"Comparison CSV generated: {output_csv}") 
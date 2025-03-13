#!/bin/bash

# Script to create a VRT file from feature TIF files

# Default values
FEATURES_DIR="data/features"
OUTPUT_VRT=""
VERBOSE=""
LOGLEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --features_dir)
      FEATURES_DIR="$2"
      shift 2
      ;;
    --output_vrt)
      OUTPUT_VRT="--output_vrt $2"
      shift 2
      ;;
    --verbose)
      VERBOSE="--verbose"
      shift
      ;;
    --loglevel)
      LOGLEVEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Run the Python script
echo "Creating VRT file for features in $FEATURES_DIR..."

python src/sampling/create_features_vrt.py \
  --features_dir "$FEATURES_DIR" \
  $OUTPUT_VRT \
  --loglevel "$LOGLEVEL" \
  $VERBOSE

# Check if the script ran successfully
if [ $? -eq 0 ]; then
  echo "VRT file created successfully!"
else
  echo "Failed to create VRT file!"
  exit 1
fi 
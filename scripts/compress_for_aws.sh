#!/bin/bash

usage() {
    cat <<EOF
Usage: $(basename "$0") PARENT_DIRECTORY

Prepares an experiment directory for AWS upload by producing a stripped
copy alongside the original, then zipping it.

Steps performed:
  1. Copies exp_code/ subdirectory (if present)
  2. Strips data2d_distorted.csv.gz from any .braidz file (2D data is
     large and not needed for reprocessing; 3D trajectories are kept)
  3. Copies all .bag and .hdf5 files
  4. Zips the resulting directory

Output location:
  The new directory and zip are created next to PARENT_DIRECTORY:
    /path/to/my_experiment        (original, untouched)
    /path/to/my_experiment_no2d/  (stripped copy)
    /path/to/my_experiment_no2d.zip

Arguments:
  PARENT_DIRECTORY    Path to the experiment directory to process

Options:
  -h, --help          Show this help message and exit

Example:
  $(basename "$0") /home/caveman/data/test_data_1
EOF
}

# Show help
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No directory provided."
    echo "Run '$(basename "$0") --help' for usage."
    exit 1
fi

PARENT_DIR="$1"
GRANDPARENT_DIR=$(dirname "$PARENT_DIR")
PARENT_BASE=$(basename "$PARENT_DIR")
NEW_DIR="${GRANDPARENT_DIR}/${PARENT_BASE}_no2d"

# Check if parent directory exists
if [ ! -d "$PARENT_DIR" ]; then
    echo "Error: Directory '$PARENT_DIR' not found."
    echo "Run '$(basename "$0") --help' for usage."
    exit 1
fi

# Create new directory
mkdir -p "$NEW_DIR"
echo "Created directory: $NEW_DIR"

# 1. Copy exp_code subdirectory if it exists
if [ -d "$PARENT_DIR/exp_code" ]; then
    cp -r "$PARENT_DIR/exp_code" "$NEW_DIR/"
    echo "Copied exp_code directory"
fi

# 2. Process .braidz file
BRAIDZ_FILE=$(find "$PARENT_DIR" -maxdepth 1 -name "*.braidz" -type f | head -n 1)

if [ -n "$BRAIDZ_FILE" ]; then
    echo "Processing $(basename "$BRAIDZ_FILE")..."
    
    BASE_NAME=$(basename "${BRAIDZ_FILE%.braidz}")
    OUTPUT_FILE="$NEW_DIR/${BASE_NAME}_no2d.braidz"
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    
    # Unzip into temporary directory
    unzip "$BRAIDZ_FILE" -d "$TEMP_DIR"
    
    # Delete the specific file
    rm -f "$TEMP_DIR/data2d_distorted.csv.gz"
    
    # Rezip from within the temp directory
    (cd "$TEMP_DIR" && zip -r - *) > "$OUTPUT_FILE"
    
    # Clean up
    rm -rf "$TEMP_DIR"
    
    echo "Created: $(basename "$OUTPUT_FILE")"
else
    echo "Warning: No .braidz file found"
fi

# 3. Copy all .bag and .hdf5 files
find "$PARENT_DIR" -maxdepth 1 -type f \( -name "*.bag" -o -name "*.hdf5" \) -exec cp {} "$NEW_DIR/" \;

BAG_COUNT=$(find "$PARENT_DIR" -maxdepth 1 -name "*.bag" -type f | wc -l)
HDF5_COUNT=$(find "$PARENT_DIR" -maxdepth 1 -name "*.hdf5" -type f | wc -l)

echo "Copied $BAG_COUNT .bag file(s)"
echo "Copied $HDF5_COUNT .hdf5 file(s)"

echo "Done! New directory created at: $NEW_DIR"

# 4. Zip the new directory
ZIP_FILE="${NEW_DIR}.zip"
echo "Zipping to $(basename "$ZIP_FILE")..."
(cd "$GRANDPARENT_DIR" && zip -r "$(basename "$ZIP_FILE")" "$(basename "$NEW_DIR")")
echo "Created zip: $ZIP_FILE"
#!/bin/bash
# Author: Laura Paccini (laura.paccini@pnnl.gov)
# Date: May 27, 2025
# Description: Script to extract environmental variables around MCS locations

#SBATCH -N 2
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -J extract_env_vars
#SBATCH -A m1867
#SBATCH --mail-user=laura.paccini@pnnl.gov
#SBATCH --mail-type=FAIL,END

module load python
module list
conda activate /global/common/software/m1867/python/lp_env/easy

# Set up paths and parameters
ROOT_DIR="/global/cfs/cdirs/m4581/gsharing/hackathon"
TRACK_FILE="${ROOT_DIR}/tracking/mcs/scream/stats/mcs_tracks_final_20190801.0000_20200901.0000.nc"
OUTPUT_DIR="/pscratch/sd/p/paccini/temp/hackathon/spatial_environmental_data"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Model and catalog settings
CATALOG_URL="https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
CATALOG_MODEL="scream_ne120_inst"
CURRENT_LOCATION="NERSC"

# Leave empty for automatic bounds based on RADII
MIN_LAT="-30"
MAX_LAT="30"
MIN_LON="-177"
MAX_LON="177"

# Set radius (single value for speed)
RADII="5"

# Variables to extract
VARIABLES=("omega850" "rh500")


# Define date ranges for parallel processing
DATE_RANGES=(
  "2019-09-01 2019-10-01"
  "2019-10-01 2019-11-01"
  "2019-11-01 2019-12-01"
  "2019-12-01 2020-01-01"
  "2020-01-01 2020-02-01"
  "2020-02-01 2020-03-01"
  "2020-03-01 2020-04-01"
  "2020-04-01 2020-05-01"
  "2020-05-01 2020-06-01"
  "2020-06-01 2020-07-01"
  "2020-07-01 2020-08-01"
  "2020-08-01 2020-09-01"
  "2020-09-01 2020-09-20"
)

# Generate all combinations of variables and date ranges
COMBINATIONS=()
for var in "${VARIABLES[@]}"; do
  for date_range in "${DATE_RANGES[@]}"; do
    COMBINATIONS+=("$var|$date_range")
  done
done

# Distribute tasks
START_IDX=0
END_IDX=$((${#COMBINATIONS[@]} - 1))

echo "Processing combinations $START_IDX to $END_IDX (out of ${#COMBINATIONS[@]} total combinations)"

# Maximum concurrent tasks (2 nodes × 128 cores ÷ 16 cores per task = 16)
MAX_CONCURRENT=16
RUNNING=0

# Process each combination
for ((i=START_IDX; i<=END_IDX; i++)); do
  if [ $i -lt ${#COMBINATIONS[@]} ]; then
    # Check if we've reached max concurrent jobs
    if [ $RUNNING -ge $MAX_CONCURRENT ]; then
      wait -n  # Wait for at least one job to complete
      RUNNING=$((RUNNING - 1))
    fi
    
    combo=${COMBINATIONS[$i]}
    var=${combo%%|*}
    date_range=${combo#*|}
    start_date=${date_range% *}
    end_date=${date_range#* }
    
    echo "Processing variable: $var for period $start_date to $end_date"
    
    # Create variable subdirectory
    mkdir -p "$OUTPUT_DIR/${var}"
    
    # Run extraction with optimized parameters
    srun -n 1 -c 16 --cpu_bind=cores python extract_spatial_data.py \
      --catalog_url "$CATALOG_URL" \
      --current_location "$CURRENT_LOCATION" \
      --catalog_model "$CATALOG_MODEL" \
      --catalog_params "$CATALOG_PARAMS" \
      --trackfile "$TRACK_FILE" \
      --output_dir "$OUTPUT_DIR/${var}" \
      --variable "$var" \
      --start_date "$start_date" \
      --end_date "$end_date" \
      --min_lat "$MIN_LAT" \
      --max_lat "$MAX_LAT" \
      --min_lon "$MIN_LON" \
      --max_lon "$MAX_LON" \
      --radii "$RADII" \
      --hours_before_init 12 \
      --include_evolution \
      --mcs_status_filter "0" \
      --batch_size 500 &
      
    PID=$!
    RUNNING=$((RUNNING + 1))
    echo "Started job $PID for $var ($start_date - $end_date)"
  fi
done

wait
echo "All processing complete at $(date)"
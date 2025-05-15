#!/bin/bash
# Author: Laura Paccini (laura.paccini@pnnl.gov)
# Date: May 15, 2025
# Description: Batch job script to process environmental variables around MCS tracks

#SBATCH -N 2
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 00:15:00
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
OUTPUT_DIR="/pscratch/sd/p/paccini/temp/hackathon/environmental_variables"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# ===== PARAMETERS TO CUSTOMIZE =====
# Model and catalog settings
CATALOG_URL="https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
CURRENT_LOCATION="NERSC"
CATALOG_MODEL="scream_ne120"  # Model name in the catalog
CATALOG_PARAMS='{"zoom": 8}'  # JSON string of catalog parameters

# Set spatial bounds
MIN_LAT="-30"
MAX_LAT="30"
MIN_LON="-177"
MAX_LON="177"

# Set radii for circular areas (in degrees)
RADII="5,3.5,2,0.5"

# Set track latitude/longitude variables
LAT_VAR="meanlat"
LON_VAR="meanlon"

# Processing options
REMOVE_LAND="--remove_land"  # Leave empty to include land areas
HOURS_BEFORE_INIT="24"
INCLUDE_EVOLUTION="--include_evolution"  # Leave empty to exclude evolution

# Set variables to extract 
VARIABLES=("hfssd" "tas")

OUTPUT_FORMAT="netcdf"
N_WORKERS=8  # Number of worker threads per variable

# Define date ranges for parallel processing
# Breaking down the full period into smaller chunks for parallel processing
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

# Get SLURM job array ID or default to 0 if not in array job
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TASK_COUNT=${#VARIABLES[@]}

# Generate all combinations of variables and date ranges
COMBINATIONS=()
for var in "${VARIABLES[@]}"; do
  for date_range in "${DATE_RANGES[@]}"; do
    COMBINATIONS+=("$var|$date_range")
  done
done

# Distribute tasks across nodes
START_IDX=0
END_IDX=$((${#COMBINATIONS[@]} - 1))

echo "Processing combinations $START_IDX to $END_IDX (out of ${#COMBINATIONS[@]} total combinations)"

MAX_CONCURRENT=32  # Maximum number of concurrent tasks (2 nodes × 128 cores ÷ 8 cores per task = 32)
RUNNING=0

# Create subfolder for each variable
create_var_subfolder() {
    local var=$1
    local dir="${OUTPUT_DIR}/${var}"
    mkdir -p "$dir"
    echo "Created subfolder for variable: $var"
    return 0
}

# Create subfolders for all variables
for var in "${VARIABLES[@]}"; do
    create_var_subfolder "$var"
done


# Process each assigned combination with srun
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
    
    srun -n 1 -c 8 --cpu_bind=cores python get_env_vars.py \
      --catalog_url "$CATALOG_URL" \
      --current_location "$CURRENT_LOCATION" \
      --catalog_model "$CATALOG_MODEL" \
      --catalog_params "$CATALOG_PARAMS" \
      --trackfile "$TRACK_FILE" \
      --output_dir "$OUTPUT_DIR/${var}" \
      --output_format "$OUTPUT_FORMAT" \
      --variable "$var" \
      --start_date "$start_date" \
      --end_date "$end_date" \
      --min_lat "$MIN_LAT" \
      --max_lat "$MAX_LAT" \
      --min_lon "$MIN_LON" \
      --max_lon "$MAX_LON" \
      --radii "$RADII" \
      --lat_var "$LAT_VAR" \
      --lon_var "$LON_VAR" \
      $REMOVE_LAND \
      --hours_before_init "$HOURS_BEFORE_INIT" \
      $INCLUDE_EVOLUTION \
      --n_workers "$N_WORKERS" &
      PID=$!
      RUNNING=$((RUNNING + 1))
      echo "Started job $PID for $var ($start_date - $end_date)"
  fi
done

wait
echo "All processing complete at $(date)"
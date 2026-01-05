#!/bin/bash
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu_a100

# If your server doesn't use SLURM, comment out the SBATCH lines above
# and run with: bash run.sh

# Script to run fine-tuning with proper logging
# Usage: 
#   For SLURM: sbatch run.sh
#   For direct run: bash run.sh

# Get the working directory
# For SLURM jobs, use the submission directory (where you ran sbatch from)
# For direct execution, use the script's directory
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    # Running under SLURM - use the directory where job was submitted
    WORK_DIR="$SLURM_SUBMIT_DIR"
    echo "SLURM job detected - using submission directory: $WORK_DIR"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    # SLURM job but no SLURM_SUBMIT_DIR - try to get script location
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    WORK_DIR="$SCRIPT_DIR"
    echo "SLURM job detected - using script directory: $WORK_DIR"
else
    # Direct execution - use script's directory
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    WORK_DIR="$SCRIPT_DIR"
    echo "Direct execution - using script directory: $WORK_DIR"
fi

# Change to the working directory
cd "$WORK_DIR"
# Print job information
echo "=========================================="
echo "Facebook XLM-RoBERTa Fine-tuning Job"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A (running directly)}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Working directory: $WORK_DIR"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Load modules (uncomment and adjust for your server)
# module load python/3.9
# module load cuda/11.8
# module load cudnn/8.6

# Activate conda environment if using conda
# Try common conda locations
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
fi

# Activate conda environment (uncomment and adjust the environment name)
# conda activate your_env_name
# OR create and activate a new environment:
# conda create -n xlm_finetune python=3.9 -y
# conda activate xlm_finetune
# conda install -c conda-forge pandas scikit-learn numpy -y
# pip install torch transformers datasets

# Or activate virtual environment if using venv
if [ -d "venv" ]; then
    echo "Activating virtual environment: venv"
    source venv/bin/activate
elif [ -d "$WORK_DIR/venv" ]; then
    echo "Activating virtual environment: $WORK_DIR/venv"
    source "$WORK_DIR/venv/bin/activate"
fi

# Check if required packages are installed
echo "Checking Python packages..."
MISSING_PACKAGES=()
python -c "import pandas" 2>/dev/null || MISSING_PACKAGES+=("pandas")
python -c "import torch" 2>/dev/null || MISSING_PACKAGES+=("torch")
python -c "import transformers" 2>/dev/null || MISSING_PACKAGES+=("transformers")
python -c "import datasets" 2>/dev/null || MISSING_PACKAGES+=("datasets")
python -c "import sklearn" 2>/dev/null || MISSING_PACKAGES+=("scikit-learn")
python -c "import numpy" 2>/dev/null || MISSING_PACKAGES+=("numpy")

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "ERROR: Missing required packages: ${MISSING_PACKAGES[*]}"
    echo ""
    echo "Please install packages BEFORE submitting the job:"
    echo "  Option 1 - Virtual environment (recommended):"
    echo "    cd $WORK_DIR"
    echo "    python -m venv venv"
    echo "    source venv/bin/activate"
    echo "    pip install -r requirements.txt"
    echo ""
    echo "  Option 2 - User installation:"
    echo "    pip install --user torch transformers datasets pandas scikit-learn numpy"
    echo ""
    echo "  Option 3 - Conda environment:"
    echo "    conda create -n xlm_finetune python=3.9 -y"
    echo "    conda activate xlm_finetune"
    echo "    conda install -c conda-forge pandas scikit-learn numpy -y"
    echo "    pip install torch transformers datasets"
    echo ""
    exit 1
else
    echo "✓ All required packages are available"
fi
echo ""

# Check Python and CUDA availability
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "Warning: nvidia-smi not found. Running on CPU."
fi
echo ""

# Create output directories if they don't exist
# Check if we have write permissions first
if [ ! -w "$WORK_DIR" ]; then
    echo "ERROR: No write permission in directory: $WORK_DIR"
    echo "Please run from a directory where you have write permissions."
    echo "Current directory: $(pwd)"
    echo "Working directory: $WORK_DIR"
    exit 1
fi

# Create directories with better error handling
echo "Creating output directories..."
if mkdir -p results 2>/dev/null; then
    RESULTS_DIR="results"
    echo "  ✓ Created 'results' directory"
else
    echo "  ✗ Failed to create 'results' directory"
    if [ -d "results" ] && [ ! -w "results" ]; then
        echo "    Directory exists but you don't have write permission"
        echo "    Try: chmod u+w results"
    fi
    RESULTS_DIR="."
    echo "  Using current directory for results"
fi

if mkdir -p logs 2>/dev/null; then
    LOGS_DIR="logs"
    echo "  ✓ Created 'logs' directory"
else
    echo "  ✗ Failed to create 'logs' directory"
    if [ -d "logs" ] && [ ! -w "logs" ]; then
        echo "    Directory exists but you don't have write permission"
        echo "    Try: chmod u+w logs"
    fi
    LOGS_DIR="."
    echo "  Using current directory for logs"
fi

if mkdir -p data 2>/dev/null; then
    echo "  ✓ Created 'data' directory"
elif [ -d "data" ]; then
    echo "  ✓ 'data' directory already exists"
else
    echo "  ✗ Failed to create 'data' directory"
    echo "  ERROR: Cannot proceed without 'data' directory"
    echo "  Please create it manually: mkdir -p data"
    echo "  Or ensure you have write permissions in: $WORK_DIR"
    exit 1
fi
echo ""

# Verify data files exist
# Check for single file OR pre-split files
if [ -f "data/syn_data_combined.csv" ]; then
    echo "Found single data file: data/syn_data_combined.csv"
    echo "Script will automatically split into train/val/test"
elif [ -f "data/train.csv" ] && [ -f "data/val.csv" ] && [ -f "data/test.csv" ]; then
    echo "Found pre-split data files: train.csv, val.csv, test.csv"
else
    echo "ERROR: Missing data files!"
    echo "Expected either:"
    echo "  - Single file: data/syn_data_combined.csv"
    echo "  - OR pre-split files: data/train.csv, data/val.csv, data/test.csv"
    exit 1
fi

# Set timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JOB_ID=${SLURM_JOB_ID:-$TIMESTAMP}
LOG_FILE="${LOGS_DIR}/training_output_${JOB_ID}.log"
ERROR_LOG="${LOGS_DIR}/training_errors_${JOB_ID}.log"
COMBINED_LOG="${LOGS_DIR}/training_combined_${JOB_ID}.log"

echo "Log files:"
echo "  Output: $LOG_FILE"
echo "  Errors: $ERROR_LOG"
echo "  Combined: $COMBINED_LOG"
echo ""

# Run the training script
# Redirect both stdout and stderr to combined log, and also separate them
echo "Starting training..."
python finetune_twitter_xlm.py > >(tee "$LOG_FILE") 2> >(tee "$ERROR_LOG" >&2) | tee "$COMBINED_LOG"

# Alternative: if tee doesn't work, use simpler redirection
# python finetune_twitter_xlm.py > "$LOG_FILE" 2> "$ERROR_LOG"

# Check exit status
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "=========================================="
    echo "Check $LOG_FILE for full output"
    echo "Check $ERROR_LOG for any warnings"
else
    echo "Training failed with exit code: $EXIT_CODE"
    echo "=========================================="
    echo "Check $ERROR_LOG for error details"
    echo ""
    echo "Last 30 lines of error log:"
    tail -30 "$ERROR_LOG"
fi

echo ""
echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="

exit $EXIT_CODE

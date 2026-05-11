#!/bin/bash
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu_a100

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - Modify these variables according to your setup
# ============================================================================

# Paths (adjust these to match your server setup)
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
MODEL_PATH="${MODEL_PATH:-./model}"
DATA_FOLDER="${DATA_FOLDER:-./data/qidian_freechapter_byparagraph_aligned}"
LOG_FILE="${LOG_FILE:-./logs/batch_prediction_$(date +%Y%m%d_%H%M%S).log}"

# Text column name in your CSV (default: 'text')
TEXT_COLUMN="${TEXT_COLUMN:-text}"

# Model parameters
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-128}"

# Python environment (uncomment and modify if using conda/virtualenv)
# CONDA_ENV="your_env_name"  # For conda
# VENV_PATH="./venv"  # For virtualenv

# ============================================================================
# SCRIPT START
# ============================================================================

echo "=========================================="
echo "Batch Sentiment Analysis Prediction Script"
echo "=========================================="
echo "Date: $(date)"
echo "Working directory: $PROJECT_DIR"
echo ""

# Change to project directory
cd "$PROJECT_DIR" || {
    echo "Error: Cannot change to directory $PROJECT_DIR"
    exit 1
}

# Create necessary directories
mkdir -p "$(dirname "$LOG_FILE")"

# Activate Python environment (if needed)
if [ -n "$CONDA_ENV" ]; then
    echo "Activating conda environment: $CONDA_ENV"
    # Try to initialize conda if not already initialized
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    conda activate "$CONDA_ENV" || {
        echo "Error: Failed to activate conda environment $CONDA_ENV"
        exit 1
    }
elif [ -n "$VENV_PATH" ] && [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate" || {
        echo "Error: Failed to activate virtual environment $VENV_PATH"
        exit 1
    }
fi

# Check Python version
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo ""

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import torch; import transformers; import pandas; import numpy; import tqdm" 2>/dev/null || {
    echo "Error: Required packages not found. Please install:"
    echo "  pip install torch transformers pandas numpy tqdm"
    exit 1
}
echo "Dependencies OK"
echo ""

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found: $MODEL_PATH"
    echo "Please ensure the model is in the correct location."
    exit 1
fi

# Check if data folder exists
if [ ! -d "$DATA_FOLDER" ]; then
    echo "Error: Data folder not found: $DATA_FOLDER"
    echo "Please provide a valid data folder path."
    exit 1
fi

# Display configuration
echo "Configuration:"
echo "  Model path: $MODEL_PATH"
echo "  Data folder: $DATA_FOLDER"
echo "  Text column: $TEXT_COLUMN"
echo "  Batch size: $BATCH_SIZE"
echo "  Max length: $MAX_LENGTH"
echo "  Log file: $LOG_FILE"
echo ""

# Check GPU availability (if using CUDA)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || echo "  nvidia-smi not available"
    echo ""
fi

# Run batch prediction
echo "Starting batch prediction..."
echo "=========================================="
echo ""

python batch_predict_sentiment.py \
    --data_folder "$DATA_FOLDER" \
    --text_column "$TEXT_COLUMN" \
    --model_path "$MODEL_PATH" \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --log_file "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Batch prediction completed successfully!"
    echo "Results saved to CSV files in: $DATA_FOLDER"
    echo "Log saved to: $LOG_FILE"
else
    echo "Batch prediction failed with exit code: $EXIT_CODE"
    echo "Check the log file for details: $LOG_FILE"
fi
echo "=========================================="

exit $EXIT_CODE

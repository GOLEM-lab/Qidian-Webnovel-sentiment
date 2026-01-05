#!/bin/bash
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu_a100

#!/bin/bash
# Run script for golden label evaluation on university server

# Print start message
echo "=========================================="
echo "Starting golden label evaluation..."
echo "Date: $(date)"
echo "=========================================="

# Try python3 first, fallback to python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

echo "Using Python: $PYTHON_CMD"
echo "Python version: $($PYTHON_CMD --version)"
echo ""

# Run the evaluation script
$PYTHON_CMD evaluate_golden_labels.py

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "Date: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Evaluation failed with exit code: $EXIT_CODE"
    echo "Date: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi
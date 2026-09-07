#!/bin/bash
#SBATCH --job-name=wn-infer
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/inference_webnovel_%j.out
#SBATCH --error=logs/inference_webnovel_%j.err

# Batch sentiment inference for webnovel comments + paragraphs.
#
# Input: the four CSVs staged by stage_september_data.py from september-data/,
# copied into ${PROJECT_DIR}/data/inference/webnovel/ (layout unchanged):
#
#   qidian_comments.csv      2,791,413   finetuned model, maxlen 256
#   webnovel_comments.csv      284,601   finetuned model, maxlen 256
#   qidian_paragraphs.csv      268,101   zero-shot model, maxlen 128
#   webnovel_paragraphs.csv    276,428   zero-shot model, maxlen 128
#                            ---------
#                            3,620,543 texts
#
# Comments:   fpianz/sentiment-webnovel-comments-xlmr-en-zh (finetuned)
#   Using local copy; HF equivalent: fpianz/sentiment-webnovel-comments-xlmr-en-zh
# Paragraphs: cardiffnlp/twitter-xlm-roberta-base-sentiment (zero-shot)
#
# Writes sentiment, valence, p_neg, p_neu, p_pos columns back into each CSV.

PROJECT_DIR="/home4/p306511/ondemand/data/sys/myjobs/projects/default/2"
cd "${PROJECT_DIR}/scripts"
mkdir -p logs

# ---- Environment ----
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$HOME/venvs/sentiment/bin/activate"

pip install sentencepiece -q 2>/dev/null

export HF_HOME="/scratch/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="/scratch/$USER/.cache/huggingface/hub"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# ---- Run inference ----
python3 -u run_inference_webnovel.py \
    --data_dir "${PROJECT_DIR}/data/inference/webnovel" \
    --comment_model "${PROJECT_DIR}/models/sentiment-comments-xlmr-en-ze" \
    --paragraph_model "cardiffnlp/twitter-xlm-roberta-base-sentiment" \
    --comment_maxlen 256 \
    --paragraph_maxlen 128 \
    --batch_size 256

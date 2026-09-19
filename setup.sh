#!/bin/bash

# Hentikan skrip jika ada satu perintah yang gagal
set -e

echo "🚀 [1/4] Installing system CLI tools..."
apt-get update && apt-get install -y tmux

echo "📦 [2/4] Installing Mamba pre-compiled wheels..."
pip install --no-deps https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install --no-deps https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

echo "📚 [3/4] Installing Python dependencies & gdown..."
pip install einops huggingface_hub transformers pandas scikit-learn matplotlib seaborn gdown

echo "📂 [4/4] Preparing dataset..."
mkdir -p data

if [ ! -f "data/organcmnist_224.npz" ]; then
    echo "Downloading dataset OrganCMNIST..."
    gdown 1y-CjRsE9K9MbVQMPR2nN_MHdQ9fiKGV_ -O data/organcmnist_224.npz
else
    echo "✅ Dataset OrganCMNIST sudah ada, melewatin proses unduh."
fi

echo "================================================="
echo "🎉 Setup Selesai! Environment dan Data Siap."
echo "================================================="
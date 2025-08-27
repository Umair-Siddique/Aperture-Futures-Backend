#!/usr/bin/env bash

# Exit on any error
set -e

echo "Starting build process..."

# Update package lists
echo "Updating package lists..."
apt-get update

# Install system dependencies
echo "Installing system dependencies..."
apt-get install -y \
    ffmpeg \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-setuptools \
    python3-wheel \
    python3-cffi \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    curl \
    wget \
    ca-certificates

# Clean up apt cache to reduce image size
echo "Cleaning up apt cache..."
apt-get clean
rm -rf /var/lib/apt/lists/*

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# Verify ffmpeg installation
echo "Verifying ffmpeg installation..."
ffmpeg -version

# Verify yt-dlp installation
echo "Verifying yt-dlp installation..."
yt-dlp --version

echo "Build completed successfully!"
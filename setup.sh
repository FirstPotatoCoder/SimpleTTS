#!/usr/bin/env bash
set -e

# System dependency required by phonemizer's espeak backend
apt-get -qq -y install espeak-ng

# Python dependencies
pip install -q -r requirements.txt

echo "✅ Setup complete."

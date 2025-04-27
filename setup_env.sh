#!/bin/bash

pip install torch==2.2.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo "✅ Successfully installed ttnn library"
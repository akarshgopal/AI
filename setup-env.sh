#!/bin/bash

# Create a Python virtual environment
python3 -m venv ai-bootcamp-env

# Activate the virtual environment
source ai-bootcamp-env/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Add virtual environment as a kernel to Jupyter Notebook
python -m ipykernel install --user --name=ai-bootcamp-env


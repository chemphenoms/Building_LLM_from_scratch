#!/bin/bash

# -----------------------------------------
# STEP 1: Install required Python packages
# -----------------------------------------
echo "Installing gdown..."
pip install gdown

# -----------------------------------------
# STEP 2: Download the zip file using a Python script
# -----------------------------------------
echo "Running download.py to download the dataset..."
python download.py

# Assumes download.py saves the zip file as 'downloaded_file.zip'
# If the filename is different, please update accordingly.

# -----------------------------------------
# STEP 3: Unzip the downloaded file
# -----------------------------------------
echo "Unzipping downloaded file..."
unzip downloaded_file.zip -d ./unzipped_data

# -----------------------------------------
# STEP 4: Clone the Gutenberg dataset repo
# -----------------------------------------
echo "Cloning PGCorpus Gutenberg GitHub repository..."
git clone https://github.com/pgcorpus/gutenberg.git

# -----------------------------------------
# STEP 5: Create a directory to store processed data
# -----------------------------------------
echo "Creating a directory for data output..."
mkdir -p data_dir

# -----------------------------------------
# STEP 6: Run the dataset preparation script
# -----------------------------------------
echo "Running the data preparation script..."
python prepare_dataset.py

# -----------------------------------------
# Done
# -----------------------------------------
echo "Dataset preparation complete."

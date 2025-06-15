# ✅ Step 1: Install the 'gdown' tool to download files from Google Drive
pip install gdown

# ✅ Step 2: Run the Python script 'download.py' to download the Alpaca instruction dataset
python download.py

# ✅ Step 3: Move the downloaded JSON file to the desired 'data' folder for training or preprocessing
mv instruction-data-alpaca.json ./data/

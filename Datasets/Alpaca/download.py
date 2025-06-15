import os
import json
from urllib import request

def download_and_load_file(file_path: str, url: str):
    """
    Downloads a JSON file from the specified URL if it does not already exist,
    saves it locally, and loads it into memory.

    Parameters:
        file_path (str): Local filename to save the file.
        url (str): URL to download the file from.

    Returns:
        data (dict or list): Parsed JSON data.
    """

    # Check if the file already exists to avoid re-downloading
    if not os.path.exists(file_path):
        print(f"Downloading from {url} ...")
        with request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")

        # Save downloaded data to file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
        print(f"Saved to {file_path}")
    else:
        print(f"File already exists at {file_path}")

    # Load JSON data from file
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        print(f"Loaded {len(data)} records from {file_path}")

    return data

if __name__ == "__main__":
    # Define file name and source URL
    file_path = "instruction-data-alpaca.json"
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

    # Download and load the Alpaca instruction dataset
    data = download_and_load_file(file_path, url)

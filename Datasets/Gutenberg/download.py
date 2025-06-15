import gdown

# Google Drive File ID
file_id = 'GIVE YOUR FILE ID'

# Construct the downloadable URL
url = f'https://drive.google.com/uc?id={file_id}&export=download'

# Specify local path for saving the downloaded file
output_path = './downloaded_file.zip'

# Start the download with a progress bar
gdown.download(url, output_path, quiet=False)

print(f"Download completed: {output_path}")

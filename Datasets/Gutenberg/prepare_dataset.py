import sys
import os
import argparse
from tqdm import tqdm
from gutenberg.src.cleanup import strip_headers
import re

# Check if text is predominantly English based on ASCII ratio
def is_english(text, threshold=0.9):
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) > threshold

# Combine multiple text files into a few large files
def combine_files(file_paths, target_dir, max_size_mb=500, separator="<|endoftext|>", fallback_encoding="latin1"):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    current_content = []
    current_size = 0
    file_counter = 1

    print("Combining files into larger chunks...")
    for file_path in tqdm(file_paths):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            tqdm.write(f"UnicodeDecodeError: Using fallback encoding for {file_path}")
            with open(file_path, "r", encoding=fallback_encoding) as file:
                content = file.read()

        if not is_english(content):
            tqdm.write(f"Skipping non-English file: {file_path}")
            continue

        content = strip_headers(content)
        content = re.sub(r'\n\s*\n', '\n\n', content)  # Clean multiple blank lines

        estimated_size = len(content.encode("utf-8"))

        # Save and start a new file if size exceeds limit
        if current_size + estimated_size > max_size_mb * 1024 * 1024:
            out_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
            with open(out_path, "w", encoding="utf-8") as out_file:
                out_file.write(separator.join(current_content))

            file_counter += 1
            current_content = [content]
            current_size = estimated_size
        else:
            current_content.append(content)
            current_size += estimated_size

    # Save the last remaining batch
    if current_content:
        out_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
        with open(out_path, "w", encoding="utf-8") as out_file:
            out_file.write(separator.join(current_content))

    return file_counter

# Entry point: parse arguments and process
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Gutenberg text files for LLM pretraining")

    parser.add_argument("--data_dir", type=str, default="Gutenberg/txt",
                        help="Input directory containing .txt files")
    parser.add_argument("--max_size_mb", type=int, default=500,
                        help="Maximum size (MB) of each combined file")
    parser.add_argument("--output_dir", type=str, default="data_dir",
                        help="Output directory for combined files")

    args = parser.parse_args()

    # Collect all .txt files recursively
    all_files = []
    for path, _, files in os.walk(args.data_dir):
        for name in files:
            if name.endswith(".txt"):
                all_files.append(os.path.join(path, name))

    print(f"Found {len(all_files)} text file(s) to process.")
    file_count = combine_files(all_files, args.output_dir, max_size_mb=args.max_size_mb)
    print(f"{file_count} file(s) saved in: {os.path.abspath(args.output_dir)}")


# How to run this code:
# python prepare_dataset.py --data_dir Gutenberg/txt --output_dir data_dir --max_size_mb 500

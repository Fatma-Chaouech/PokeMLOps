import os

# Define the DVC remote name
REMOTE_NAME = "my_remote"

# Define the path to the DVC file
DVC_FILE = "data/raw/data.dvc"

# Define the output path
OUTPUT_PATH = "data/raw/"

# Pull the raw data from the DVC remote
os.system(f"dvc pull -r {REMOTE_NAME} {DVC_FILE}")

# Unpack the raw data
os.system(f"dvc unpack {DVC_FILE} -o {OUTPUT_PATH}")
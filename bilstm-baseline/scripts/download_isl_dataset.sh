#!/bin/bash

# The base URL for the API request
base_url="https://zenodo.org/api/records/4010759"

# Fetch the JSON metadata from Zenodo
response=$(curl -s "$base_url")

# Parse JSON to extract file URLs and names using jq
echo "$response" | jq -r '.files[] | .links.self + " " + .key' | while read -r file_url file_name
do
  # Use curl to download each file and save it with the respective name
  echo "Downloading $file_name from $file_url..."
  curl -o "$file_name" "$file_url"
  echo "$file_name downloaded."
done

echo "All files downloaded."

# Loop through all zip files in the current directory
for file in *.zip; do
    # Unzip each file into a directory with the same name as the zip file without the extension
    unzip "${file%.zip}"
done

echo "All files unzipped."

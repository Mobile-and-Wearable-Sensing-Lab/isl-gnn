#!/bin/bash

# The root directory containing your test, train, and validation folders.
BASE_DIR="/Users/yashrb/Projects/isl_videos/dataset"

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg could not be found. Please install it to continue."
    exit
fi

echo "Starting video conversion from .MOV to .mp4..."

# Find all .MOV files (case-insensitive) and convert them
find "$BASE_DIR" -type f -iname "*.mov" | while read -r filepath; do
  # Define the output filename by replacing .MOV with .mp4
  output_filepath="${filepath%.*}.mp4"
  
  echo "Converting: $filepath"
  
  # Convert the video. -c copy is fast and preserves quality.
  # -loglevel error shows only critical errors.
  ffmpeg -i "$filepath" -c:v copy -c:a copy -loglevel error "$output_filepath"
  
  # Check if the conversion was successful
  if [ $? -eq 0 ]; then
    echo "Successfully created: $output_filepath"
    # Uncomment the line below if you want to delete the original .MOV file after conversion
    # rm "$filepath"
  else
    echo "ERROR: Failed to convert $filepath"
  fi
done

echo "Conversion process finished."

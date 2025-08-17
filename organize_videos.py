import os

base_directory = "/Users/yashrb/Projects/isl_videos/"
split = "val"
file = base_directory + f"labels/include_{split}.txt"
destination_dir = base_directory + f"data_splits/{split}/"

# read each line in the .txt file, which is a file path in itself
with open(file, "r") as f:
    video_files = f.readlines()
    video_files = [x.strip() for x in video_files]

for video in video_files:
    # check if video exists
    video_path = base_directory + f"new_dataset/{video}"
    print(video_path)
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
    else:
        mp4_video = video.replace(".MOV", ".mp4")
        destination_path = destination_dir + mp4_video
        
        # Extract the directory from the destination path
        destination_subdir = os.path.dirname(destination_path)
        
        # Create the destination directory if it doesn't exist
        if not os.path.exists(destination_subdir):
            print(f"Creating directory: {destination_subdir}")
            os.makedirs(destination_subdir)
            
        print(f"Moving {video_path} to {destination_path}")
        os.rename(video_path, destination_path)
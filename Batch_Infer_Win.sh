#!/bin/bash

# Root folder containing the directories
ROOT_FOLDER="E:\\spatial_temporal_water_seg\\trif_example\\"

# JSON file path
JSON_FILE="E:\\spatial_temporal_water_seg\\spatial_temporal_water_seg\\dataset_dirs.json"

# Loop through each sub-directory in the root folder
for dir in "$ROOT_FOLDER"/*; do
    if [ -d "$dir" ]; then
        # Extract the folder name
        # FOLDER_NAME="$dir"
        FOLDER_NAME=${dir/\//}
        # Update the JSON file
        jq --arg folder_name "$FOLDER_NAME" '.thp_timeseries = $folder_name' "$JSON_FILE" > "$JSON_FILE.tmp" && mv "$JSON_FILE.tmp" "$JSON_FILE"

        # Execute the command
        CUDA_VISIBLE_DEVICES=0 python ./PL_Support_Codes/infer.py E:\\Zhijie_PL_Pipeline\\Trained_model\\CBAM\\checkpoints\\THP_CBAM_HPC.ckpt thp_timeseries all
    fi
done
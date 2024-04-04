#!/bin/bash



# Directory path
dir_path="/media/mule/Projects/NASA/THP/Data/THP_Official/PS"

# regions=($(ls -d "$dir_path"/*/))
regions=("SPS")
# echo $regions
for region in "${regions[@]}"
do
    region_name=$(basename "$region")
    region_path="$dir_path/$region_name"
    # adjust here if you want to process certain time stamps
    # time_stamps=($(ls -d "$region_path"/*/))
    # time_stamps=("Pre-Flood" "Post-Flood")
    time_stamps=("Pre-Flood")
    for time_stamp in "${time_stamps[@]}"
    do
        # Get the paths of all directories under dir_path
        time_stamp_name=$(basename "$time_stamp")
        region_timestamp_path="$region_path/$time_stamp_name"
        # echo $region_timestamp_path
        # directories=($(find "$region_timestamp_path" -type d))
        directories=("/media/mule/Projects/NASA/THP/Data/THP_Official/PS/SPS/Pre-Flood/2020_11_11-p3")

        # Loop over the directories
        for dir in "${directories[@]}"
        do
            echo "-------------------------Start--------------------------"
            echo "Now processing:"
            echo $region_name
            echo $time_stamp_name
            echo $dir
            echo " "
            # Create a JSON object
            json_string=$(jq -n \
                            --arg v "$dir" \
                            '{thp_timeseries: $v}')

            # Write the JSON object to a file
            echo $json_string > dataset_dirs.json

            # Get the start time
            start_time=$(date +%s)

            # Execute the Python script with arguments
            # python3 ./st_water/infer.py all
            CUDA_VISIBLE_DEVICES=1 python ./st_water_seg/infer.py /home/zhijiezhang/spatial_temporal_water_seg/outputs/2023-07-21/fine_tune_low+no_PS_harmonized/checkpoints/model-epoch=16-val_MulticlassJaccardIndex=0.7844.ckpt thp_timeseries all "$region_name" "$time_stamp_name"

            # Get the end time
            end_time=$(date +%s)

            # Calculate the duration
            duration=$((end_time - start_time))

            # Print a message and the duration
            echo "Iteration for $dir completed in $duration seconds"
            echo "-------------------------END--------------------------"
            echo " "
        done
    done
done


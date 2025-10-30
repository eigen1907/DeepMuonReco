#!/usr/bin/env fish

set input_dir /home/joshin/workspace-gate/store-hdfs/DeepMuonReco/CMSSW_14_0_21_patch1/mu2024pu
set output_dir ./

# check if input directory exists
if not test -d $input_dir
    echo "Error: Input directory $input_dir does not exist."
    exit 1
end

# create output directory if it doesn't exist
if not test -d $output_dir
    mkdir -p $output_dir
end

# check if command hadd is available
if not type -q hadd
    echo "Error: hadd command not found. Please install ROOT."
    exit 1
end


set name_array "train" "val" "test"
set start_array 0 640 800
set end_array 639 799 999

# loop over 1, 2, 3
for i in 1 2
    set name $name_array[$i]
    set start $start_array[$i]
    set end $end_array[$i]
    echo "Merging files for $name from index $start to $end"

    set output_file {$output_dir}/{$name}.root

    # check if output file already exists
    echo "  - checking output file"
    if test -f $output_file
        echo "Warning: Output file $output_file already exists. Skipping merge for $name."
        continue
    end

    echo "  - checking input files"
    set input_file_array {$input_dir}/recosim-mu2024pu-TFile-(seq $start $end).root
    # for file in $input_file_array
    #     echo "file: $file"
    #     if not test -f $file
    #         echo "FileNotFoundError: $file" 2>&1
    #         exit 1
    #     end
    # end

    echo "  - merging files into $output_file"
    hadd $output_file $input_file_array
end

#!/usr/bin/env bash

localsrc=${1}
dst=${2}
# check if the number of arguments is correct. If not, print usage and exit
if [ $# -ne 2 ]; then
    >&2 echo "Usage: $0 <local_source_file> <hdfs_destination_path>"
    exit 1
fi

echo "localsrc=${localsrc}"
echo "dst=${dst}"

# if dst startswith /hdfs, remove the /hdfs prefix
if [[ ${dst} == /hdfs* ]]; then
    dst=${dst:5}
    echo "dst modified to ${dst}"
fi

file_size=$(stat -c %s ${localsrc})
echo "file_size: ${file_size}"

block_size=$(( (file_size + 511) / 512 * 512 ))
if [ ${block_size} -lt 1048576 ]; then
    block_size=1048576
fi
echo "block_size: ${block_size}"

hdfs dfs -Ddfs.block.size=${block_size} -put ${localsrc} ${dst}


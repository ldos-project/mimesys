username="$1"
remote_host="$2"
remote_destination="$3"  # Remote destination directory for results
chunk_idx="$4"  # Chunk index for processing
host_name=$(hostname)

fleetbench_dir="/users/$username/fleetbench"

cd $fleetbench_dir
bash collect_mimesys_data.sh

cd $HOME
# Add any batch-specific processing logic here
target_dir=validation-$chunk_idx

zip -r "$target_dir.zip" results
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$target_dir.zip" $username@$remote_host:$remote_destination/$host_name-$target_dir.zip
sudo rm results/*
rm -rf $target_dir.zip

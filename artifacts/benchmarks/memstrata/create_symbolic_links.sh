for dir in /home/ubuntu/shared/*; do
  ln -s "$dir" "$(basename "$dir")"
done

cd /home/ubuntu
rm -f result_*
cd /home/ubuntu/Private-Pond/gapbs/cc-web
sed -i 's|export OMP_NUM_THREADS=.*|export OMP_NUM_THREADS=8|' ./cmd.sh

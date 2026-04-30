cd /home/ubuntu
rm -f result_*
cd /home/ubuntu/Private-Pond/gapbs/bc-urand
sed -i 's|export OMP_NUM_THREADS=.*|export OMP_NUM_THREADS=8|' ./cmd.sh

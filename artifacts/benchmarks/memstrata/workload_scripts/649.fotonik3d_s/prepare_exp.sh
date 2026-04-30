cd /home/ubuntu
rm -f result_*
cd /home/ubuntu/Private-Pond/cpu2017/649.fotonik3d_s
sed -i 's|export OMP_NUM_THREADS=.*|export OMP_NUM_THREADS=4|' ./cmd.sh

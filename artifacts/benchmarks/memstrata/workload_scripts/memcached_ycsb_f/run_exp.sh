cd /home/ubuntu/YCSB
python2 ./bin/ycsb run memcached -s -P workloads/workloadf -p "memcached.hosts=127.0.0.1" -p "threadcount=2" >> /home/ubuntu/result_app_perf.txt 2>> /home/ubu\
ntu/result_app_perf.txt

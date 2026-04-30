cd /home/ubuntu/YCSB
python2 ./bin/ycsb run redis -s -P workloads/workloadc -p "redis.host=127.0.0.1" -p "redis.port=6379" -p "threadcount=3" >> /home/ubuntu/result_app_perf.txt 2>> /home/ubuntu/result_app_perf.txt

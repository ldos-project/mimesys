cd /home/ubuntu/silo
/usr/bin/time -vao /home/ubuntu/result_time.txt ./out-perf.masstree/benchmarks/dbtest \
    --verbose \
    --bench tpcc \
    --num-threads 1 \
    --scale-factor 1 \
    --runtime 300 >> /home/ubuntu/result_app_perf.txt 2>> /home/ubuntu/result_app_perf.txt



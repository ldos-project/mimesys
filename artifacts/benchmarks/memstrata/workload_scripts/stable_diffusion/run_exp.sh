/home/ubuntu/.local/bin/mlcr run-mlperf,inference,_find-performance,_full,_r5.1-dev \
   --model=sdxl \
   --implementation=reference \
   --framework=pytorch \
   --category=edge \
   --scenario=Offline \
   --execution_mode=test \
   --device=cpu  \
   --test_query_count=10 --rerun >> /home/ubuntu/result_app_perf.txt 2>> /home/ubuntu/result_app_perf.txt

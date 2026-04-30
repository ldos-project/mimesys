/home/ubuntu/.local/bin/mlcr run-mlperf,inference,_find-performance,_full,_r5.1-dev \
   --model=resnet50 \
   --implementation=reference \
   --framework=onnxruntime \
   --category=datacenter \
   --scenario=Offline \
   --device=cpu  \
   --execution_mode=test \
   --test_query_count=1000 --rerun >> /home/ubuntu/result_app_perf.txt 2>> /home/ubuntu/result_app_perf.txt

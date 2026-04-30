cd /home/ubuntu
touch result_app_perf.txt
cd /home/ubuntu/pg-tpch
./tpch_runone 22
cat /home/ubuntu/pg-tpch/perfdata-10GB/q22/exectime.txt >> /home/ubuntu/result_app_perf.txt

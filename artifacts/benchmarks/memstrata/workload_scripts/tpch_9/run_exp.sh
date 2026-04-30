cd /home/ubuntu
touch result_app_perf.txt
cd /home/ubuntu/pg-tpch
./tpch_runone 9
cat /home/ubuntu/pg-tpch/perfdata-10GB/q09/exectime.txt >> /home/ubuntu/result_app_perf.txt

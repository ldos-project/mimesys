cd /home/ubuntu
rm -f result_*
cd /home/ubuntu/HiBench
rm -f report/hibench.report

yes "Y" | ~/hadoop/bin/hdfs namenode -format
rm -rf /home/ubuntu/hdfs/datanode/*

~/hadoop/sbin/start-dfs.sh
~/hadoop/sbin/start-yarn.sh
~/spark/sbin/start-master.sh

bin/workloads/ml/svm/prepare/prepare.sh

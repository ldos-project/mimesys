cd /home/ubuntu
rm -f result_*

cd YCSB
sed -i 's/recordcount=.*/recordcount=5000000/' workloads/workloadf
sed -i 's/operationcount=.*/operationcount=5000000/' workloads/workloadf
sed -i 's/requestdistribution=.*/requestdistribution=zipfian/' workloads/workloadf

sudo sed -i 's/^-m [0-9]*/-m 8192/' /etc/memcached.conf
sudo systemctl restart memcached.service

python2 ./bin/ycsb load memcached -s -P workloads/workloadf -p "memcached.hosts=127.0.0.1" -p "threadcount=2"

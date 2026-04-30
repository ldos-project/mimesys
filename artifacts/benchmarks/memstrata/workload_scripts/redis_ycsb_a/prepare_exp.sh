cd /home/ubuntu
rm -f result_*

while [ ! -z "$(pgrep -nf redis)" ]; do
    sudo kill $(pgrep -fn redis)
    sleep 1
done

cd redis
src/redis-server redis.conf > /dev/null 2> /dev/null < /dev/null & 
cd ..
sleep 5

cd YCSB
sed -i 's/recordcount=.*/recordcount=5000000/' workloads/workloada
sed -i 's/operationcount=.*/operationcount=15000000/' workloads/workloada
sed -i 's/requestdistribution=.*/requestdistribution=zipfian/' workloads/workloada

python2 ./bin/ycsb load redis -s -P workloads/workloada -p "redis.host=127.0.0.1" -p "redis.port=6379" -p "threadcount=3"

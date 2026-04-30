sudo docker stop $(sudo docker ps -aq)
sudo docker rm $(sudo docker ps -aq)

cd /home/ubuntu/DeathStarBench/socialNetwork
sudo docker-compose up -d
python3 scripts/init_social_graph.py --graph=socfb-Reed98

../wrk2/wrk -D exp -t 1 -c 1 -d 300 -L -s \
    ./wrk2/scripts/social-network/mixed-workload.lua \
    http://localhost:8080/wrk2-api/home-timeline/read \
    http://localhost:8080/wrk2-api/user-timeline/read \
    http://localhost:8080/wrk2-api/post/compose \
    -R 4000 >> /home/ubuntu/result_app_perf.txt 2>> /home/ubuntu/result_app_perf.txt

sudo docker stop $(sudo docker ps -aq)
sudo docker rm $(sudo docker ps -aq)
sudo docker volume prune -f

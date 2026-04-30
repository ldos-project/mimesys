# stop all containers
sudo docker stop $(sudo docker ps -aq)
sudo docker rm $(sudo docker ps -aq)

cd /home/ubuntu/DeathStarBench/mediaMicroservices
sudo docker-compose up -d
python3 scripts/write_movie_info.py -c datasets/tmdb/casts.json -m datasets/tmdb/movies.json
scripts/register_users.sh
scripts/register_movies.sh

../wrk2/wrk -D exp -t 1 -c 1 -d 300 \
    -L -s ./wrk2/scripts/media-microservices/compose-review.lua \
    http://localhost:8080/wrk2-api/review/compose \
    -R 200 >> /home/ubuntu/result_app_perf.txt 2>> /home/ubuntu/result_app_perf.txt

# stop all containers
sudo docker stop $(sudo docker ps -aq)
sudo docker rm $(sudo docker ps -aq)
sudo docker volume prune -f

#!/bin/bash
# Install DeathStarBench (socialNetwork + mediaMicroservices).
set -eu -o pipefail
source "$(dirname "$0")/_common.sh"

sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
    libssl-dev libz-dev luarocks lua-socket

sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-linux-x86_64" \
    -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo python3 -m pip install asyncio aiohttp
sudo luarocks install luasocket

cd "$BASE_PATH"
if [ ! -d "DeathStarBench" ]; then
    git clone https://github.com/delimitrou/DeathStarBench.git
fi
cd DeathStarBench
git submodule update --init --recursive

# socialNetwork
cd socialNetwork && sudo docker-compose up -d
cd ../wrk2 && make
cd ../socialNetwork
sudo luarocks install luasocket
sudo docker stop $(sudo docker ps -aq) 2>/dev/null || true
sudo docker rm   $(sudo docker ps -aq) 2>/dev/null || true

# mediaMicroservices
cd "$BASE_PATH/DeathStarBench/mediaMicroservices" && sudo docker-compose up -d
cd ../wrk2 && make
cd ../mediaMicroservices
sudo docker stop $(sudo docker ps -aq) 2>/dev/null || true
sudo docker rm   $(sudo docker ps -aq) 2>/dev/null || true
sudo docker volume prune -f

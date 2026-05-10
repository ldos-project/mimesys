#!/bin/bash
# Install DaCapo benchmark suite + Renaissance JAR (for finagle workloads).
set -eu -o pipefail
source "$(dirname "$0")/_common.sh"

cd "$BASE_PATH"
if [ ! -f "renaissance-gpl-0.14.2.jar" ]; then
    wget https://github.com/renaissance-benchmarks/renaissance/releases/download/v0.14.2/renaissance-gpl-0.14.2.jar
fi

sudo apt-get install -y \
    default-jre default-jdk openjdk-8-jre openjdk-8-jdk \
    ant cvs subversion nodejs npm python3 python3-pip

sudo python3 -m pip install colorama future tabulate requests wheel

cd "$BASE_PATH"
if [ ! -d "dacapobench" ]; then
    git clone https://github.com/dacapobench/dacapobench.git
fi
cd dacapobench
git checkout tags/v23.10-RC4-chopin

cd benchmarks
grep -q '^jdk.8.home' default.properties \
    || echo "jdk.8.home=/usr/lib/jvm/java-8-openjdk-amd64" >> default.properties
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

[ -L /bin/python ] || sudo ln -sf /bin/python3 /bin/python

sed -i 's|<target name="compile" depends=.*>|<target name="compile" depends="cassandra,kafka,spring,tomcat">|' build.xml
sed -i 's|<property name="lib-url" .*>|<property name="lib-url" value="https://archive.apache.org/dist/commons/logging/source/"/>|' libs/commons-logging/build.xml
ant

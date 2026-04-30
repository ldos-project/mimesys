set -eu -o pipefail

sed -i '/^case $- in$/,/^esac$/d' ~/.bashrc
sed -i '1i force_color_prompt=""' ~/.bashrc
sed -i '1i color_prompt=""' ~/.bashrc

sudo apt update
sudo apt-get install memcached -y

BASE_PATH=$HOME/shared

APP_TYPE=${1:-}


if [ ! -d "Private-Pond" ]; then
    cd $BASE_PATH
    git clone https://github.com/MoatLab/Pond.git
    mv Pond Private-Pond
    cd Private-Pond
    git checkout e9ae753669f98497f36c9ba52525e062515c5bf1
fi

if [[ "$APP_TYPE" == "database" ]]; then
    # silo
    sudo apt install libjemalloc-dev libdb++-dev build-essential libaio-dev libnuma-dev libssl-dev zlib1g-dev autoconf -y
    cd $BASE_PATH
    if [ ! -d "silo" ]; then
        git clone https://github.com/yuhong-zhong/silo.git
    fi
    cd silo

    MODE=perf make -j dbtest
fi

if [[ "$APP_TYPE" == "bigdata" ]]; then
    # spark
    sudo apt install openjdk-8-jre-headless openjdk-8-jdk-headless -y
    sudo apt-get install ssh pdsh bc scala python2 -y

    cd $BASE_PATH
    echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> ~/.bashrc
    source ~/.bashrc
    sudo bash -c "echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> /etc/environment"
    source /etc/environment

    cd $BASE_PATH

    if [ ! -d "hadoop" ]; then
        wget https://dlcdn.apache.org/hadoop/common/hadoop-3.2.4/hadoop-3.2.4.tar.gz
        tar -xzvf hadoop-3.2.4.tar.gz
        mv hadoop-3.2.4 hadoop
        cd hadoop

        sed -i "s|<configuration>||" etc/hadoop/core-site.xml
        sed -i "s|</configuration>||" etc/hadoop/core-site.xml
        echo "<configuration>" >> etc/hadoop/core-site.xml
        echo "<property>" >> etc/hadoop/core-site.xml
        echo "<name>fs.defaultFS</name>" >> etc/hadoop/core-site.xml
        echo "<value>hdfs://localhost:8020</value>" >> etc/hadoop/core-site.xml
        echo "</property>" >> etc/hadoop/core-site.xml
        echo "</configuration>" >> etc/hadoop/core-site.xml

        sed -i "s|<configuration>||" etc/hadoop/hdfs-site.xml
        sed -i "s|</configuration>||" etc/hadoop/hdfs-site.xml
        cat >> etc/hadoop/hdfs-site.xml <<- End
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>/home/ubuntu/hdfs/datanode</value>
    </property>
</configuration>
End

        mkdir -p /home/ubuntu/hdfs/datanode
        chmod -R 777 /home/ubuntu/hdfs/datanode

        ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
        cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
        chmod 0600 ~/.ssh/authorized_keys

        # Formate the filesystem
        bin/hdfs namenode -format

        echo 'export PDSH_RCMD_TYPE=ssh' >> ~/.bashrc
        source ~/.bashrc

        sed -i "s|<configuration>||" etc/hadoop/mapred-site.xml
        sed -i "s|</configuration>||" etc/hadoop/mapred-site.xml
        HADOOP_MAPRED_HOME='$HADOOP_MAPRED_HOME'
        cat >> etc/hadoop/mapred-site.xml <<- End
<configuration>
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
    <property>
        <name>yarn.app.mapreduce.am.env</name>
        <value>HADOOP_MAPRED_HOME=/home/ubuntu/hadoop</value>
    </property>
    <property>
        <name>mapreduce.map.env</name>
        <value>HADOOP_MAPRED_HOME=/home/ubuntu/hadoop</value>
    </property>
    <property>
        <name>mapreduce.reduce.env</name>
        <value>HADOOP_MAPRED_HOME=/home/ubuntu/hadoop</value>
    </property>
    <property>
        <name>mapreduce.application.classpath</name>
        <value>$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/*,$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/lib/*,$HADOOP_MAPRED_HOME/share/hadoop/common/*,$HADOOP_MAPRED_HOME/share/hadoop/common/lib/*,$HADOOP_MAPRED_HOME/share/hadoop/yarn/*,$HADOOP_MAPRED_HOME/share/hadoop/yarn/lib/*,$HADOOP_MAPRED_HOME/share/hadoop/hdfs/*,$HADOOP_MAPRED_HOME/share/hadoop/hdfs/lib/*</value>
    </property>
</configuration>
End

        sed -i "s|<configuration>||" etc/hadoop/yarn-site.xml
        sed -i "s|</configuration>||" etc/hadoop/yarn-site.xml
        cat >> etc/hadoop/yarn-site.xml <<- End
<configuration>
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>
    <property>
        <name>yarn.nodemanager.env-whitelist</name>
        <value>JAVA_HOME,HADOOP_COMMON_HOME,HADOOP_HDFS_HOME,HADOOP_CONF_DIR,CLASSPATH_PREPEND_DISTCACHE,HADOOP_YARN_HOME,HADOOP_HOME,PATH,LANG,TZ,HADOOP_MAPRED_HOME</value>
    </property>
</configuration>
End
    else
        cd hadoop
        mkdir -p /home/ubuntu/hdfs/datanode
        chmod -R 777 /home/ubuntu/hdfs/datanode

        ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
        cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
        chmod 0600 ~/.ssh/authorized_keys

        # Formate the filesystem
        bin/hdfs namenode -format

        echo 'export PDSH_RCMD_TYPE=ssh' >> ~/.bashrc
        source ~/.bashrc
    fi

    cd $BASE_PATH
    if [ ! -d "spark" ]; then
        wget https://archive.apache.org/dist/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz
        tar -xzvf spark-2.4.0-bin-hadoop2.7.tgz
        mv spark-2.4.0-bin-hadoop2.7 spark
    fi

    cd spark
    echo "export SPARK_HOME=$(pwd)" >> ~/.bashrc
    echo 'export PATH=$PATH:$SPARK_HOME/bin' >> ~/.bashrc
    source ~/.bashrc

    cd $BASE_PATH
    if [ ! -d "apache-maven" ]; then
        wget https://downloads.apache.org/maven/maven-3/3.9.11/binaries/apache-maven-3.9.11-bin.tar.gz
        tar -xzvf apache-maven-3.9.11-bin.tar.gz
        mv apache-maven-3.9.11 apache-maven
    fi
    cd apache-maven
    echo "export M2_HOME=$(pwd)" >> ~/.bashrc
    echo 'export PATH=$PATH:$M2_HOME/bin' >> ~/.bashrc
    source ~/.bashrc

    cd $BASE_PATH

    if [ ! -d "HiBench" ]; then
        wget https://github.com/Intel-bigdata/HiBench/archive/refs/tags/v7.1.1.tar.gz
        tar -xzvf v7.1.1.tar.gz
        mv HiBench-7.1.1 HiBench
    fi
    cd HiBench

    cp hadoopbench/nutchindexing/pom.xml hadoopbench/nutchindexing/pom.xml.bak
    cat hadoopbench/nutchindexing/pom.xml \
        | sed 's|<url>http://archive.apache.org/dist/nutch/apache-nutch-1.2-bin.tar.gz</url>|<url>https://www.apache.org/dyn/closer.cgi/nutch/apache-nutch-1.2-bin.tar.gz</url>|' \
        > ./pom.xml.tmp
    mv ./pom.xml.tmp hadoopbench/nutchindexing/pom.xml

    cp hadoopbench/sql/pom.xml hadoopbench/sql/pom.xml.bak
    cat hadoopbench/sql/pom.xml \
        | sed 's|<repo>http://archive.apache.org</repo>|<repo>https://www.apache.org/dyn/closer.cgi</repo>|' \
        | sed 's|<file>dist/hive|<file>hive|' \
        > ./pom.xml.tmp
    mv ./pom.xml.tmp hadoopbench/sql/pom.xml

    cp hadoopbench/mahout/pom.xml hadoopbench/mahout/pom.xml.bak
    cat hadoopbench/mahout/pom.xml \
        | sed 's|<repo1>http://archive.apache.org</repo1>|<repo1>https://www.apache.org/dyn/closer.cgi</repo1>|' \
        | sed 's|<file1>dist/mahout|<file1>mahout|' \
        | sed 's|<checksum1>32bb8d9429671c651ff8233676739f1f</checksum1>|<checksum1>19a1ef32c1580d27a3ac14749d0f515a</checksum1>|' \
        | sed 's|<repo2>http://archive.cloudera.com</repo2>|<repo2>https://www.apache.org/dyn/closer.cgi</repo2>|' \
        | sed 's|cdh5/cdh/5/mahout-0.9-cdh5.1.0.tar.gz|mahout/0.9/mahout-distribution-0.9.tar.gz|' \
        | sed 's|aa953e0353ac104a22d314d15c88d78f|c2dac576f5ee3e16ff74bda9aec8b816|' \
        > ./pom.xml.tmp
    mv ./pom.xml.tmp hadoopbench/mahout/pom.xml

    mvn -Phadoopbench -Psparkbench -Dspark=2.4 -Dscala=2.11 clean package

    cp conf/hadoop.conf.template conf/hadoop.conf
    sed -i "s|^hibench.hadoop.home.*|hibench.hadoop.home /home/ubuntu/hadoop|" conf/hadoop.conf
    echo "hibench.hadoop.examples.jar /home/ubuntu/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.2.4.jar" >> conf/hadoop.conf

    cp conf/spark.conf.template conf/spark.conf
    sed -i "s|hibench.spark.home.*|hibench.spark.home /home/ubuntu/spark|" conf/spark.conf
    sed -i "s|hibench.yarn.executor.num.*|hibench.yarn.executor.num 2|" conf/spark.conf
    sed -i "s|hibench.yarn.executor.cores.*|hibench.yarn.executor.cores 2|" conf/spark.conf
    sed -i "s|spark.executor.memory.*|spark.executor.memory 2g|" conf/spark.conf
    sed -i "s|spark.driver.memory.*|spark.driver.memory 2g|" conf/spark.conf

    echo "hibench.masters.hostnames localhost" >> conf/spark.conf
    echo "hibench.slaves.hostnames localhost" >> conf/spark.conf

    sed -i "s|hibench.scale.profile.*|hibench.scale.profile large|" conf/hibench.conf
fi


if [[ "$APP_TYPE" == "kvstore" ]]; then
    # faster
    sudo apt install libtbb-dev build-essential pkg-config -y
    sudo apt install libaio-dev libaio1 uuid-dev libnuma-dev cmake -y
    sudo apt-get install default-jre default-jdk openjdk-8-jre openjdk-8-jdk -y
    sudo apt-get install ssh pdsh bc scala python2 -y

    cd $BASE_PATH
    if [ ! -d "FASTER" ]; then
        git clone https://github.com/yuhong-zhong/FASTER.git
    fi
    cd FASTER/cc
    mkdir -p build/Release
    cd build/Release
    cmake -DCMAKE_BUILD_TYPE=Release ../..
    make pmem_benchmark

    # redis
    cd $BASE_PATH
    if [ ! -d "redis" ]; then
        wget https://github.com/redis/redis/archive/refs/tags/7.2.3.tar.gz
        tar -xzvf 7.2.3.tar.gz
        mv redis-7.2.3 redis
    fi
    cd redis
    make -j8
    sudo bash -c "echo 'vm.overcommit_memory=1' >> /etc/sysctl.conf"
    sed -i -e '$a save ""' redis.conf

    # maven
    cd $BASE_PATH
    if [ ! -d "apache-maven" ]; then
        wget https://downloads.apache.org/maven/maven-3/3.9.11/binaries/apache-maven-3.9.11-bin.tar.gz
        tar -xzvf apache-maven-3.9.11-bin.tar.gz
        mv apache-maven-3.9.11 apache-maven
    fi
    cd apache-maven
    echo "export M2_HOME=$(pwd)" >> ~/.bashrc
    echo 'export PATH=$PATH:$M2_HOME/bin' >> ~/.bashrc
    source ~/.bashrc

    # ycsb
    cd $BASE_PATH
    if [ ! -d "YCSB" ]; then
        git clone https://github.com/brianfrankcooper/YCSB.git
    fi
    cd YCSB
    mvn -pl site.ycsb:redis-binding -am clean package
    mvn -pl site.ycsb:memcached-binding -am clean package
fi

if [[ "$APP_TYPE" == "web" ]]; then
    # renaissance
    cd $BASE_PATH

    if [ ! -f "renaissance-gpl-0.14.2.jar" ]; then
        wget https://github.com/renaissance-benchmarks/renaissance/releases/download/v0.14.2/renaissance-gpl-0.14.2.jar
    fi

    # dacapo
    sudo apt-get install default-jre default-jdk openjdk-8-jre openjdk-8-jdk -y
    sudo apt-get install ant cvs subversion nodejs npm python3 python3-pip -y

    cd $BASE_PATH
    sudo python3 -m pip install colorama future tabulate requests wheel

    if [ ! -d "dacapobench" ]; then
        git clone https://github.com/dacapobench/dacapobench.git
    fi
    cd dacapobench
    git checkout tags/v23.10-RC4-chopin

    cd benchmarks
    echo "" >> default.properties
    echo "jdk.8.home=/usr/lib/jvm/java-8-openjdk-amd64" >> default.properties
    export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

    sudo rm -f /bin/python
    sudo ln -s /bin/python3 /bin/python

    # sed -i 's|<target name="compile" depends=.*>|<target name="compile" depends="cassandra,kafka,spring,tomcat,luindex,lusearch">|' build.xml
    sed -i 's|<target name="compile" depends=.*>|<target name="compile" depends="cassandra,kafka,spring,tomcat">|' build.xml
    sed -i 's|<property name="lib-url" .*>|<property name="lib-url" value="https://archive.apache.org/dist/commons/logging/source/"/>|' libs/commons-logging/build.xml
    ant
fi

# deathstarbench
if [[ "$APP_TYPE" == "deathstarbench" ]]; then
    cd $BASE_PATH
    sudo apt-get update
    sudo apt-get install ca-certificates curl gnupg -y

    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
    "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose

    sudo apt-get install libssl-dev libz-dev luarocks lua-socket -y
    sudo python3 -m pip install asyncio aiohttp

    sudo luarocks install luasocket

    if [ ! -d "DeathStarBench" ]; then
        git clone https://github.com/delimitrou/DeathStarBench.git
    fi
    cd DeathStarBench
    git submodule update --init --recursive
    cd socialNetwork
    sudo docker-compose up -d

    cd ../wrk2
    make
    cd ../socialNetwork

    sudo luarocks install luasocket

    sudo docker stop $(sudo docker ps -aq)
    sudo docker rm $(sudo docker ps -aq)

    cd $BASE_PATH
    cd DeathStarBench/mediaMicroservices/
    sudo docker-compose up -d

    cd ../wrk2
    make
    cd ../mediaMicroservices

    sudo docker stop $(sudo docker ps -aq)
    sudo docker rm $(sudo docker ps -aq)
    sudo docker volume prune -f
fi

if [[ "$APP_TYPE" == "ml" ]]; then
    sudo apt-get install python3-pip python3-venv llvm libgl1-mesa-glx -y
    pip install mlc-scripts
    pip install networkx fsspec sympy
    /home/ubuntu/.local/bin/mlcr install,python-venv --name=mlperf
    export MLC_SCRIPT_EXTRA_CMD="--adr.python.name=mlperf"
    yes "" | /home/ubuntu/.local/bin/mlcr run-mlperf,inference,_find-performance,_full,_r5.1-dev \
    --model=sdxl \
    --implementation=reference \
    --framework=pytorch \
    --category=datacenter \
    --scenario=Offline \
    --device=cpu \
    --actions=download,preprocess

    yes "" | /home/ubuntu/.local/bin/mlcr run-mlperf,inference,_find-performance,_full,_r5.1-dev \
   --model=resnet50 \
   --implementation=reference \
   --framework=onnxruntime \
   --category=datacenter \
   --scenario=Offline \
   --execution_mode=test \
   --device=cpu  \
    --actions=download,preprocess
fi

if [[ "$APP_TYPE" == "graph" ]]; then
    sudo apt install build-essential -y
    cd $BASE_PATH
    git clone https://github.com/sbeamer/gapbs.git
    cd gapbs
    make
fi

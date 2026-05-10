#!/bin/bash
# Install Hadoop 3.2.4 + Spark 2.4.0 + HiBench 7.1.1.
set -eu -o pipefail
source "$(dirname "$0")/_common.sh"

sudo apt-get install -y \
    openjdk-8-jre-headless openjdk-8-jdk-headless \
    ssh pdsh bc scala python2

echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> ~/.bashrc
sudo bash -c "echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> /etc/environment"
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

cd "$BASE_PATH"
if [ ! -d "hadoop" ]; then
    wget https://dlcdn.apache.org/hadoop/common/hadoop-3.2.4/hadoop-3.2.4.tar.gz
    tar -xzvf hadoop-3.2.4.tar.gz
    mv hadoop-3.2.4 hadoop

    cd hadoop

    # core-site.xml
    sed -i "s|<configuration>||"  etc/hadoop/core-site.xml
    sed -i "s|</configuration>||" etc/hadoop/core-site.xml
    cat >> etc/hadoop/core-site.xml <<'EOF'
<configuration>
<property>
<name>fs.defaultFS</name>
<value>hdfs://localhost:8020</value>
</property>
</configuration>
EOF

    # hdfs-site.xml
    sed -i "s|<configuration>||"  etc/hadoop/hdfs-site.xml
    sed -i "s|</configuration>||" etc/hadoop/hdfs-site.xml
    cat >> etc/hadoop/hdfs-site.xml <<'EOF'
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
EOF

    mkdir -p /home/ubuntu/hdfs/datanode
    chmod -R 777 /home/ubuntu/hdfs/datanode
    [ -f ~/.ssh/id_rsa ] || ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
    chmod 0600 ~/.ssh/authorized_keys

    bin/hdfs namenode -format
    echo 'export PDSH_RCMD_TYPE=ssh' >> ~/.bashrc

    sed -i "s|<configuration>||"  etc/hadoop/mapred-site.xml
    sed -i "s|</configuration>||" etc/hadoop/mapred-site.xml
    cat >> etc/hadoop/mapred-site.xml <<'EOF'
<configuration>
    <property><name>mapreduce.framework.name</name><value>yarn</value></property>
    <property><name>yarn.app.mapreduce.am.env</name><value>HADOOP_MAPRED_HOME=/home/ubuntu/hadoop</value></property>
    <property><name>mapreduce.map.env</name><value>HADOOP_MAPRED_HOME=/home/ubuntu/hadoop</value></property>
    <property><name>mapreduce.reduce.env</name><value>HADOOP_MAPRED_HOME=/home/ubuntu/hadoop</value></property>
    <property><name>mapreduce.application.classpath</name><value>$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/*,$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/lib/*,$HADOOP_MAPRED_HOME/share/hadoop/common/*,$HADOOP_MAPRED_HOME/share/hadoop/common/lib/*,$HADOOP_MAPRED_HOME/share/hadoop/yarn/*,$HADOOP_MAPRED_HOME/share/hadoop/yarn/lib/*,$HADOOP_MAPRED_HOME/share/hadoop/hdfs/*,$HADOOP_MAPRED_HOME/share/hadoop/hdfs/lib/*</value></property>
</configuration>
EOF

    sed -i "s|<configuration>||"  etc/hadoop/yarn-site.xml
    sed -i "s|</configuration>||" etc/hadoop/yarn-site.xml
    cat >> etc/hadoop/yarn-site.xml <<'EOF'
<configuration>
    <property><name>yarn.nodemanager.aux-services</name><value>mapreduce_shuffle</value></property>
    <property><name>yarn.nodemanager.env-whitelist</name><value>JAVA_HOME,HADOOP_COMMON_HOME,HADOOP_HDFS_HOME,HADOOP_CONF_DIR,CLASSPATH_PREPEND_DISTCACHE,HADOOP_YARN_HOME,HADOOP_HOME,PATH,LANG,TZ,HADOOP_MAPRED_HOME</value></property>
</configuration>
EOF
fi

cd "$BASE_PATH"
if [ ! -d "spark" ]; then
    wget https://archive.apache.org/dist/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz
    tar -xzvf spark-2.4.0-bin-hadoop2.7.tgz
    mv spark-2.4.0-bin-hadoop2.7 spark
fi
echo "export SPARK_HOME=$BASE_PATH/spark"           >> ~/.bashrc
echo 'export PATH=$PATH:$SPARK_HOME/bin'            >> ~/.bashrc

cd "$BASE_PATH"
if [ ! -d "apache-maven" ]; then
    wget https://downloads.apache.org/maven/maven-3/3.9.11/binaries/apache-maven-3.9.11-bin.tar.gz
    tar -xzvf apache-maven-3.9.11-bin.tar.gz
    mv apache-maven-3.9.11 apache-maven
fi
echo "export M2_HOME=$BASE_PATH/apache-maven"       >> ~/.bashrc
echo 'export PATH=$PATH:$M2_HOME/bin'               >> ~/.bashrc
export M2_HOME="$BASE_PATH/apache-maven"
export PATH="$PATH:$M2_HOME/bin"

cd "$BASE_PATH"
if [ ! -d "HiBench" ]; then
    wget https://github.com/Intel-bigdata/HiBench/archive/refs/tags/v7.1.1.tar.gz
    tar -xzvf v7.1.1.tar.gz
    mv HiBench-7.1.1 HiBench
fi
cd HiBench

# Patch broken upstream URLs (preserved from the original installer).
for f in hadoopbench/nutchindexing/pom.xml hadoopbench/sql/pom.xml hadoopbench/mahout/pom.xml; do
    [ -f "$f.bak" ] || cp "$f" "$f.bak"
done
sed -i 's|<url>http://archive.apache.org/dist/nutch/apache-nutch-1.2-bin.tar.gz</url>|<url>https://www.apache.org/dyn/closer.cgi/nutch/apache-nutch-1.2-bin.tar.gz</url>|' hadoopbench/nutchindexing/pom.xml
sed -i 's|<repo>http://archive.apache.org</repo>|<repo>https://www.apache.org/dyn/closer.cgi</repo>|;s|<file>dist/hive|<file>hive|' hadoopbench/sql/pom.xml
sed -i 's|<repo1>http://archive.apache.org</repo1>|<repo1>https://www.apache.org/dyn/closer.cgi</repo1>|;s|<file1>dist/mahout|<file1>mahout|;s|<checksum1>32bb8d9429671c651ff8233676739f1f</checksum1>|<checksum1>19a1ef32c1580d27a3ac14749d0f515a</checksum1>|;s|<repo2>http://archive.cloudera.com</repo2>|<repo2>https://www.apache.org/dyn/closer.cgi</repo2>|;s|cdh5/cdh/5/mahout-0.9-cdh5.1.0.tar.gz|mahout/0.9/mahout-distribution-0.9.tar.gz|;s|aa953e0353ac104a22d314d15c88d78f|c2dac576f5ee3e16ff74bda9aec8b816|' hadoopbench/mahout/pom.xml

mvn -Phadoopbench -Psparkbench -Dspark=2.4 -Dscala=2.11 clean package

cp conf/hadoop.conf.template conf/hadoop.conf
sed -i "s|^hibench.hadoop.home.*|hibench.hadoop.home /home/ubuntu/hadoop|" conf/hadoop.conf
echo "hibench.hadoop.examples.jar /home/ubuntu/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.2.4.jar" >> conf/hadoop.conf

cp conf/spark.conf.template conf/spark.conf
sed -i "s|hibench.spark.home.*|hibench.spark.home /home/ubuntu/spark|"            conf/spark.conf
sed -i "s|hibench.yarn.executor.num.*|hibench.yarn.executor.num 2|"               conf/spark.conf
sed -i "s|hibench.yarn.executor.cores.*|hibench.yarn.executor.cores 2|"           conf/spark.conf
sed -i "s|spark.executor.memory.*|spark.executor.memory 2g|"                      conf/spark.conf
sed -i "s|spark.driver.memory.*|spark.driver.memory 2g|"                          conf/spark.conf
echo "hibench.masters.hostnames localhost"                                       >> conf/spark.conf
echo "hibench.slaves.hostnames localhost"                                        >> conf/spark.conf

sed -i "s|hibench.scale.profile.*|hibench.scale.profile large|" conf/hibench.conf

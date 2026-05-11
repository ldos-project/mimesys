#!/bin/bash
# Install DaCapo benchmark suite + Renaissance JAR (for finagle workloads).
set -eu -o pipefail
source "$(dirname "$0")/_common.sh"

cd "$BASE_PATH"
if [ ! -f "renaissance-gpl-0.14.2.jar" ]; then
    wget https://github.com/renaissance-benchmarks/renaissance/releases/download/v0.14.2/renaissance-gpl-0.14.2.jar
fi

sudo apt-get install -y \
    default-jre default-jdk openjdk-8-jre openjdk-8-jdk openjdk-11-jdk \
    ant cvs subversion nodejs npm python3 python3-pip

sudo python3 -m pip install colorama future tabulate requests wheel

cd "$BASE_PATH"
if [ ! -d "dacapobench" ]; then
    git clone https://github.com/dacapobench/dacapobench.git
fi
cd dacapobench
git checkout tags/v23.10-RC4-chopin

cd benchmarks
# Restore default.properties in case a previous run did `echo >> default.properties`
# without a trailing newline, which fused the appended key onto the previous line.
git checkout -- default.properties 2>/dev/null || true
# Use local.properties (loaded with higher precedence) for our overrides instead
# of mutating the upstream defaults. The cassandra/kafka/spring/tomcat benchmarks
# in v23.10-chopin require a real JDK 11.
# maven.url override: the DaCapo cassandra build pulls ~50 JARs via ant `<get>`
# directly from repo.maven.apache.org, which aggressively rate-limits scripted
# downloads and serves 404s with `retry-after: 111` mid-build. Google's GCS
# mirror of Maven Central is a full read-through proxy with no rate limit, and
# is much closer than repo1.maven.org/Alibaba from US-hosted runners. ant's
# property precedence is first-definition-wins and benchmarks/build.xml loads
# local.properties before dacapo.properties, so this overrides the default.
cat > local.properties <<EOF
jdk.8.home=/usr/lib/jvm/java-8-openjdk-amd64
jdk.11.home=/usr/lib/jvm/java-11-openjdk-amd64
maven.url=https://maven-central.storage-download.googleapis.com/maven2
EOF
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

[ -L /bin/python ] || sudo ln -sf /bin/python3 /bin/python

sed -i 's|<target name="compile" depends=.*>|<target name="compile" depends="cassandra,kafka,spring,tomcat">|' build.xml
sed -i 's|<property name="lib-url" .*>|<property name="lib-url" value="https://archive.apache.org/dist/commons/logging/source/"/>|' libs/commons-logging/build.xml
# Skip ant build if a previous VM already produced the final dacapo jar on the
# shared tree -- concurrent ant runs race on intermediate build artifacts and
# the Maven downloads underneath the cassandra/kafka build.xmls.
if ! ls dacapo-evaluation-git-*.jar >/dev/null 2>&1; then
    ant
fi

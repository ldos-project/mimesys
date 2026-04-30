set -eu -o pipefail

query_arr=()
query_arr+=("22 3 19 2 9 5 7 20 8 12 10 17 18 16 6 4 1 13 21 14 15 11")
query_arr+=("5 14 2 9 12 7 4 1 20 18 21 11 16 8 15 22 10 19 6 13 17 3")
query_arr+=("4 13 11 14 9 7 12 17 10 16 21 2 22 8 18 20 3 15 5 1 19 6")
query_arr+=("18 7 11 22 9 12 3 8 10 16 20 6 19 21 17 4 15 2 13 14 1 5")

cd /home/ubuntu
touch result_app_perf.txt
cd /home/ubuntu/pg-tpch

BASEDIR=$(pwd)
. "$BASEDIR/pgtpch_defaults"
PERFDATADIR="$PERFDATADIR-${SCALE}GB"

# Start a new instance of Postgres
sudo -u $PGUSER $PGBINDIR/postgres -D "$PGDATADIR" -p $PGPORT > /dev/null 2> /dev/null < /dev/null &
PGPID=$!
while ! sudo -u $PGUSER $PGBINDIR/pg_ctl status -D $PGDATADIR | grep "server is running" -q; do
  echo "Waiting for the Postgres server to start"
  sleep 1
done

# Wait for it to finish starting
sleep 5
echo "Postgres running, pid $PGPID"

thread_fn() {
    rm -f "result_thread_$2.txt"
    touch "result_thread_$2.txt"
    cur_query_arr=($1)
    for q in "${cur_query_arr[@]}"; do
        ii=$(printf "%02d" $q)
        f="queries/q$ii.sql"
        start_time=$(date +%s.%N)
        sudo -u $PGUSER $PGBINDIR/psql -h /tmp -p $PGPORT -d $DB_NAME <"$BASEDIR/$f"
        end_time=$(date +%s.%N)
        delay=$(echo "$end_time - $start_time" | bc)
        echo "$2,$q,$delay" >> "result_thread_$2.txt"
    done
}

# Start 4 threads and wait for them to finish
pid_arr=()
for t in 0 1 2 3; do
    thread_fn "${query_arr[$t]}" $t &
    pid_arr+=($!)
done
for t in 0 1 2 3; do
    wait "${pid_arr[$t]}"
    cat "result_thread_$t.txt" >> /home/ubuntu/result_app_perf.txt
done

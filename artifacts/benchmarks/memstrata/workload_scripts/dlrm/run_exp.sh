RESULTS_DIR="results"
mkdir -p $RESULTS_DIR

EXPORTS_FILE="dlrm/paths.export"
while read -r LINE
do
    export $LINE
done < "$EXPORTS_FILE"

export TEST_RESULTS_NAME=dlrm-$(date +%m%d)-$(date +%H%M%S)
sed -i "s/--num-batches=.*/--num-batches=6400/" $DLRM_SYSTEM/scripts/collect_1s.sh
/usr/bin/time -vo $RESULTS_DIR/$TEST_RESULTS_NAME.time conda run -n dlrm_cpu $DLRM_SYSTEM/scripts/collect_1s.sh \
              1> $RESULTS_DIR/$TEST_RESULTS_NAME.stdout 2> $RESULTS_DIR/$TEST_RESULTS_NAME.stderr

cat $RESULTS_DIR/$TEST_RESULTS_NAME.stdout >> /home/ubuntu/result_app_perf.txt

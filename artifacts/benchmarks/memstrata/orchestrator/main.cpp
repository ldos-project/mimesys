#include <unistd.h>
#include <getopt.h>
#include <signal.h>
#include <string.h>
#include <fcntl.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <chrono>
#include <fstream>
#include <limits>
#include <random>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <regex>
#include <thread>
#include <mutex>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <linux/page_coloring.h>
#include <numaif.h>
#include <mqueue.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <exception>
#include <limits>

#define COLOR_THP
// #undef COLOR_THP

using namespace std;

#define LIKELY(x) (__builtin_expect((x), 1))
#define UNLIKELY(x) (__builtin_expect((x), 0))

#define PHYS_TO_DRAM_PHYS(phys)	((phys) % (DRAM_SIZE_PER_NODE))
#define PMEM_NODE 2

#define ROUND_DOWN(a, b) ((a) / (b) * (b))
#define ROUND_UP(a, b) (((a) + (b) - 1) / (b) * (b))

#define BUG_ON(cond)											\
	do {												\
		if (cond) {										\
			fprintf(stderr, "BUG_ON: %s (L%d) %s\n", __FILE__, __LINE__, __FUNCTION__);	\
			raise(SIGABRT);									\
		}											\
	} while (0)

#define TRACE_PIPE_PATH "/sys/kernel/debug/tracing/trace_pipe"
#define COLORINFO_PATH "/proc/colorinfo"
#define BITMAP_SIZE ((NR_COLORS + sizeof(uint64_t) * 8 - 1) / (sizeof(uint64_t) * 8))
#define PAGE_SHIFT 12UL
#define PAGE_SIZE (1UL << PAGE_SHIFT)
#define HUGE_PAGE_SHIFT 21UL
#define HUGE_PAGE_SIZE (1UL << HUGE_PAGE_SHIFT)
#define MAX_PATH_LEN 512UL
#define MAX_NUM_CORES 512UL
#define TRACE_BUF_SIZE PAGE_SIZE
#define CONTENDED_PAGE_DEGREE 2

#define DAMON_MAX_NR_ACCESSES 2000  // 10s / 5ms

#ifdef COLOR_THP
#define COLOR_PAGE_SHIFT HUGE_PAGE_SHIFT
#define COLOR_PAGE_SIZE HUGE_PAGE_SIZE
#else
#define COLOR_PAGE_SHIFT PAGE_SHIFT
#define COLOR_PAGE_SIZE PAGE_SIZE
#endif

#define MQ_BUFFER_SIZE 8192

/* System-Related Stats */
#define TSC_FREQ 2.2e9
#define L3_MISS_LAT_TO_2LM_MR_INTERCEPT 0
#define L3_MISS_LAT_TO_2LM_MR_SLOPE 0

template<typename F>
void parse_list_str(char *list, F callback) {
	char *rest_list;
	char *interval = strtok_r(list, ",", &rest_list);
	while (interval != NULL) {
		long start, end;
		if (strstr(interval, "-") != NULL) {
			char *rest_token;
			char *token = strtok_r(interval, "-", &rest_token);
			start = atol(token);
			token = strtok_r(NULL, "-", &rest_token);
			end = atol(token);
		} else {
			start = atol(interval);
			end = start;
		}
		for (int i = start; i <= end; ++i)
			callback(i);
		interval = strtok_r(NULL, ",", &rest_list);
	}
}

double get_percentile(vector<uint64_t> &vec, double percentile) {
	if (vec.empty())
		return 0;
	double exact_index = (double)(vec.size() - 1) * percentile;
	double left_index = floor(exact_index);
	double right_index = ceil(exact_index);

	double left_value = (double) vec[(unsigned long) left_index];
	double right_value = (double) vec[(unsigned long) right_index];

	double value = left_value + (exact_index - left_index) * (right_value - left_value);
	return value;
}

struct perf_read_format {
	uint64_t value;
	uint64_t time_enabled;
	uint64_t time_running;
};

#define PERF_READ_FORMAT (PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING)

static int perf_event_open(struct perf_event_attr *attr,
	pid_t pid, int core, int group_fd, unsigned long flags) {
	return syscall(__NR_perf_event_open, attr, pid, core, group_fd, flags);
}

int perf_event_setup(int core, uint64_t config, uint64_t config1, bool pin_event,
	bool exclude_user, bool exclude_kernel, bool exclude_hv, bool exclude_idle,
	bool exclude_host, bool exclude_guest, int group_fd) {
	struct perf_event_attr event_attr;
	memset(&event_attr, 0, sizeof(event_attr));
	event_attr.type = PERF_TYPE_RAW;
	event_attr.size = sizeof(event_attr);
	event_attr.config = config;
	event_attr.config1 = config1;
	event_attr.disabled = 0;
	event_attr.read_format = PERF_READ_FORMAT;
	if (pin_event)
		event_attr.pinned = 1;
	if (exclude_user)
		event_attr.exclude_user = 1;
	if (exclude_kernel)
		event_attr.exclude_kernel = 1;
	if (exclude_hv)
		event_attr.exclude_hv = 1;
	if (exclude_idle)
		event_attr.exclude_idle = 1;
	if (exclude_host)
		event_attr.exclude_host = 1;
	if (exclude_guest)
		event_attr.exclude_guest = 1;

	int fd = perf_event_open(&event_attr, -1, core, group_fd, 0);
	return fd;
}

void perf_event_reset(int event_fd) {
	int ret = ioctl(event_fd, PERF_EVENT_IOC_RESET, 0);
	BUG_ON(ret < 0);
}

void perf_event_enable(int event_fd) {
	int ret = ioctl(event_fd, PERF_EVENT_IOC_ENABLE, 0);
	BUG_ON(ret < 0);
}

void perf_event_disable(int event_fd) {
	int ret = ioctl(event_fd, PERF_EVENT_IOC_DISABLE, 0);
	BUG_ON(ret < 0);
}

uint64_t perf_event_read(int event_fd) {
	struct perf_read_format data;
	int ret = read(event_fd, &data, sizeof(data));
	BUG_ON(ret != sizeof(data));
	double value = ((double) data.value) * ((double) data.time_enabled) / ((double) data.time_running);
	return (uint64_t) value;
}

void perf_event_teardown(int fd) {
	int ret;
	ret = close(fd);
	BUG_ON(ret != 0);
}

enum PerfEventType {
	PERF_EVENT_TYPE_INST_RETIRED = 0,
	PERF_EVENT_TYPE_CPU_CLK_UNHALTED_THREAD,
	PERF_EVENT_TYPE_CPU_CLK_UNHALTED_REF_TSC,
	PERF_EVENT_TYPE_DTLB_LOAD_MISSES_WALK_ACTIVE,
	PERF_EVENT_TYPE_DTLB_LOAD_MISSES_WALK_COMPLETED,
	PERF_EVENT_TYPE_OCR_OUTSTANDING_L3_MISS_DEMAND_DATA_RD,
	PERF_EVENT_TYPE_OCR_L3_MISS_DEMAND_DATA_RD,
	PERF_EVENT_TYPE_OCR_OUTSTANDING_DEMAND_DATA_RD,
	PERF_EVENT_TYPE_OCR_DEMAND_DATA_RD,
	PERF_EVENT_TYPE_MEM_LOAD_RETIRED_L2_MISS,
	PERF_EVENT_TYPE_L2_LINES_IN_ALL,
	PERF_EVENT_TYPE_OCR_MODIFIED_WRITE_ANY_RESPONSE,
	PERF_EVENT_TYPE_OCR_RFO_TO_CORE_L3_HIT_M,
	PERF_EVENT_TYPE_OCR_READS_TO_CORE_L3_MISS_LOCAL_SOCKET,
	PERF_EVENT_TYPE_OCR_HWPF_L3_L3_MISS_LOCAL,
	PERF_EVENT_TYPE_MAX
};

// Config format: X86_RAW_EVENT_MASK
uint64_t perf_event_config_arr[] = {
	0x000000C0UL,	// INST_RETIRED.ANY
	0x0000003CUL,	// CPU_CLK_UNHALTED.THREAD
	0x00000300UL,	// CPU_CLK_UNHALTED.REF_TSC
	0x01001012UL,	// DTLB_LOAD_MISSES.WALK_ACTIVE
	0x00000E12UL,	// DTLB_LOAD_MISSES.WALK_COMPLETED
	0x00001020UL,	// OFFCORE_REQUESTS_OUTSTANDING.L3_MISS_DEMAND_DATA_RD
	0x00001021UL,	// OFFCORE_REQUESTS.L3_MISS_DEMAND_DATA_RD
	0x00000120UL,	// OFFCORE_REQUESTS_OUTSTANDING.DEMAND_DATA_RD
	0x00000121UL,	// OFFCORE_REQUESTS.DEMAND_DATA_RD
	0x000010D1UL,	// MEM_LOAD_RETIRED.L2_MISS
	0x00001F25UL,	// L2_LINES_IN.ALL
	0x0000012AUL,	// OCR.MODIFIED_WRITE.ANY_RESPONSE
	0x0000012BUL,	// OCR.RFO_TO_CORE.L3_HIT_M
	0x0000012AUL,	// OCR.READS_TO_CORE.L3_MISS_LOCAL_SOCKET
	0x0000012BUL,	// OCR.HWPF_L3.L3_MISS_LOCAL
};

uint64_t perf_event_config1_arr[] = {
	0x0000000000UL,	// INST_RETIRED.ANY
	0x0000000000UL,	// CPU_CLK_UNHALTED.THREAD
	0x0000000000UL,	// CPU_CLK_UNHALTED.REF_TSC
	0x0000000000UL,	// DTLB_LOAD_MISSES.WALK_ACTIVE
	0x0000000000UL,	// DTLB_LOAD_MISSES.WALK_COMPLETED
	0x0000000000UL,	// OFFCORE_REQUESTS_OUTSTANDING.L3_MISS_DEMAND_DATA_RD
	0x0000000000UL,	// OFFCORE_REQUESTS.L3_MISS_DEMAND_DATA_RD
	0x0000000000UL,	// OFFCORE_REQUESTS_OUTSTANDING.DEMAND_DATA_RD
	0x0000000000UL,	// OFFCORE_REQUESTS.DEMAND_DATA_RD
	0x0000000000UL,	// MEM_LOAD_RETIRED.L2_MISS
	0x0000000000UL,	// L2_LINES_IN.ALL
	0x0000010808UL,	// OCR.MODIFIED_WRITE.ANY_RESPONSE
	0x1F80040022UL,	// OCR.RFO_TO_CORE.L3_HIT_M
	0x070CC04477UL,	// OCR.READS_TO_CORE.L3_MISS_LOCAL_SOCKET
	0x0084002380UL,	// OCR.HWPF_L3.L3_MISS_LOCAL
};

bool perf_event_is_leader_arr[] = {
	true,	// INST_RETIRED.ANY
	true,	// CPU_CLK_UNHALTED.THREAD
	true,	// CPU_CLK_UNHALTED.REF_TSC
	true,	// DTLB_LOAD_MISSES.WALK_ACTIVE
	false,	// DTLB_LOAD_MISSES.WALK_COMPLETED
	true,	// OFFCORE_REQUESTS_OUTSTANDING.L3_MISS_DEMAND_DATA_RD
	true,	// OFFCORE_REQUESTS.L3_MISS_DEMAND_DATA_RD
	true,	// OFFCORE_REQUESTS_OUTSTANDING.DEMAND_DATA_RD
	false,	// OFFCORE_REQUESTS.DEMAND_DATA_RD
	true,	// MEM_LOAD_RETIRED.L2_MISS
	true,	// L2_LINES_IN.ALL
	true,	// OCR.MODIFIED_WRITE.ANY_RESPONSE
	false,	// OCR.RFO_TO_CORE.L3_HIT_M
	true,	// OCR.READS_TO_CORE.L3_MISS_LOCAL_SOCKET
	false,	// OCR.HWPF_L3.L3_MISS_LOCAL
};

bool perf_event_pin_arr[] = {
	true,	// INST_RETIRED.ANY
	true,	// CPU_CLK_UNHALTED.THREAD
	true,	// CPU_CLK_UNHALTED.REF_TSC
	false,	// DTLB_LOAD_MISSES.WALK_ACTIVE
	false,	// DTLB_LOAD_MISSES.WALK_COMPLETED
	true,	// OFFCORE_REQUESTS_OUTSTANDING.L3_MISS_DEMAND_DATA_RD
	true,	// OFFCORE_REQUESTS.L3_MISS_DEMAND_DATA_RD
	false,	// OFFCORE_REQUESTS_OUTSTANDING.DEMAND_DATA_RD
	false,	// OFFCORE_REQUESTS.DEMAND_DATA_RD
	false,	// MEM_LOAD_RETIRED.L2_MISS
	false,	// L2_LINES_IN.ALL
	false,	// OCR.MODIFIED_WRITE.ANY_RESPONSE
	false,	// OCR.RFO_TO_CORE.L3_HIT_M
	false,	// OCR.READS_TO_CORE.L3_MISS_LOCAL_SOCKET
	false,	// OCR.HWPF_L3.L3_MISS_LOCAL
};

const char *perf_event_name_arr[] = {
	"INST_RETIRED.ANY",
	"CPU_CLK_UNHALTED.THREAD",
	"CPU_CLK_UNHALTED.REF_TSC",
	"DTLB_LOAD_MISSES.WALK_ACTIVE",
	"DTLB_LOAD_MISSES.WALK_COMPLETED",
	"OFFCORE_REQUESTS_OUTSTANDING.L3_MISS_DEMAND_DATA_RD",
	"OFFCORE_REQUESTS.L3_MISS_DEMAND_DATA_RD",
	"OFFCORE_REQUESTS_OUTSTANDING.DEMAND_DATA_RD",
	"OFFCORE_REQUESTS.DEMAND_DATA_RD",
	"MEM_LOAD_RETIRED.L2_MISS",
	"L2_LINES_IN.ALL",
	"OCR.MODIFIED_WRITE.ANY_RESPONSE",
	"OCR.RFO_TO_CORE.L3_HIT_M",
	"OCR.READS_TO_CORE.L3_MISS_LOCAL_SOCKET",
	"OCR.HWPF_L3.L3_MISS_LOCAL",
};

static_assert(
	(sizeof(perf_event_config_arr) / sizeof(*perf_event_config_arr)) == PERF_EVENT_TYPE_MAX,
	"perf_event_config_arr does not match with PERF_EVENT_TYPE_MAX");
static_assert(
	(sizeof(perf_event_config1_arr) / sizeof(*perf_event_config_arr)) == PERF_EVENT_TYPE_MAX,
	"perf_event_config1_arr does not match with PERF_EVENT_TYPE_MAX");
static_assert(
	(sizeof(perf_event_is_leader_arr) / sizeof(*perf_event_is_leader_arr)) == PERF_EVENT_TYPE_MAX,
	"perf_event_is_leader_arr does not match with PERF_EVENT_TYPE_MAX");
static_assert(
	(sizeof(perf_event_pin_arr) / sizeof(*perf_event_pin_arr)) == PERF_EVENT_TYPE_MAX,
	"perf_event_pin_arr does not match with PERF_EVENT_TYPE_MAX");
static_assert(
	(sizeof(perf_event_name_arr) / sizeof(*perf_event_name_arr)) == PERF_EVENT_TYPE_MAX,
	"perf_event_name_arr does not match with PERF_EVENT_TYPE_MAX");

enum MetricType {
	METRIC_TYPE_DTLB_LOAD_MISS_LAT = 0,		// metric_DTLB load miss latency (in core clks)
	METRIC_TYPE_DLTB_LOAD_MPI,				// metric_DTLB (2nd level) load MPI
	METRIC_TYPE_LOAD_L3_MISS_LAT,			// metric_Load_L3_Miss_Latency_using_ORO_events(ns)
	METRIC_TYPE_LOAD_L2_MISS_LAT,			// metric_Load_L2_Miss_Latency_using_ORO_events(ns)
	METRIC_TYPE_L2_DEMAND_DATA_RD_MPI,		// metric_L2 demand data read MPI
	METRIC_TYPE_L2_MPI,						// metric_L2 MPI (includes code+data+rfo w/ prefetches)
	METRIC_TYPE_CORE_READ,					// metric_core initiated local socket memory read bandwidth (MB/sec)
	METRIC_TYPE_CORE_WRITE,					// metric_core initiated write bandwidth (MB/sec)
	METRIC_TYPE_2LM_MISS_RATIO,				// metric_memory mode near memory cache read miss rate% (estimated)
	METRIC_TYPE_2LM_MPKI,					// metric_memory mode MPKI (estimated)
	METRIC_TYPE_2LM_PER_PAGE_MISS_RATE,		// metric_memory mode per-page miss rate (estimated)
	METRIC_TYPE_MAX,
};

#define MIN_MISS_METRIC METRIC_TYPE_2LM_PER_PAGE_MISS_RATE
#define KARMA_METRIC METRIC_TYPE_2LM_PER_PAGE_MISS_RATE

const char *metric_name_arr[] = {
	"metric_DTLB load miss latency (in core clks)",
	"metric_DTLB (2nd level) load MPI",
	"metric_Load_L3_Miss_Latency_using_ORO_events(ns)",
	"metric_Load_L2_Miss_Latency_using_ORO_events(ns)",
	"metric_L2 demand data read MPI",
	"metric_L2 MPI (includes code+data+rfo w/ prefetches)",
	"metric_core initiated local socket memory read bandwidth (MB/sec)",
	"metric_core initiated write bandwidth (MB/sec)",
	"metric_memory mode near memory cache read miss rate% (estimated)",
	"metric_memory mode MPKI (estimated)",
	"metric_memory mode per-page miss rate (estimated)",
};

double metric_max_arr[] = {
	200,							// metric_DTLB load miss latency (in core clks)
	0.5,							// metric_DTLB (2nd level) load MPI
	350,							// metric_Load_L3_Miss_Latency_using_ORO_events(ns)
	200,							// metric_Load_L2_Miss_Latency_using_ORO_events(ns)
	0.5,							// metric_L2 demand data read MPI
	0.5,							// metric_L2 MPI (includes code+data+rfo w/ prefetches)
	512000,							// metric_core initiated local socket memory read bandwidth (MB/sec)
	512000,							// metric_core initiated write bandwidth (MB/sec)
	100,							// metric_memory mode near memory cache read miss rate% (estimated)
	100,							// metric_memory mode MPKI (estimated)
	numeric_limits<double>::max(),	// metric_memory mode per-page miss rate (estimated)
};

static_assert(
	(sizeof(metric_name_arr) / sizeof(*metric_name_arr)) == METRIC_TYPE_MAX,
	"metric_name_arr does not match with METRIC_TYPE_MAX");

static_assert(
	(sizeof(metric_max_arr) / sizeof(*metric_max_arr)) == METRIC_TYPE_MAX,
	"metric_max_arr does not match with METRIC_TYPE_MAX");

struct page_xchg_stats {
	int num_get_page_err;
	int num_add_page_err;
	int num_skipped_page;
	int num_malloc_err;
	int num_migrate_err;
	int num_succeeded;
	int num_thp_succeeded;
};

void exchange_pages(int pid_1, int pid_2, int num_pages, void **page_arr_1, void **page_arr_2,
	void **skipped_page_arr_1, void **skipped_page_arr_2, struct page_xchg_stats *stats) {
	struct color_swap_req req;
	req.pid_1 = pid_1;
	req.pid_2 = pid_2;
	req.num_pages = num_pages;
	req.page_arr_1 = page_arr_1;
	req.page_arr_2 = page_arr_2;
	req.swap_ppool_index = false;
	req.skipped_page_arr_1 = skipped_page_arr_1;
	req.skipped_page_arr_2 = skipped_page_arr_2;

	int fd = open(COLORINFO_PATH, O_RDONLY);
	BUG_ON(fd < 0);
	int ioctl_ret = ioctl(fd, COLOR_IOC_SWAP, &req);
	int ret = close(fd);
	BUG_ON(ret != 0);

	if (ioctl_ret != 0) {
		stats->num_get_page_err = num_pages;
		stats->num_add_page_err = 0;
		stats->num_skipped_page = 0;
		stats->num_malloc_err = 0;
		stats->num_migrate_err = 0;
		stats->num_succeeded = 0;
		stats->num_thp_succeeded = 0;
	} else {
		stats->num_get_page_err = req.num_get_page_err;
		stats->num_add_page_err = req.num_add_page_err;
		stats->num_skipped_page = req.num_skipped_page;
		stats->num_malloc_err = req.num_malloc_err;
		stats->num_migrate_err = req.num_migrate_err;
		stats->num_succeeded = req.num_succeeded;
		stats->num_thp_succeeded = req.num_thp_succeeded;
	}
}

struct damon_region {
	uint64_t start;
	uint64_t end;

	uint64_t nr_accesses;
	uint64_t age;

	uint64_t sampling_addr;
	uint64_t region_id;
};

struct VM {
	int pid;
	string core_str;
	uint64_t base_virt_addr;
	uint64_t memory_size_MB;

	int num_cores;
	int *perf_event_fds[PERF_EVENT_TYPE_MAX];
	uint64_t cum_perf_event_counts[PERF_EVENT_TYPE_MAX];
	uint64_t delta_perf_event_counts[PERF_EVENT_TYPE_MAX];
	double cur_metrics[METRIC_TYPE_MAX];
	double ewma_metrics[METRIC_TYPE_MAX];
	bool ewma_initialized;
	bool has_slowdown;

	vector<uint64_t> *virt_addr_vec_uncontended;
	vector<uint64_t> *virt_addr_vec_contended;
	uint64_t num_excluded_uncontended;
	uint64_t num_excluded_contended;
	unordered_map<uint64_t, uint64_t> *virt_addr_map_uncontended;
	unordered_map<uint64_t, uint64_t> *virt_addr_map_contended;
	uint64_t initial_num_uncontented_pages;

	// Hot-cold separation
	vector<struct damon_region> *damon_region_vec;
	int hot_cold_step_size;

	// Karma
	double ewma_avg_karma_metric;
	bool ewma_avg_karma_initialized;
	double cur_sum_karma_metric;
	int cur_sum_karma_num_dp;

	uint64_t fair_share;
	long credits;
};

void virt_addr_swap(vector<uint64_t> &virt_addr_vec, unordered_map<uint64_t, uint64_t> &virt_addr_map,
                    uint64_t page_degree, uint64_t i, uint64_t j) {
	if (i == j)
		return;
	for (uint64_t k = 0; k < page_degree; ++k) {
		swap(virt_addr_vec[i * page_degree + k], virt_addr_vec[j * page_degree + k]);
		swap(virt_addr_map[virt_addr_vec[i * page_degree + k]],
		     virt_addr_map[virt_addr_vec[j * page_degree + k]]);
	}
}

void virt_addr_shuffle_last_n_elem(vector<uint64_t> &virt_addr_vec, unordered_map<uint64_t, uint64_t> &virt_addr_map,
                                   default_random_engine &rand_eng, uint64_t page_degree, uint64_t num_dram_pages_to_shuffle) {
	if (num_dram_pages_to_shuffle == 0)
		return;
	BUG_ON(page_degree == 0);
	BUG_ON(virt_addr_vec.size() % page_degree != 0);
	BUG_ON(num_dram_pages_to_shuffle * page_degree > virt_addr_vec.size());
	BUG_ON(virt_addr_map.size() != virt_addr_vec.size());

	uint64_t num_dram_pages = virt_addr_vec.size() / page_degree;
	if (num_dram_pages == 1)
		return;

	for (uint64_t i = 0; i < num_dram_pages_to_shuffle; ++i) {
		uniform_int_distribution<uint64_t> dist(0, num_dram_pages - i - 1);
		uint64_t rand_idx = dist(rand_eng);
		virt_addr_swap(virt_addr_vec, virt_addr_map, page_degree, num_dram_pages - i - 1, rand_idx);
	}
}

void find_pages_2lm(struct VM *vm, default_random_engine &rand_eng) {
	char pagemap_path[MAX_PATH_LEN];
	sprintf(pagemap_path, "/proc/%d/pagemap", vm->pid);
	int pagemap_fd = open(pagemap_path, O_RDONLY);
	BUG_ON(pagemap_fd < 0);

	uint64_t vm_num_pages = (vm->memory_size_MB << 20) >> COLOR_PAGE_SHIFT;
	uint64_t vm_num_4KB_pages = (vm->memory_size_MB << 20) >> PAGE_SHIFT;
	uint64_t *buffer = (uint64_t *) malloc(vm_num_4KB_pages * sizeof(*buffer));
	BUG_ON(buffer == NULL);
	BUG_ON(vm->base_virt_addr % COLOR_PAGE_SIZE != 0);
	ssize_t pread_ret = pread(pagemap_fd, buffer, vm_num_4KB_pages * sizeof(*buffer),
	                          (vm->base_virt_addr >> PAGE_SHIFT) * sizeof(*buffer));
	BUG_ON(pread_ret != vm_num_4KB_pages * sizeof(*buffer));

	unordered_map<uint64_t, vector<uint64_t>> dram_addr_to_virt_addr_map;
	unordered_set<uint64_t> phys_addr_set;
	unordered_set<uint64_t> dram_addr_to_exclude;
	for (uint64_t jj = 0; jj < vm_num_pages; ++jj) {
		uint64_t j = jj << (COLOR_PAGE_SHIFT - PAGE_SHIFT);
		BUG_ON((buffer[j] & (1UL << 63)) == 0);  // Not present
		BUG_ON((buffer[j] & (1UL << 62)) != 0);  // Swapped
		BUG_ON((buffer[j] & (1UL << 60)) == 0);  // Not PPOOLED
		BUG_ON((buffer[j] & (1UL << 56)) == 0);  // Not exclusively mapped
		uint64_t phys_addr = (buffer[j] & ((1UL << 55) - 1)) << PAGE_SHIFT;
		BUG_ON(phys_addr == 0);
		BUG_ON(phys_addr_set.find(phys_addr) != phys_addr_set.end());
		phys_addr_set.insert(phys_addr);

		uint64_t dram_addr = PHYS_TO_DRAM_PHYS(phys_addr);
		dram_addr_to_virt_addr_map[dram_addr].push_back(vm->base_virt_addr + jj * COLOR_PAGE_SIZE);
		if ((buffer[j] & (1UL << 59)) == 0)  // Has extra references
			dram_addr_to_exclude.insert(dram_addr);
	}
	free(buffer);
	vm->num_excluded_uncontended = 0;
	vm->num_excluded_contended = 0;
	for (auto &pair : dram_addr_to_virt_addr_map) {
		if (pair.second.size() == CONTENDED_PAGE_DEGREE) {
			if (dram_addr_to_exclude.find(pair.first) != dram_addr_to_exclude.end()) {
				vm->num_excluded_contended += CONTENDED_PAGE_DEGREE;
				continue;
			}
			for (int j = 0; j < CONTENDED_PAGE_DEGREE; ++j) {
				vm->virt_addr_vec_contended->push_back(pair.second[j]);
				(*vm->virt_addr_map_contended)[pair.second[j]] = vm->virt_addr_vec_contended->size() - 1;
			}
			virt_addr_shuffle_last_n_elem(*vm->virt_addr_vec_contended, *vm->virt_addr_map_contended, rand_eng, CONTENDED_PAGE_DEGREE, 1);
		} else if (pair.second.size() == 1) {
			if (dram_addr_to_exclude.find(pair.first) != dram_addr_to_exclude.end()) {
				vm->num_excluded_uncontended += 1;
				continue;
			}
			vm->virt_addr_vec_uncontended->push_back(pair.second[0]);
			(*vm->virt_addr_map_uncontended)[pair.second[0]] = vm->virt_addr_vec_uncontended->size() - 1;
			virt_addr_shuffle_last_n_elem(*vm->virt_addr_vec_uncontended, *vm->virt_addr_map_uncontended, rand_eng, 1, 1);
		} else {
			BUG_ON(true);
		}
	}
	close(pagemap_fd);
}

void find_and_move_pages_1lm(struct VM *vm, double dram_pct, default_random_engine &rand_eng) {
	char pagemap_path[MAX_PATH_LEN];
	sprintf(pagemap_path, "/proc/%d/pagemap", vm->pid);
	int pagemap_fd = open(pagemap_path, O_RDONLY);
	BUG_ON(pagemap_fd < 0);

	uint64_t vm_num_pages = (vm->memory_size_MB << 20) >> COLOR_PAGE_SHIFT;
	uint64_t vm_num_4KB_pages = (vm->memory_size_MB << 20) >> PAGE_SHIFT;
	uint64_t *buffer = (uint64_t *) malloc(vm_num_4KB_pages * sizeof(*buffer));
	BUG_ON(buffer == NULL);
	BUG_ON(vm->base_virt_addr % COLOR_PAGE_SIZE != 0);
	ssize_t pread_ret = pread(pagemap_fd, buffer, vm_num_4KB_pages * sizeof(*buffer),
	                          (vm->base_virt_addr >> PAGE_SHIFT) * sizeof(*buffer));
	BUG_ON(pread_ret != vm_num_4KB_pages * sizeof(*buffer));

	unordered_set<uint64_t> phys_addr_set;
	vector<uint64_t> virt_addr_vec;
	for (uint64_t jj = 0; jj < vm_num_pages; ++jj) {
		uint64_t j = jj << (COLOR_PAGE_SHIFT - PAGE_SHIFT);
		BUG_ON((buffer[j] & (1UL << 63)) == 0);  // Not present
		BUG_ON((buffer[j] & (1UL << 62)) != 0);  // Swapped
		// BUG_ON((buffer[j] & (1UL << 60)) == 0);  // Not PPOOLED
		BUG_ON((buffer[j] & (1UL << 56)) == 0);  // Not exclusively mapped
		uint64_t phys_addr = (buffer[j] & ((1UL << 55) - 1)) << PAGE_SHIFT;
		BUG_ON(phys_addr == 0);
		BUG_ON(phys_addr_set.find(phys_addr) != phys_addr_set.end());
		phys_addr_set.insert(phys_addr);

		if ((buffer[j] & (1UL << 59)) != 0)  // Does not have extra references
			virt_addr_vec.push_back(vm->base_virt_addr + jj * COLOR_PAGE_SIZE);
	}
	free(buffer);

	vm->num_excluded_contended = 0;
	vm->num_excluded_uncontended = vm_num_pages - virt_addr_vec.size();

	uint64_t num_pmem_pages = ROUND_DOWN((uint64_t) ((double) virt_addr_vec.size() * (1 - dram_pct)), 3);
	uint64_t num_dram_pages = virt_addr_vec.size() - num_pmem_pages;
	shuffle(virt_addr_vec.begin(), virt_addr_vec.end(), rand_eng);

	int *nodes = (int *) malloc(num_pmem_pages * sizeof(*nodes));
	BUG_ON(nodes == NULL);
	for (uint64_t i = 0; i < num_pmem_pages; ++i)
		nodes[i] = PMEM_NODE;
	int *status = (int *) malloc(num_pmem_pages * sizeof(*status));
	BUG_ON(status == NULL);
	long move_pages_ret = move_pages(vm->pid, num_pmem_pages, (void **) virt_addr_vec.data(),
	                                 nodes, status, MPOL_MF_MOVE_ALL);
	BUG_ON(move_pages_ret != 0);
	free(nodes);
	free(status);

	vm->virt_addr_vec_contended->insert(vm->virt_addr_vec_contended->end(), virt_addr_vec.begin(), virt_addr_vec.begin() + num_pmem_pages);
	for (uint64_t i = 0; i < num_pmem_pages; ++i) {
		uint64_t virt_addr = (*vm->virt_addr_vec_contended)[i];
		(*vm->virt_addr_map_contended)[virt_addr] = i;
	}
	vm->virt_addr_vec_uncontended->insert(vm->virt_addr_vec_uncontended->end(), virt_addr_vec.begin() + num_pmem_pages, virt_addr_vec.end());
	for (uint64_t i = 0; i < num_dram_pages; ++i) {
		uint64_t virt_addr = (*vm->virt_addr_vec_uncontended)[i];
		(*vm->virt_addr_map_uncontended)[virt_addr] = i;
	}

	close(pagemap_fd);
}

inline uint64_t vm_count_all_uncontended(struct VM *vm) {
	return vm->virt_addr_vec_uncontended->size() + vm->num_excluded_uncontended;
}

inline uint64_t vm_count_all_contended(struct VM *vm) {
	return vm->virt_addr_vec_contended->size() + vm->num_excluded_contended;
}

void setup_vm(struct VM *vm, int pid, string core_str,
              uint64_t base_virt_addr, uint64_t memory_size_MB, default_random_engine &rand_eng,
			  bool is_1lm, double dram_pct, int hot_cold_step_size, int karma_init_credit_factor) {
	vm->pid = pid;
	vm->core_str = core_str;
	vm->base_virt_addr = base_virt_addr;
	vm->memory_size_MB = memory_size_MB;

	// Count the number of cores
	vm->num_cores = 0;
	string tmp_core_str = string(vm->core_str);
	parse_list_str((char *) tmp_core_str.c_str(), [&](long core) {++vm->num_cores;});
	BUG_ON(vm->num_cores <= 0);
	for (int j = 0; j < PERF_EVENT_TYPE_MAX; ++j) {
		vm->perf_event_fds[j] = (int *) malloc(vm->num_cores * sizeof(int));
		BUG_ON(vm->perf_event_fds[j] == NULL);
		vm->cum_perf_event_counts[j] = 0;
		vm->delta_perf_event_counts[j] = 0;
	}
	for (int j = 0; j < METRIC_TYPE_MAX; ++j) {
		vm->cur_metrics[j] = 0;
		vm->ewma_metrics[j] = 0;
	}
	vm->ewma_initialized = false;
	vm->has_slowdown = false;

	// Set up perf counters
	int core_index = 0;
	tmp_core_str = string(vm->core_str);
	parse_list_str((char *) tmp_core_str.c_str(), [&](long core) {
		BUG_ON(core < 0);
		int leader_fd = -1;
		for (int j = 0; j < PERF_EVENT_TYPE_MAX; ++j) {
			if (perf_event_is_leader_arr[j])
				leader_fd = -1;
			vm->perf_event_fds[j][core_index] = perf_event_setup((int) core,
				perf_event_config_arr[j], perf_event_config1_arr[j],
				/* pin_event */ perf_event_pin_arr[j], false, false, false, false,
				/* exclude_host */ false,  /* exclude_guest */ false,
				leader_fd);
			BUG_ON(vm->perf_event_fds[j][core_index] < 0);
			if (leader_fd == -1)
				leader_fd = vm->perf_event_fds[j][core_index];
		}
		core_index++;
	});

	// Find uncontended and contended pages
	vm->virt_addr_vec_uncontended = new vector<uint64_t>;
	vm->virt_addr_vec_contended = new vector<uint64_t>;

	vm->virt_addr_map_uncontended = new unordered_map<uint64_t, uint64_t>;
	vm->virt_addr_map_contended = new unordered_map<uint64_t, uint64_t>;

	if (!is_1lm) {
		find_pages_2lm(vm, rand_eng);
	} else {
		find_and_move_pages_1lm(vm, dram_pct, rand_eng);
	}
	vm->initial_num_uncontented_pages = vm_count_all_uncontended(vm);

	vm->damon_region_vec = new vector<struct damon_region>;
	vm->hot_cold_step_size = hot_cold_step_size;

	vm->ewma_avg_karma_metric = 0;
	vm->ewma_avg_karma_initialized = false;
	vm->cur_sum_karma_metric = 0;
	vm->cur_sum_karma_num_dp = 0;
	// Assume the initial allocation is fair
	// FIXME: Need to be updated if the VM is added on the fly
	vm->fair_share = vm_count_all_uncontended(vm);
	vm->credits = (long) (vm_count_all_uncontended(vm) + vm_count_all_contended(vm)) * karma_init_credit_factor;
}

void teardown_vm(struct VM *vm) {
	for (int j = 0; j < PERF_EVENT_TYPE_MAX; ++j) {
		uint64_t new_cum_count = 0;
		for (int k = 0; k < vm->num_cores; ++k) {
			uint64_t count = perf_event_read(vm->perf_event_fds[j][k]);
			new_cum_count += count;
			perf_event_teardown(vm->perf_event_fds[j][k]);
		}
		if (new_cum_count < vm->cum_perf_event_counts[j]) {
			// This could happen because of the running_time / enabled_time estimation
			new_cum_count = vm->cum_perf_event_counts[j];
		}
		vm->delta_perf_event_counts[j] = new_cum_count - vm->cum_perf_event_counts[j];
		vm->cum_perf_event_counts[j] = new_cum_count;
		free(vm->perf_event_fds[j]);
	}
	delete vm->virt_addr_vec_uncontended;
	delete vm->virt_addr_vec_contended;
	delete vm->virt_addr_map_uncontended;
	delete vm->virt_addr_map_contended;
	delete vm->damon_region_vec;
	vm->pid = -1;
}

enum PolicyType {
	POLICY_MIN_MISS,
	POLICY_KARMA,
};

void update_perf_counters(vector<struct VM> &vm_vec, ofstream &perf_log_file, uint64_t epoch) {
	for (int i = 0; i < vm_vec.size(); ++i) {
		struct VM *vm = &vm_vec[i];
		if (vm->pid < 0)
			continue;
		perf_log_file << epoch << "," << i;
		for (int j = 0; j < PERF_EVENT_TYPE_MAX; ++j) {
			uint64_t new_cum_count = 0;
			for (int k = 0; k < vm->num_cores; ++k) {
				uint64_t count = perf_event_read(vm->perf_event_fds[j][k]);
				new_cum_count += count;
			}
			if (new_cum_count < vm->cum_perf_event_counts[j]) {
				// This could happen because of the running_time / enabled_time estimation
				new_cum_count = vm->cum_perf_event_counts[j];
			}
			vm->delta_perf_event_counts[j] = new_cum_count - vm->cum_perf_event_counts[j];
			vm->cum_perf_event_counts[j] = new_cum_count;
			perf_log_file << "," << perf_event_name_arr[j]
			              << "," << vm->delta_perf_event_counts[j]
			              << "," << vm->cum_perf_event_counts[j];
		}
		perf_log_file << endl;
	}
}

void update_metrics(vector<struct VM> &vm_vec, ofstream &metric_log_file,
                    double ewma_constant, uint64_t epoch, int operation_window_s) {
	for (int i = 0; i < vm_vec.size(); ++i) {
		struct VM *vm = &vm_vec[i];
		if (vm->pid < 0)
			continue;

		vm->cur_metrics[METRIC_TYPE_DTLB_LOAD_MISS_LAT] =
			(double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_DTLB_LOAD_MISSES_WALK_ACTIVE]
			/ (double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_DTLB_LOAD_MISSES_WALK_COMPLETED];
		vm->cur_metrics[METRIC_TYPE_DLTB_LOAD_MPI] =
			(double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_DTLB_LOAD_MISSES_WALK_COMPLETED]
			/ (double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_INST_RETIRED];
		vm->cur_metrics[METRIC_TYPE_LOAD_L3_MISS_LAT] =
			1e9 * ((double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_OCR_OUTSTANDING_L3_MISS_DEMAND_DATA_RD]
					/ (double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_OCR_L3_MISS_DEMAND_DATA_RD])
			/ ((double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_CPU_CLK_UNHALTED_THREAD]
				/ (double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_CPU_CLK_UNHALTED_REF_TSC]
				* TSC_FREQ);
		vm->cur_metrics[METRIC_TYPE_LOAD_L2_MISS_LAT] =
			1e9 * ((double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_OCR_OUTSTANDING_DEMAND_DATA_RD]
					/ (double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_OCR_DEMAND_DATA_RD])
			/ ((double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_CPU_CLK_UNHALTED_THREAD]
				/ (double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_CPU_CLK_UNHALTED_REF_TSC]
				* TSC_FREQ);
		vm->cur_metrics[METRIC_TYPE_L2_DEMAND_DATA_RD_MPI] =
			(double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_MEM_LOAD_RETIRED_L2_MISS]
			/ (double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_INST_RETIRED];
		vm->cur_metrics[METRIC_TYPE_L2_MPI] =
			(double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_L2_LINES_IN_ALL]
			/ (double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_INST_RETIRED];

		double a, b;
		a = (double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_OCR_READS_TO_CORE_L3_MISS_LOCAL_SOCKET];
		b = (double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_OCR_HWPF_L3_L3_MISS_LOCAL];
		double read_count = a + b;
		vm->cur_metrics[METRIC_TYPE_CORE_READ] = read_count * 64 / 1e6;
		vm->cur_metrics[METRIC_TYPE_CORE_READ] /= operation_window_s;

		a = (double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_OCR_MODIFIED_WRITE_ANY_RESPONSE];
		b = (double) vm->delta_perf_event_counts[PERF_EVENT_TYPE_OCR_RFO_TO_CORE_L3_HIT_M];
		double write_count = max(0.0, (a - b));
		vm->cur_metrics[METRIC_TYPE_CORE_WRITE] = (a - b) * 64 / 1e6;
		vm->cur_metrics[METRIC_TYPE_CORE_WRITE] /= operation_window_s;

		vm->cur_metrics[METRIC_TYPE_2LM_MISS_RATIO] = L3_MISS_LAT_TO_2LM_MR_INTERCEPT
			+ L3_MISS_LAT_TO_2LM_MR_SLOPE * vm->cur_metrics[METRIC_TYPE_LOAD_L3_MISS_LAT];
		vm->cur_metrics[METRIC_TYPE_2LM_MISS_RATIO] = min(1.0, max(0.0, vm->cur_metrics[METRIC_TYPE_2LM_MISS_RATIO]));

		vm->cur_metrics[METRIC_TYPE_2LM_MPKI] =
			1e3 * vm->cur_metrics[METRIC_TYPE_2LM_MISS_RATIO]
			* (read_count + write_count)
			/ vm->delta_perf_event_counts[PERF_EVENT_TYPE_INST_RETIRED];

		vm->cur_metrics[METRIC_TYPE_2LM_PER_PAGE_MISS_RATE] =
			vm->cur_metrics[METRIC_TYPE_2LM_MISS_RATIO]
			* (read_count + write_count)
			/ (vm_count_all_contended(vm) + ((128 << 20) >> COLOR_PAGE_SHIFT))
			/ operation_window_s;

		for (int j = 0; j < METRIC_TYPE_MAX; ++j) {
			vm->cur_metrics[j] = min(metric_max_arr[j], max(0.0, vm->cur_metrics[j]));
		}

		// Calculate EWMA
		for (int j = 0; j < METRIC_TYPE_MAX; ++j) {
			if (vm->ewma_initialized) {
				vm->ewma_metrics[j] = ewma_constant * vm->cur_metrics[j] + (1 - ewma_constant) * vm->ewma_metrics[j];
			} else {
				vm->ewma_metrics[j] = vm->cur_metrics[j];
			}
		}
		vm->ewma_initialized = true;

		// Log metrics
		metric_log_file << epoch << "," << i;
		for (int j = 0; j < METRIC_TYPE_MAX; ++j) {
			metric_log_file << "," << metric_name_arr[j] << "," << vm->cur_metrics[j] << "," << vm->ewma_metrics[j];
		}
		metric_log_file << endl;
	}
}

void update_karma_avg_metrics(vector<struct VM> &vm_vec, double ewma_constant, int avg_size) {
	for (int i = 0; i < vm_vec.size(); ++i) {
		struct VM *vm = &vm_vec[i];
		if (vm->pid < 0)
			continue;

		// Karma does not use weights
		BUG_ON(vm->cur_sum_karma_num_dp >= avg_size);
		vm->cur_sum_karma_metric += vm->ewma_metrics[KARMA_METRIC];
		vm->cur_sum_karma_num_dp += 1;
		if (vm->cur_sum_karma_num_dp == avg_size) {
			double cur_avg_karma_metric = vm->cur_sum_karma_metric / (double) avg_size;
			vm->cur_sum_karma_metric = 0;
			vm->cur_sum_karma_num_dp = 0;

			if (!vm->ewma_avg_karma_initialized) {
				vm->ewma_avg_karma_metric = cur_avg_karma_metric;
				vm->ewma_avg_karma_initialized = true;
			} else {
				vm->ewma_avg_karma_metric = ewma_constant * cur_avg_karma_metric + (1 - ewma_constant) * vm->ewma_avg_karma_metric;
			}
		}
	}
}

void log_and_print_vms(vector<struct VM> &vm_vec, ofstream &log_file, uint64_t epoch) {
	printf("Time: %ld seconds\n", epoch);
	for (int i = 0; i < vm_vec.size(); ++i) {
		struct VM *vm = &vm_vec[i];
		if (vm->pid < 0)
			continue;

		vector<uint64_t> nr_accesses_vec;
		for (auto &region : *vm->damon_region_vec) {
			nr_accesses_vec.push_back(region.nr_accesses);
		}
		sort(nr_accesses_vec.begin(), nr_accesses_vec.end());

		log_file << epoch << "," << i << ","
		         << vm->virt_addr_vec_uncontended->size() << "," << vm->virt_addr_vec_contended->size() << ","
		         << vm->num_excluded_uncontended << "," << vm->num_excluded_contended << ","
		         << vm->damon_region_vec->size() << "," << (nr_accesses_vec.empty() ? 0 : nr_accesses_vec.at(0)) << ","
		         << (nr_accesses_vec.empty() ? 0 : nr_accesses_vec.at(nr_accesses_vec.size() - 1)) << ","
		         << get_percentile(nr_accesses_vec, 0.25) << "," << get_percentile(nr_accesses_vec, 0.5) << "," << get_percentile(nr_accesses_vec, 0.75) << ","
		         << (vm->has_slowdown ? "1" : "0") << endl;

		uint64_t num_uncontended_pages = vm_count_all_uncontended(vm);
		uint64_t num_contended_pages = vm_count_all_contended(vm);
		printf("* VM %d: PID = %d, Core = %s, Memory Size = %ld MB, Hot-Cold Separation Step Size = %d MB\n"
		       "    #Uncontended Pages = %lu (%lu MB, %lu excluded), #Contended Pages = %lu (%lu MB, %lu excluded)\n"
		       "    Number of DAMON Regions = %lu, Min #Accesses = %ld, Max #Accesses = %ld, p25 = %lf, p50 = %lf, p75 = %lf\n"
		       "    Has Slowdown = %d\n",
		       i, vm->pid, vm->core_str.c_str(), vm->memory_size_MB, (vm->hot_cold_step_size << COLOR_PAGE_SHIFT) >> 20,
		       num_uncontended_pages, (num_uncontended_pages << COLOR_PAGE_SHIFT) >> 20, vm->num_excluded_uncontended,
		       num_contended_pages, (num_contended_pages << COLOR_PAGE_SHIFT) >> 20, vm->num_excluded_contended,
		       vm->damon_region_vec->size(),
		       nr_accesses_vec.empty() ? 0 : nr_accesses_vec[0],
		       nr_accesses_vec.empty() ? 0 : nr_accesses_vec[nr_accesses_vec.size() - 1],
		       get_percentile(nr_accesses_vec, 0.25), get_percentile(nr_accesses_vec, 0.5), get_percentile(nr_accesses_vec, 0.75),
		       vm->has_slowdown ? 1	: 0);
		for (int j = 0; j < METRIC_TYPE_MAX; ++j) {
			printf("    %s = %lf (EWMA %lf)\n", metric_name_arr[j], vm->cur_metrics[j], vm->ewma_metrics[j]);
		}
	}
}

enum State {
	STATE_RUNNING,
	STATE_PAUSED,
};

volatile enum State state = STATE_RUNNING;
mutex state_mutex;

volatile bool has_signal = false;

void sigint_handler(int signo) {
	has_signal = true;
}

void parse_damon_trace_line(string &line, struct damon_region &region,
                            uint64_t &target_id, uint64_t &nr_regions) {
	regex pattern(".*damon_aggregated: target_id=([0-9]+) nr_regions=([0-9]+) ([0-9]+)-([0-9]+): "
	              "([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+) ([-0-9]+)");

	smatch matches;
	bool can_match = regex_search(line, matches, pattern);
	if (!can_match) {
		cout << "Error: cannot match damon_aggregated line: " << line << endl;
		BUG_ON(true);
	}

	target_id = stol(matches.str(1));
	nr_regions = stol(matches.str(2));
	region.start = stol(matches.str(3));
	region.end = stol(matches.str(4));
	region.nr_accesses = stol(matches.str(5));
	region.age = stol(matches.str(6));
	region.sampling_addr = stol(matches.str(7));
	region.region_id = stol(matches.str(8));
	// last_aggregation = stol(matches.str(9));
}

string get_damon_trace_line(FILE *trace_file, char *trace_buf) {
	// Caller should ensure that there is data in the trace file
	size_t trace_buf_size = TRACE_BUF_SIZE;
	ssize_t num_bytes_read = getline(&trace_buf, &trace_buf_size, trace_file);
	BUG_ON(num_bytes_read <= 0);
	BUG_ON(trace_buf_size != TRACE_BUF_SIZE);  // Resize should not happen
	return string(trace_buf);
}

bool damon_trace_has_data(FILE *trace_file) {
	int fd = fileno(trace_file);

	fd_set read_set;
	FD_ZERO(&read_set);
	FD_SET(fd, &read_set);

	struct timeval timeout;
	timeout.tv_sec = 0;
	timeout.tv_usec = 0;
	int ready = select(fd + 1, &read_set, NULL, NULL, &timeout);
	return ready > 0;
}

void update_damon_regions(FILE *trace_file, vector<struct VM> &vm_vec, ofstream &damon_log_file, uint64_t epoch) {
	if (!damon_trace_has_data(trace_file))
		return;

	char *trace_buf = (char *) malloc(TRACE_BUF_SIZE);  // TODO: Use a global buffer
	BUG_ON(trace_buf == NULL);
	do {
		string line = get_damon_trace_line(trace_file, trace_buf);

		struct damon_region region;
		uint64_t target_id, nr_regions;
		parse_damon_trace_line(line, region, target_id, nr_regions);
		BUG_ON(target_id != 0);
		BUG_ON(region.region_id != 0);
		BUG_ON(nr_regions == 0);

		int num_active_vms = 0;
		for (int i = 0; i < vm_vec.size(); ++i) {
			struct VM *vm = &vm_vec[i];
			if (vm->pid < 0)
				continue;
			else
				num_active_vms++;
			vm->damon_region_vec->clear();

			if (num_active_vms != 1) {
				// Only the first line is already read
				line = get_damon_trace_line(trace_file, trace_buf);
				parse_damon_trace_line(line, region, target_id, nr_regions);
			}
			for (uint64_t j = 0; j < nr_regions; ++j) {
				if (j != 0) {
					// Only the first line is already read
					line = get_damon_trace_line(trace_file, trace_buf);
					parse_damon_trace_line(line, region, target_id, nr_regions);
				}
				BUG_ON(target_id != num_active_vms - 1);
				BUG_ON(region.region_id != j);
				vm->damon_region_vec->push_back(region);
			}
		}
	} while (damon_trace_has_data(trace_file));
	free(trace_buf);

	// Sort DAMON regions
	for (int i = 0; i < vm_vec.size(); ++i) {
		struct VM *vm = &vm_vec[i];
		if (vm->pid < 0)
			continue;
		sort(vm->damon_region_vec->begin(), vm->damon_region_vec->end(),
		     [&](struct damon_region a, struct damon_region b) {
			if (a.nr_accesses == b.nr_accesses)
				return a.age > b.age;
			else
				return a.nr_accesses > b.nr_accesses;
		});
	}

	// Log DAMON regions
	for (int i = 0; i < vm_vec.size(); ++i) {
		struct VM *vm = &vm_vec[i];
		if (vm->pid < 0)
			continue;
		for (auto &region : *vm->damon_region_vec) {
			uint64_t num_uncontended_pages = 0;
			for (uint64_t virt_addr = ROUND_UP(region.start, COLOR_PAGE_SIZE); virt_addr < region.end; virt_addr += COLOR_PAGE_SIZE) {
				if (vm->virt_addr_map_uncontended->find(virt_addr) != vm->virt_addr_map_uncontended->end()) {
					num_uncontended_pages += 1;
				}
			}
			damon_log_file << epoch << "," << i << "," << region.start << "," << region.end << ","
			               << region.nr_accesses << "," << region.age << "," << region.sampling_addr << ","
			               << region.region_id << "," << num_uncontended_pages << endl;
		}
	}
}

void exchange_pages_between_vms(vector<struct VM> &vm_vec, int selected_vm_idx, int victim_vm_idx, int num_pages_to_exchange, default_random_engine &rand_eng) {
	if (num_pages_to_exchange == 0)
		return;
	if (selected_vm_idx == victim_vm_idx)
		return;
	struct VM *selected_vm = &vm_vec[selected_vm_idx];
	struct VM *victim_vm = &vm_vec[victim_vm_idx];
	BUG_ON(num_pages_to_exchange < 0);
	BUG_ON(num_pages_to_exchange % CONTENDED_PAGE_DEGREE != 0);
	BUG_ON(selected_vm->virt_addr_vec_contended->size() < num_pages_to_exchange);
	BUG_ON(victim_vm->virt_addr_vec_uncontended->size() < num_pages_to_exchange);

	uint64_t *selected_contended = selected_vm->virt_addr_vec_contended->data()
	                               + selected_vm->virt_addr_vec_contended->size()
	                               - num_pages_to_exchange;
	uint64_t *victim_uncontended = victim_vm->virt_addr_vec_uncontended->data()
	                               + victim_vm->virt_addr_vec_uncontended->size()
	                               - num_pages_to_exchange;
	struct page_xchg_stats stats;

	uint64_t *skipped_page_arr_1 = (uint64_t *) malloc(num_pages_to_exchange * sizeof(*skipped_page_arr_1));
	BUG_ON(skipped_page_arr_1 == NULL);
	uint64_t *skipped_page_arr_2 = (uint64_t *) malloc(num_pages_to_exchange * sizeof(*skipped_page_arr_2));
	BUG_ON(skipped_page_arr_2 == NULL);

	exchange_pages(selected_vm->pid, victim_vm->pid, num_pages_to_exchange,
	               (void **) selected_contended, (void **) victim_uncontended,
	               (void **) skipped_page_arr_1, (void **) skipped_page_arr_2, &stats);
	BUG_ON(stats.num_succeeded != num_pages_to_exchange << (COLOR_PAGE_SHIFT - PAGE_SHIFT));

	free(skipped_page_arr_1);
	free(skipped_page_arr_2);

	selected_vm->virt_addr_vec_uncontended->insert(selected_vm->virt_addr_vec_uncontended->end(),
	                                               selected_contended,
	                                               selected_contended + num_pages_to_exchange);
	for (uint64_t j = 0; j < num_pages_to_exchange; j++)
		(*selected_vm->virt_addr_map_uncontended)[selected_contended[j]] = selected_vm->virt_addr_vec_uncontended->size() - num_pages_to_exchange + j;
	virt_addr_shuffle_last_n_elem(*selected_vm->virt_addr_vec_uncontended, *selected_vm->virt_addr_map_uncontended, rand_eng, 1, num_pages_to_exchange);
	victim_vm->virt_addr_vec_contended->insert(victim_vm->virt_addr_vec_contended->end(),
	                                           victim_uncontended,
	                                           victim_uncontended + num_pages_to_exchange);
	for (uint64_t j = 0; j < num_pages_to_exchange; j++)
		(*victim_vm->virt_addr_map_contended)[victim_uncontended[j]] = victim_vm->virt_addr_vec_contended->size() - num_pages_to_exchange + j;
	virt_addr_shuffle_last_n_elem(*victim_vm->virt_addr_vec_contended, *victim_vm->virt_addr_map_contended, rand_eng, CONTENDED_PAGE_DEGREE, num_pages_to_exchange / CONTENDED_PAGE_DEGREE);

	for (uint64_t j = 0; j < num_pages_to_exchange; j++)
		(*selected_vm->virt_addr_map_contended).erase(selected_contended[j]);
	for (uint64_t j = 0; j < num_pages_to_exchange; j++)
		(*victim_vm->virt_addr_map_uncontended).erase(victim_uncontended[j]);

	selected_vm->virt_addr_vec_contended->resize(selected_vm->virt_addr_vec_contended->size() - num_pages_to_exchange);
	victim_vm->virt_addr_vec_uncontended->resize(victim_vm->virt_addr_vec_uncontended->size() - num_pages_to_exchange);
}

void borrow_and_donate_pages(vector<struct VM> &vm_vec, unordered_map<int, uint64_t> &borrower_map, unordered_map<int, uint64_t> &donator_map,
                             ofstream &exchange_log_file, uint64_t epoch, default_random_engine &rand_eng) {
	vector<int> borrower_vec;
	vector<int> donator_vec;

	for (auto &it : borrower_map)
		borrower_vec.push_back(it.first);
	for (auto &it : donator_map)
		donator_vec.push_back(it.first);

	// Prioritize borrowers that get more pages
	sort(borrower_vec.begin(), borrower_vec.end(), [&](int a, int b) {
		return borrower_map[a] > borrower_map[b];
	});
	// Prioritize donators that provide more pages to minimize the number of page exchange syscalls
	sort(donator_vec.begin(), donator_vec.end(), [&](int a, int b) {
		return donator_map[a] > donator_map[b];
	});

	printf("+ Borrower list:\n");
	for (int i = 0; i < borrower_vec.size(); ++i) {
		int borrower_idx = borrower_vec[i];
		uint64_t num_pages = borrower_map[borrower_idx];
		BUG_ON(num_pages == 0);
		BUG_ON(num_pages % CONTENDED_PAGE_DEGREE != 0);

		printf("+   %d: VM %d, %lu pages (%ld MB)\n", i, borrower_idx, num_pages, (num_pages << COLOR_PAGE_SHIFT) >> 20);
	}
	printf("+ Donator list:\n");
	for (int i = 0; i < donator_vec.size(); ++i) {
		int donator_idx = donator_vec[i];
		uint64_t num_pages = donator_map[donator_idx];
		BUG_ON(num_pages == 0);
		BUG_ON(num_pages % CONTENDED_PAGE_DEGREE != 0);

		printf("+   %d: VM %d, %lu pages (%ld MB)\n", i, donator_idx, num_pages, (num_pages << COLOR_PAGE_SHIFT) >> 20);
	}

	printf("+ Page exchanges:\n");
	int j = 0;
	for (int i = 0; i < borrower_vec.size(); ++i) {
		int borrower_idx = borrower_vec[i];
		uint64_t num_pages = borrower_map[borrower_idx];
		while (num_pages > 0 && j < donator_vec.size()) {
			int donator_idx = donator_vec[j];
			BUG_ON(donator_map[donator_idx] == 0);
			uint64_t num_pages_to_exchange = min(num_pages, donator_map[donator_idx]);
			BUG_ON(num_pages_to_exchange == 0);
			BUG_ON(num_pages_to_exchange % CONTENDED_PAGE_DEGREE != 0);

			exchange_pages_between_vms(vm_vec, borrower_idx, donator_idx, num_pages_to_exchange, rand_eng);
			printf("+   VM %d <- VM %d: %lu (%lu MB)\n", borrower_idx, donator_idx,
			       num_pages_to_exchange, (num_pages_to_exchange << COLOR_PAGE_SHIFT) >> 20);
			exchange_log_file << epoch << "," << borrower_idx << "," << donator_idx << "," << num_pages_to_exchange
			                  << endl;

			num_pages -= num_pages_to_exchange;
			donator_map[donator_idx] -= num_pages_to_exchange;
			if (donator_map[donator_idx] == 0)
				j++;
		}
	}
}

void exchange_pages_within_vm(struct VM *vm, vector<uint64_t> &virt_addr_vec_contended, vector<uint64_t> &virt_addr_vec_uncontended) {
	BUG_ON(virt_addr_vec_contended.size() != virt_addr_vec_uncontended.size());
	if (virt_addr_vec_contended.size() == 0)
		return;

	struct page_xchg_stats stats;
	uint64_t *skipped_page_arr_1 = (uint64_t *) malloc(virt_addr_vec_contended.size() * sizeof(*skipped_page_arr_1));
	BUG_ON(skipped_page_arr_1 == NULL);
	uint64_t *skipped_page_arr_2 = (uint64_t *) malloc(virt_addr_vec_contended.size() * sizeof(*skipped_page_arr_2));
	BUG_ON(skipped_page_arr_2 == NULL);

	exchange_pages(vm->pid, vm->pid, virt_addr_vec_contended.size(),
		(void **) virt_addr_vec_contended.data(), (void **) virt_addr_vec_uncontended.data(),
		(void **) skipped_page_arr_1, (void **) skipped_page_arr_2, &stats);
	BUG_ON(stats.num_succeeded != virt_addr_vec_contended.size() << (COLOR_PAGE_SHIFT - PAGE_SHIFT));
	free(skipped_page_arr_1);
	free(skipped_page_arr_2);

	// Update page vecs and maps
	for (uint64_t j = 0; j < virt_addr_vec_contended.size(); ++j) {
		uint64_t virt_addr_contended = virt_addr_vec_contended[j];
		uint64_t virt_addr_uncontended = virt_addr_vec_uncontended[j];
		uint64_t index_contended = (*vm->virt_addr_map_contended)[virt_addr_contended];
		uint64_t index_uncontended = (*vm->virt_addr_map_uncontended)[virt_addr_uncontended];
		BUG_ON(virt_addr_contended != (*vm->virt_addr_vec_contended)[index_contended]);
		BUG_ON(virt_addr_uncontended != (*vm->virt_addr_vec_uncontended)[index_uncontended]);
		swap((*vm->virt_addr_vec_contended)[index_contended], (*vm->virt_addr_vec_uncontended)[index_uncontended]);
		vm->virt_addr_map_contended->erase(virt_addr_contended);
		vm->virt_addr_map_uncontended->erase(virt_addr_uncontended);
		(*vm->virt_addr_map_contended)[virt_addr_uncontended] = index_contended;
		(*vm->virt_addr_map_uncontended)[virt_addr_contended] = index_uncontended;
	}
}

void min_miss_exchange(vector<struct VM> &vm_vec, double step_ratio, ofstream &policy_log_file, ofstream &exchange_log_file,
                       uint64_t epoch, default_random_engine &rand_eng) {
	vector<int> vm_index_vec;
	uint64_t total_num_pages = 0;
	for (int i = 0; i < vm_vec.size(); ++i) {
		struct VM *vm = &vm_vec[i];
		if (vm->pid < 0)
			continue;
		vm_index_vec.push_back(i);
		total_num_pages += vm_count_all_contended(vm) + vm_count_all_uncontended(vm);
	}
	if (vm_index_vec.size() < 2)
		return;

	// Sort in ascending order
	sort(vm_index_vec.begin(), vm_index_vec.end(),
	     [&](int a, int b) {
		if (vm_vec[a].has_slowdown != vm_vec[b].has_slowdown)
			return vm_vec[b].has_slowdown;
		return (vm_vec[a].ewma_metrics[MIN_MISS_METRIC]
		        < vm_vec[b].ewma_metrics[MIN_MISS_METRIC]);
	});

	uint64_t total_num_pages_to_exchange = (uint64_t) (total_num_pages * step_ratio);
	total_num_pages_to_exchange = ROUND_DOWN(total_num_pages_to_exchange, CONTENDED_PAGE_DEGREE);
	unordered_map<int, uint64_t> borrower_map;
	unordered_map<int, uint64_t> donator_map;
	int donator_i = 0;
	for (int borrower_i = vm_index_vec.size() - 1; borrower_i >= 0 && total_num_pages_to_exchange > 0; --borrower_i) {
		struct VM* borrower = &vm_vec[vm_index_vec[borrower_i]];
		if (!borrower->has_slowdown)
			break;
		uint64_t borrower_num_contended_pages = borrower->virt_addr_vec_contended->size();
		uint64_t borrower_max_exchange = (uint64_t) ((vm_count_all_contended(borrower) + vm_count_all_uncontended(borrower)) * step_ratio);
		borrower_num_contended_pages = min(borrower_num_contended_pages, borrower_max_exchange);
		borrower_num_contended_pages = ROUND_DOWN(borrower_num_contended_pages, CONTENDED_PAGE_DEGREE);
		if (borrower_num_contended_pages == 0)
			continue;

		while (donator_i < borrower_i && total_num_pages_to_exchange > 0 && borrower_num_contended_pages > 0) {
			struct VM* donator = &vm_vec[vm_index_vec[donator_i]];
			uint64_t donator_num_uncontended_pages = donator->virt_addr_vec_uncontended->size();
			uint64_t donator_max_exchange = (uint64_t) ((vm_count_all_contended(donator) + vm_count_all_uncontended(donator)) * step_ratio);
			// Donator might have already donated some pages
			if (donator_map.find(vm_index_vec[donator_i]) != donator_map.end()) {
				BUG_ON(donator_map[vm_index_vec[donator_i]] > donator_num_uncontended_pages);
				BUG_ON(donator_map[vm_index_vec[donator_i]] > donator_max_exchange);
				donator_num_uncontended_pages -= donator_map[vm_index_vec[donator_i]];
				donator_max_exchange -= donator_map[vm_index_vec[donator_i]];
			}
			donator_num_uncontended_pages = min(donator_num_uncontended_pages, donator_max_exchange);
			if (ROUND_DOWN(donator_num_uncontended_pages, CONTENDED_PAGE_DEGREE) == 0) {
				donator_i++;
				continue;
			}

			// Can consider smoothing or limit per-VM exchange rate
			uint64_t num_to_exchange = min(donator_num_uncontended_pages, borrower_num_contended_pages);
			num_to_exchange = min(num_to_exchange, total_num_pages_to_exchange);
			num_to_exchange = ROUND_DOWN(num_to_exchange, CONTENDED_PAGE_DEGREE);
			BUG_ON(num_to_exchange == 0);
			borrower_map[vm_index_vec[borrower_i]] += num_to_exchange;
			donator_map[vm_index_vec[donator_i]] += num_to_exchange;
			total_num_pages_to_exchange -= num_to_exchange;
			borrower_num_contended_pages -= num_to_exchange;
		}
	}

	for (int i : vm_index_vec) {
		struct VM *vm = &vm_vec[i];
		printf("+ VM %d: Metric (EWMA) = %lf, Borrow = %ld, Donate = %ld\n",
		       i, vm->ewma_metrics[MIN_MISS_METRIC],
		       borrower_map.find(i) == borrower_map.end() ? 0 : borrower_map[i],
		       donator_map.find(i) == donator_map.end() ? 0 : donator_map[i]);
		policy_log_file << epoch << "," << i << "," << vm->ewma_metrics[MIN_MISS_METRIC] << ","
		                << (borrower_map.find(i) == borrower_map.end() ? 0 : borrower_map[i]) << ","
		                << (donator_map.find(i) == donator_map.end() ? 0 : donator_map[i]) << endl;
	}
	borrow_and_donate_pages(vm_vec, borrower_map, donator_map, exchange_log_file, epoch, rand_eng);
}

inline double vm_norm_credits(struct VM *vm) {
	return (double) vm->credits / (double) vm->memory_size_MB;
}

inline double vm_norm_alloc(struct VM *vm, long allocation) {
	return (double) allocation / (double) vm->memory_size_MB;
}

void karma_exchange(vector<struct VM> &vm_vec, double step_ratio, int karma_unit,
                    ofstream &policy_log_file, ofstream &exchange_log_file,
                    uint64_t epoch, default_random_engine &rand_eng) {
	vector<int> vm_index_vec;
	uint64_t total_num_pages = 0;
	for (int i = 0; i < vm_vec.size(); ++i) {
		struct VM *vm = &vm_vec[i];
		if (vm->pid < 0)
			continue;
		vm_index_vec.push_back(i);
		total_num_pages += vm_count_all_contended(vm) + vm_count_all_uncontended(vm);
	}
	if (vm_index_vec.size() < 2)
		return;

	// Calculate demands
	unordered_map<int, long> demand_map;
	for (int i : vm_index_vec) {
		struct VM *vm = &vm_vec[i];
		BUG_ON(vm->ewma_avg_karma_metric < 0 || vm->ewma_metrics[KARMA_METRIC] < 0);
		if (vm->ewma_avg_karma_metric == 0) {
			demand_map[i] = (long) vm->fair_share;
		} else {
			demand_map[i] = (long) ((double) vm->fair_share * vm->ewma_metrics[KARMA_METRIC] / vm->ewma_avg_karma_metric);
		}
		// Make sure demand is always achievable
		demand_map[i] = min(demand_map[i], (long) (vm_count_all_uncontended(vm) + vm_count_all_contended(vm) - vm->num_excluded_contended));
		demand_map[i] = max(demand_map[i], (long) vm->num_excluded_uncontended);
	}

	// Reuse the previous allocation and update credits
	unordered_map<int, long> allocation_map;
	for (int i : vm_index_vec) {
		struct VM *vm = &vm_vec[i];
		vm->credits += ((long) vm->fair_share) - ((long) vm_count_all_uncontended(vm));
		allocation_map[i] = vm_count_all_uncontended(vm);
	}

	// Adjust the current allocation based on demands and credits
	long num_pages_to_reallocate = (long) (total_num_pages * step_ratio);
	while (num_pages_to_reallocate > 0) {
		long cur_unit = min(num_pages_to_reallocate, (long) karma_unit);

		// Find the borrower with most credits
		int borrower_idx = -1;
		double max_norm_credits = 0;
		for (int i : vm_index_vec) {
			struct VM *vm = &vm_vec[i];
			if (demand_map[i] > allocation_map[i] && vm->credits > 0) {
				if (vm_norm_credits(vm) > max_norm_credits) {
					borrower_idx = i;
					max_norm_credits = vm_norm_credits(vm);
				}
			}
		}

		if (borrower_idx != -1) {
			struct VM *borrower_vm = &vm_vec[borrower_idx];
			cur_unit = min(cur_unit, demand_map[borrower_idx] - allocation_map[borrower_idx]);
			cur_unit = min(cur_unit, vm_vec[borrower_idx].credits);
			BUG_ON(cur_unit <= 0);

			// Find the voluntary donator with fewest credits
			int donator_idx = -1;
			double min_norm_credits = numeric_limits<double>::max();
			for (int i : vm_index_vec) {
				struct VM *vm = &vm_vec[i];
				if (i == borrower_idx)
					continue;
				if (allocation_map[i] > demand_map[i]) {
					if (vm_norm_credits(vm) < min_norm_credits) {
						donator_idx = i;
						min_norm_credits = vm_norm_credits(vm);
					}
				}
			}
			if (donator_idx != -1) {
				// Do not hurt the voluntary donator
				cur_unit = min(cur_unit, allocation_map[donator_idx] - demand_map[donator_idx]);
			}

			// Cannot find a voluntary donator, find an involuntary donator
			if (donator_idx == -1) {
				double min_norm_credits = numeric_limits<double>::max();
				for (int i : vm_index_vec) {
					struct VM *vm = &vm_vec[i];
					if (i == borrower_idx)
						continue;
					if (allocation_map[i] > (long) vm->num_excluded_uncontended
					    && vm_norm_credits(vm) < vm_norm_credits(borrower_vm)) {
						if (vm_norm_credits(vm) < min_norm_credits) {
							donator_idx = i;
							min_norm_credits = vm_norm_credits(vm);
						}
					}
				}
				if (donator_idx != -1) {
					cur_unit = min(cur_unit, allocation_map[donator_idx] - (long) vm_vec[donator_idx].num_excluded_uncontended);
				}
			}

			if (donator_idx != -1) {
				// Find a borrower-donator pair, update allocations and credits
				BUG_ON(cur_unit <= 0);
				allocation_map[borrower_idx] += cur_unit;
				allocation_map[donator_idx] -= cur_unit;
				vm_vec[borrower_idx].credits -= cur_unit;
				vm_vec[donator_idx].credits += cur_unit;
				num_pages_to_reallocate -= cur_unit;
				continue;
			}
		}

		// Cannot find a borrower-donator pair, adjust the allocation towards the fair-share scheme
		cur_unit = min(num_pages_to_reallocate, (long) karma_unit);

		// Find the recipient with smallest allocation
		int recipient_idx = -1;
		double min_norm_alloc = numeric_limits<double>::max();
		for (int i : vm_index_vec) {
			struct VM *vm = &vm_vec[i];
			if (allocation_map[i] < (long) (vm->fair_share) && vm->credits > 0) {
				if (vm_norm_alloc(vm, allocation_map[i]) < min_norm_alloc) {
					recipient_idx = i;
					min_norm_alloc = vm_norm_alloc(vm, allocation_map[i]);
				}
			}
		}
		if (recipient_idx != -1) {
			struct VM *recipient_vm = &vm_vec[recipient_idx];
			cur_unit = min(cur_unit, (long) (recipient_vm->fair_share) - allocation_map[recipient_idx]);
			cur_unit = min(cur_unit, recipient_vm->credits);
			BUG_ON(cur_unit <= 0);

			// Find the provider with largest allocation
			int provider_idx = -1;
			double max_norm_alloc = 0;
			for (int i : vm_index_vec) {
				struct VM *vm = &vm_vec[i];
				if (i == recipient_idx)
					continue;
				/*
				 * Only consider the VMs that satisfy the following conditions:
				 * 1. The VM's allocation is larger than its fair-share
				 * 2. The VM either has negative credits or has more allocation than demand
				 * 3. The VM has uncontended pages to provide
				 */
				if (allocation_map[i] > (long) (vm->fair_share)
				    && (vm->credits < 0 || allocation_map[i] > demand_map[i])
				    && (allocation_map[i] > vm->num_excluded_uncontended)) {
					if (vm_norm_alloc(vm, allocation_map[i]) > max_norm_alloc) {
						provider_idx = i;
						max_norm_alloc = vm_norm_alloc(vm, allocation_map[i]);
					}
				}
			}
			if (provider_idx != -1) {
				struct VM *provider_vm = &vm_vec[provider_idx];
				cur_unit = min(cur_unit, allocation_map[provider_idx] - (long) provider_vm->fair_share);
				if (provider_vm->credits >= 0) {
					cur_unit = min(cur_unit, allocation_map[provider_idx] - demand_map[provider_idx]);
				} else {
					cur_unit = min(cur_unit, -provider_vm->credits);
					cur_unit = min(cur_unit, allocation_map[provider_idx] - (long) provider_vm->num_excluded_uncontended);
				}

				// Update allocations and credits
				BUG_ON(cur_unit <= 0);
				allocation_map[recipient_idx] += cur_unit;
				allocation_map[provider_idx] -= cur_unit;
				vm_vec[recipient_idx].credits -= cur_unit;
				vm_vec[provider_idx].credits += cur_unit;
				num_pages_to_reallocate -= cur_unit;
			} else {
				break;
			}
		} else {
			break;
		}
	}

	// Exchange pages based on the allocation map
	unordered_map<int, uint64_t> borrower_map;
	unordered_map<int, uint64_t> donator_map;
	for (int i : vm_index_vec) {
		struct VM *vm = &vm_vec[i];
		long delta = allocation_map[i] - (long) vm_count_all_uncontended(vm);
		BUG_ON(allocation_map[i] < (long) vm->num_excluded_uncontended);
		BUG_ON(allocation_map[i] > (long) (vm_count_all_uncontended(vm) + vm_count_all_contended(vm) - vm->num_excluded_contended));
		uint64_t num_pages;
		if (delta > 0) {
			BUG_ON(delta > (long) vm->virt_addr_vec_contended->size());
			num_pages = ROUND_DOWN((uint64_t) delta, CONTENDED_PAGE_DEGREE);
			if (num_pages > 0) {
				borrower_map[i] = num_pages;
			}
			// Rectify credits error caused by rounding
			if (num_pages < delta) {
				vm->credits += delta - (long) num_pages;
			}
		} else if (delta < 0) {
			BUG_ON(-delta > (long) vm->virt_addr_vec_uncontended->size());
			num_pages = ROUND_DOWN((uint64_t) (-delta), CONTENDED_PAGE_DEGREE);
			if (num_pages > 0) {
				donator_map[i] = num_pages;
			}
			// Rectify credits error caused by rounding
			if (num_pages < (long) (-delta)) {
				vm->credits -= (-delta) - (long) num_pages;
			}
		}
		printf("+ VM %d (Total %ld MB, Fair Share %ld MB): Metric (EWMA) = %lf, EWMA Average = %lf, Demand = %ld (%ld MB), Allocation = %ld (%ld MB), Credits = %ld (%ld MB)\n",
		       i, vm->memory_size_MB, (vm->fair_share << COLOR_PAGE_SHIFT) >> 20, vm->ewma_metrics[KARMA_METRIC], vm->ewma_avg_karma_metric,
		       demand_map[i], (demand_map[i] << COLOR_PAGE_SHIFT) >> 20,
		       allocation_map[i], (allocation_map[i] << COLOR_PAGE_SHIFT) >> 20, vm->credits, (vm->credits << COLOR_PAGE_SHIFT) >> 20);
		policy_log_file << epoch << "," << i << "," << vm->ewma_metrics[KARMA_METRIC] << "," << vm->ewma_avg_karma_metric << ","
		                << demand_map[i] << "," << allocation_map[i] << "," << vm->credits << ","
		                << vm->virt_addr_vec_uncontended->size() << "," << vm->virt_addr_vec_contended->size() << ","
		                << vm->num_excluded_uncontended << "," << vm->num_excluded_contended << ","
		                << delta << "," << num_pages << endl;
	}
	borrow_and_donate_pages(vm_vec, borrower_map, donator_map, exchange_log_file, epoch, rand_eng);
}

void hot_cold_separation(vector<struct VM> &vm_vec, int hot_threshold, int cold_threshold, bool random_hot_cold,
                         ofstream &exchange_log_file, int epoch, default_random_engine &rand_eng) {
	for (int i = 0; i < vm_vec.size(); ++i) {
		struct VM *vm = &vm_vec[i];
		if (vm->pid < 0)
			continue;

		int hot_cold_step_size = vm->hot_cold_step_size;
		if (hot_cold_step_size == 0)
			continue;

		// Search hot contended pages
		vector<uint64_t> hot_contended_virt_addr_vec;
		uint64_t hot_region_index = 0;
		uint64_t hot_region_nr_accesses = 0;
		for (uint64_t j = 0; j < vm->damon_region_vec->size(); ++j) {
			if (hot_contended_virt_addr_vec.size() >= hot_cold_step_size)
				break;
			struct damon_region *region = &(*vm->damon_region_vec)[j];
			if (region->nr_accesses < hot_threshold)
				break;
			for (uint64_t virt_addr = ROUND_UP(region->start, COLOR_PAGE_SIZE); virt_addr < region->end; virt_addr += COLOR_PAGE_SIZE) {
				if (hot_contended_virt_addr_vec.size() >= hot_cold_step_size)
					break;
				BUG_ON(virt_addr % COLOR_PAGE_SIZE != 0);
				if (vm->virt_addr_map_contended->find(virt_addr) != vm->virt_addr_map_contended->end()) {
					hot_contended_virt_addr_vec.push_back(virt_addr);
					hot_region_index = j;
					hot_region_nr_accesses = region->nr_accesses;
				}
				// TODO: Can use random to choose pages when tied
			}
		}
		BUG_ON(hot_contended_virt_addr_vec.size() > hot_cold_step_size);
		if (hot_contended_virt_addr_vec.empty()) {
			printf("- No hot contended pages in VM %d\n", i);
			continue;
		}

		// Search cold uncontended pages
		vector<uint64_t> cold_uncontended_virt_addr_vec;
		for (uint64_t j = vm->damon_region_vec->size() - 1; j > hot_region_index; --j) {
			if (cold_uncontended_virt_addr_vec.size() >= hot_contended_virt_addr_vec.size())
				break;
			struct damon_region *region = &(*vm->damon_region_vec)[j];
			if (region->nr_accesses > cold_threshold)
				break;
			if (region->nr_accesses >= hot_region_nr_accesses)
				break;
			for (uint64_t virt_addr = ROUND_UP(region->start, COLOR_PAGE_SIZE); virt_addr < region->end; virt_addr += COLOR_PAGE_SIZE) {
				if (cold_uncontended_virt_addr_vec.size() >= hot_contended_virt_addr_vec.size())
					break;
				BUG_ON(virt_addr % COLOR_PAGE_SIZE != 0);
				if (vm->virt_addr_map_uncontended->find(virt_addr) != vm->virt_addr_map_uncontended->end())
					cold_uncontended_virt_addr_vec.push_back(virt_addr);
			}
		}
		BUG_ON(cold_uncontended_virt_addr_vec.size() > hot_contended_virt_addr_vec.size());
		hot_contended_virt_addr_vec.resize(cold_uncontended_virt_addr_vec.size());
		if (hot_contended_virt_addr_vec.empty()) {
			printf("- No cold uncontended pages in VM %d\n", i);
			continue;
		}
		BUG_ON(hot_contended_virt_addr_vec.size() != cold_uncontended_virt_addr_vec.size());

		if (random_hot_cold) {
			// Debug only
			printf("- Randomly selecting hot and cold pages to exchange in VM %d\n", i);
			uint64_t num_pages = hot_contended_virt_addr_vec.size();
			hot_contended_virt_addr_vec.clear();
			cold_uncontended_virt_addr_vec.clear();

			hot_contended_virt_addr_vec.insert(hot_contended_virt_addr_vec.end(),
				vm->virt_addr_vec_contended->begin(),
				vm->virt_addr_vec_contended->end());
			cold_uncontended_virt_addr_vec.insert(cold_uncontended_virt_addr_vec.end(),
				vm->virt_addr_vec_uncontended->begin(),
				vm->virt_addr_vec_uncontended->end());
			shuffle(hot_contended_virt_addr_vec.begin(), hot_contended_virt_addr_vec.end(), rand_eng);
			shuffle(cold_uncontended_virt_addr_vec.begin(), cold_uncontended_virt_addr_vec.end(), rand_eng);
			hot_contended_virt_addr_vec.resize(num_pages);
			cold_uncontended_virt_addr_vec.resize(num_pages);
		}

		exchange_pages_within_vm(vm, hot_contended_virt_addr_vec, cold_uncontended_virt_addr_vec);
		printf("- Exchanged %lu pages (%ld MB) in VM %d for hot-cold separation\n",
		       hot_contended_virt_addr_vec.size(), (hot_contended_virt_addr_vec.size() << COLOR_PAGE_SHIFT) >> 20, i);
		exchange_log_file << epoch << "," << i << "," << hot_contended_virt_addr_vec.size() << endl;
	}
}

void fill_vm(vector<struct VM> *vm_vec, int new_vm_index, uint64_t target_num_uncontended_pages,
             ofstream &exchange_log_file, default_random_engine &rand_eng,
             chrono::time_point<chrono::steady_clock> start_time) {
	unordered_map<int, uint64_t> borrower_map;
	unordered_map<int, uint64_t> donator_map;

	vector<int> vm_index_vec;
	uint64_t total_num_movable_uncontended_pages = 0;
	uint64_t total_num_movable_contended_pages = 0;
	for (int i = 0; i < vm_vec->size(); ++i) {
		struct VM *vm = &(*vm_vec)[i];
		if (vm->pid < 0)
			continue;
		if (i == new_vm_index)
			continue;
		vm_index_vec.push_back(i);
		total_num_movable_uncontended_pages += vm->virt_addr_vec_uncontended->size();
		total_num_movable_contended_pages += vm->virt_addr_vec_contended->size();
	}

	struct VM *new_vm = &(*vm_vec)[new_vm_index];
	if (target_num_uncontended_pages > vm_count_all_uncontended(new_vm)) {
		// Evenly steal uncontended pages from other VMs
		uint64_t total_num_pages_to_move = target_num_uncontended_pages - vm_count_all_uncontended(new_vm);
		total_num_pages_to_move = min(total_num_pages_to_move, new_vm->virt_addr_vec_contended->size());
		total_num_pages_to_move = min(total_num_pages_to_move, total_num_movable_uncontended_pages);
		total_num_pages_to_move = ROUND_DOWN(total_num_pages_to_move, CONTENDED_PAGE_DEGREE);
		if (total_num_pages_to_move == 0)
			return;

		borrower_map[new_vm_index] = total_num_pages_to_move;
		sort(vm_index_vec.begin(), vm_index_vec.end(),
		     [&](int a, int b) {
			return (*vm_vec)[a].virt_addr_vec_uncontended->size()
			       < (*vm_vec)[b].virt_addr_vec_uncontended->size();
		});
		for (int ii = 0; ii < vm_index_vec.size() && total_num_pages_to_move > 0; ++ii) {
			struct VM *vm = &(*vm_vec)[vm_index_vec[ii]];
			uint64_t num_pages_to_move = vm->virt_addr_vec_uncontended->size();
			num_pages_to_move = min(num_pages_to_move, total_num_pages_to_move / (vm_index_vec.size() - ii));
			num_pages_to_move = ROUND_DOWN(num_pages_to_move, CONTENDED_PAGE_DEGREE);
			if (num_pages_to_move == 0)
				continue;
			donator_map[vm_index_vec[ii]] = num_pages_to_move;
			total_num_pages_to_move -= num_pages_to_move;
		}
		BUG_ON(total_num_pages_to_move != 0);
	} else if (target_num_uncontended_pages < vm_count_all_uncontended(new_vm)) {
		// Evenly give uncontended pages away to other VMs
		uint64_t total_num_pages_to_move = vm_count_all_uncontended(new_vm) - target_num_uncontended_pages;
		total_num_pages_to_move = min(total_num_pages_to_move, new_vm->virt_addr_vec_uncontended->size());
		total_num_pages_to_move = min(total_num_pages_to_move, total_num_movable_contended_pages);
		total_num_pages_to_move = ROUND_DOWN(total_num_pages_to_move, CONTENDED_PAGE_DEGREE);
		if (total_num_pages_to_move == 0)
			return;

		donator_map[new_vm_index] = total_num_pages_to_move;
		sort(vm_index_vec.begin(), vm_index_vec.end(),
		     [&](int a, int b) {
			return (*vm_vec)[a].virt_addr_vec_contended->size()
			       < (*vm_vec)[b].virt_addr_vec_contended->size();
		});
		for (int ii = 0; ii < vm_index_vec.size() && total_num_pages_to_move > 0; ++ii) {
			struct VM *vm = &(*vm_vec)[vm_index_vec[ii]];
			uint64_t num_pages_to_move = vm->virt_addr_vec_contended->size();
			num_pages_to_move = min(num_pages_to_move, total_num_pages_to_move / (vm_index_vec.size() - ii));
			num_pages_to_move = ROUND_DOWN(num_pages_to_move, CONTENDED_PAGE_DEGREE);
			if (num_pages_to_move == 0)
				continue;
			borrower_map[vm_index_vec[ii]] = num_pages_to_move;
			total_num_pages_to_move -= num_pages_to_move;
		}
		BUG_ON(total_num_pages_to_move != 0);
	}
	if (!borrower_map.empty()) {
		auto cur_time = chrono::steady_clock::now();
		uint64_t epoch = chrono::duration_cast<chrono::seconds>(cur_time - start_time).count();

		printf("+ Filling new VM %d: Borrow = %ld, Donate = %ld\n",
		       new_vm_index,
		       borrower_map.find(new_vm_index) == borrower_map.end() ? 0 : borrower_map[new_vm_index],
		       donator_map.find(new_vm_index) == donator_map.end() ? 0 : donator_map[new_vm_index]);
		borrow_and_donate_pages(*vm_vec, borrower_map, donator_map, exchange_log_file, epoch, rand_eng);
	}
}

void mq_thread_fn(mqd_t request_queue, mqd_t response_queue, vector<struct VM> *vm_vec,
                  bool is_1lm, double dram_pct, int karma_init_credit_factor, ofstream *exchange_log_file,
				  chrono::time_point<chrono::steady_clock> start_time) {
	char buffer[MQ_BUFFER_SIZE];
	struct timespec timeout;
	default_random_engine rand_eng(0xBEE);
	while (!has_signal) {
		clock_gettime(CLOCK_REALTIME, &timeout);
		timeout.tv_sec += 1;
		ssize_t num_bytes = mq_timedreceive(request_queue, buffer, MQ_BUFFER_SIZE, NULL, &timeout);
		if (num_bytes < 0) {
			BUG_ON(errno != ETIMEDOUT && errno != EINTR);
			continue;
		}

		string message(buffer);
		istringstream sstream(message);
		string token;
		sstream >> token;
		if (token == "pause") {
			unique_lock<mutex> lock(state_mutex);
			BUG_ON(state != STATE_RUNNING);
			state = STATE_PAUSED;
			lock.unlock();

			string response = "pause";
			BUG_ON(mq_send(response_queue, response.c_str(), response.size() + 1, 0) == -1);
		} else if (token == "resume") {
			unique_lock<mutex> lock(state_mutex);
			BUG_ON(state != STATE_PAUSED);
			state = STATE_RUNNING;
			lock.unlock();

			string response = "resume";
			BUG_ON(mq_send(response_queue, response.c_str(), response.size() + 1, 0) == -1);
		} else if (token == "remove") {
			unique_lock<mutex> lock(state_mutex);
			sstream >> token;
			int pid = stoi(token);
			BUG_ON(pid < 0);
			for (int i = 0; i < vm_vec->size(); ++i) {
				struct VM *vm = &(*vm_vec)[i];
				if (vm->pid == pid) {
					teardown_vm(vm);
				}
			}
			lock.unlock();

			string response = "remove";
			BUG_ON(mq_send(response_queue, response.c_str(), response.size() + 1, 0) == -1);
		} else if (token == "add") {
			unique_lock<mutex> lock(state_mutex);

			sstream >> token;
			int pid = stoi(token);
			BUG_ON(pid < 0);

			sstream >> token;
			string core_str = token;

			sstream >> token;
			uint64_t base_virt_addr = stoull(token);
			BUG_ON(base_virt_addr == 0);
			BUG_ON(base_virt_addr % COLOR_PAGE_SIZE != 0);

			sstream >> token;
			uint64_t memory_size_MB = stoull(token);

			sstream >> token;
			int hot_cold_step_size = stoi(token);
			BUG_ON(hot_cold_step_size < 0);

			vm_vec->resize(vm_vec->size() + 1);
			struct VM *vm = &vm_vec->back();
			setup_vm(vm, pid, core_str, base_virt_addr, memory_size_MB,
			         rand_eng, is_1lm, dram_pct, hot_cold_step_size,
			         karma_init_credit_factor);
			lock.unlock();

			string response = "add";
			BUG_ON(mq_send(response_queue, response.c_str(), response.size() + 1, 0) == -1);
		} else if (token == "fill") {
			unique_lock<mutex> lock(state_mutex);

			sstream >> token;
			int pid = stoi(token);
			BUG_ON(pid < 0);

			sstream >> token;
			uint64_t target_num_uncontended_pages = stoull(token);

			for (int i = 0; i < vm_vec->size(); ++i) {
				struct VM *vm = &(*vm_vec)[i];
				if (vm->pid == pid) {
					BUG_ON(target_num_uncontended_pages > vm_count_all_contended(vm) + vm_count_all_uncontended(vm));
					fill_vm(vm_vec, i, target_num_uncontended_pages, *exchange_log_file, rand_eng, start_time);
				}
			}
			lock.unlock();

			string response = "fill";
			BUG_ON(mq_send(response_queue, response.c_str(), response.size() + 1, 0) == -1);
		} else {
			printf("Unknown message: %s\n", buffer);
		}
	}
}

bool rf_predict_slowdown(Ort::Session &session, std::vector<const char*> &input_names, std::vector<const char*> &output_names,
                         std::vector<int64_t> &input_dims, std::vector<int64_t> &output_dims, struct VM *vm) {
	std::vector<float> inputTensorValues;
	std::vector<int64_t> outputTensorValues(1);

	inputTensorValues.push_back(vm->ewma_metrics[METRIC_TYPE_DTLB_LOAD_MISS_LAT]);
	// inputTensorValues.push_back(vm->ewma_metrics[METRIC_TYPE_DLTB_LOAD_MPI]);
	inputTensorValues.push_back(vm->ewma_metrics[METRIC_TYPE_LOAD_L3_MISS_LAT]);
	inputTensorValues.push_back(vm->ewma_metrics[METRIC_TYPE_LOAD_L2_MISS_LAT]);
	inputTensorValues.push_back(vm->ewma_metrics[METRIC_TYPE_L2_DEMAND_DATA_RD_MPI]);
	// inputTensorValues.push_back(vm->ewma_metrics[METRIC_TYPE_L2_MPI]);
	inputTensorValues.push_back(vm->ewma_metrics[METRIC_TYPE_2LM_MPKI]);

	std::vector<Ort::Value> inputTensors;
	std::vector<Ort::Value> outputTensors;
	Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
			OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorValues.size(), input_dims.data(),input_dims.size()));
	outputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, outputTensorValues.data(), 1, output_dims.data(), output_dims.size()));

	session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputTensors.data(), 1, output_names.data(), outputTensors.data(), 1);
	int64_t value = *(outputTensors[0].GetTensorMutableData<int64_t>());
	BUG_ON(value != 0 && value != 1);
	return value == 1;
}

void update_rf_prediction(vector<struct VM> &vm_vec, Ort::Session &session, std::vector<const char*> &input_names,
                          std::vector<const char*> &output_names, std::vector<int64_t> &input_dims, std::vector<int64_t> &output_dims) {
	for (int i = 0; i < vm_vec.size(); ++i) {
		struct VM *vm = &vm_vec[i];
		if (vm->pid < 0)
			continue;
		vm->has_slowdown = rf_predict_slowdown(session, input_names, output_names, input_dims, output_dims, vm);
	}
}

void print_help_message(char *argv0) {
	printf("Usage: %s [-h] [-w <operation window>] [-s <step ratio>] "
		"[-p <perf log>] [-e <exchange log>] [-l <VMs log>] [--metric-log <metric log>] "
		"[--policy <min_miss|karma>] [--policy-log <policy log>] [--ewma <EWMA constant>] "
		"[--karma-avg-ewma <EWMA constant for average>] [--karma-avg-size <number of data points in each avg window>] "
		"[--karma-init-credit-factor <initial credit factor in Karma>] [--karma-unit <a unit number of pages in Karma>] "
		"[--hot-cold-step-size <hot cold separation step size list>] [--hot-thres <hot threshold>] "
		"[--cold-thres <cold threshold>] [--damon-log <damon log>] "
		"[--hot-cold-exchange-log <hot cold separation exchange log>] [--random-hot-cold] "
		"[--1lm] [--dram-pct <DRAM percentage>] [--dram-ready-file <DRAM ready file>] "
		"[--req-queue <request message queue>] [--res-queue <response message queue>] "
		"[--rf-path <random forest model path>] "
		"-i <pid list> -c <core list> -v <base virtual addr list> -g <memory size (MB) list>\n",
		argv0);
}

int main(int argc, char** argv) {
	int operation_window_s = 10;  // Operation window: 10 seconds
	double step_ratio = 0.05;
	double ewma_constant = 0.2;

	string perf_log = "./result_perf.log";
	string metric_log = "./result_metric.log";
	string policy_log = "./result_policy.log";
	string exchange_log = "./result_exchange.log";
	string vms_log = "./result_vms.log";
	vector<int> pid_vec;
	vector<string> core_str_vec;
	vector<uint64_t> base_virt_addr_vec;
	vector<uint64_t> memory_size_MB_vec;

	enum PolicyType policy = POLICY_MIN_MISS;

	vector<int> hot_cold_step_size_vec;
	int hot_threshold = 1;
	int cold_threshold = DAMON_MAX_NR_ACCESSES;
	string damon_log = "./result_damon.log";
	string hot_cold_exchange_log = "./result_hot_cold_exchange.log";
	bool random_hot_cold = false;

	double karma_avg_ewma_constant = 0.2;
	int karma_avg_size = 180;
	int karma_init_credit_factor = 1;
	int karma_unit = 8192;

	bool is_1lm = false;
	double dram_pct = 0.5;
	string dram_ready_file = "./result_dram_ready.txt";

	string request_queue_name = "/memstrata_request";
	string response_queue_name = "/memstrata_response";

	string rf_path = "";

	if (argc == 1) {
		print_help_message(argv[0]);
		return -1;
	}
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-h") == 0) {
			print_help_message(argv[0]);
			return 0;
		} else if (strcmp(argv[i], "-w") == 0) {
			i++;
			operation_window_s = atoi(argv[i]);
			BUG_ON(operation_window_s < 0);
		} else if (strcmp(argv[i], "-s") == 0) {
			i++;
			step_ratio = atof(argv[i]);
			BUG_ON(step_ratio < 0 || step_ratio > 0.5);
		} else if (strcmp(argv[i], "-p") == 0) {
			i++;
			perf_log = string(argv[i]);
		} else if (strcmp(argv[i], "-e") == 0) {
			i++;
			exchange_log = string(argv[i]);
		} else if (strcmp(argv[i], "-l") == 0) {
			i++;
			vms_log = string(argv[i]);
		} else if (strcmp(argv[i], "--metric-log") == 0) {
			i++;
			metric_log = string(argv[i]);
		} else if (strcmp(argv[i], "--policy") == 0) {
			i++;
			if (strcmp(argv[i], "min_miss") == 0) {
				policy = POLICY_MIN_MISS;
			} else if (strcmp(argv[i], "karma") == 0) {
				policy = POLICY_KARMA;
			} else {
				printf("Invalid policy: %s\n", argv[i]);
				return -1;
			}
		} else if (strcmp(argv[i], "--policy-log") == 0) {
			i++;
			policy_log = string(argv[i]);
		} else if (strcmp(argv[i], "--ewma") == 0) {
			i++;
			ewma_constant = atof(argv[i]);
			BUG_ON(ewma_constant < 0 || ewma_constant > 1);
		} else if (strcmp(argv[i], "--karma-avg-ewma") == 0) {
			i++;
			karma_avg_ewma_constant = atof(argv[i]);
			BUG_ON(karma_avg_ewma_constant < 0 || karma_avg_ewma_constant > 1);
		} else if (strcmp(argv[i], "--karma-avg-size") == 0) {
			i++;
			karma_avg_size = atoi(argv[i]);
			BUG_ON(karma_avg_size <= 0);
		} else if (strcmp(argv[i], "--karma-init-credit-factor") == 0) {
			i++;
			karma_init_credit_factor = atoi(argv[i]);
			BUG_ON(karma_init_credit_factor <= 0);
		} else if (strcmp(argv[i], "--karma-unit") == 0) {
			i++;
			karma_unit = atoi(argv[i]);
			BUG_ON(karma_unit <= 0);
		} else if (strcmp(argv[i], "--hot-cold-step-size") == 0) {
			i++;
			while (i < argc && argv[i][0] != '-') {
				hot_cold_step_size_vec.push_back(atoi(argv[i]));
				BUG_ON(hot_cold_step_size_vec.back() < 0);
				i++;
			}
			i--;
		} else if (strcmp(argv[i], "--hot-thres") == 0) {
			i++;
			hot_threshold = atoi(argv[i]);
		} else if (strcmp(argv[i], "--cold-thres") == 0) {
			i++;
			cold_threshold = atoi(argv[i]);
		} else if (strcmp(argv[i], "--damon-log") == 0) {
			i++;
			damon_log = string(argv[i]);
		} else if (strcmp(argv[i], "--hot-cold-exchange-log") == 0) {
			i++;
			hot_cold_exchange_log = string(argv[i]);
		} else if (strcmp(argv[i], "--random-hot-cold") == 0) {
			random_hot_cold = true;
		} else if (strcmp(argv[i], "--1lm") == 0) {
			is_1lm = true;
		} else if (strcmp(argv[i], "--dram-pct") == 0) {
			i++;
			dram_pct = atof(argv[i]);
			BUG_ON(dram_pct < 0 || dram_pct > 1);
		} else if (strcmp(argv[i], "--dram-ready-file") == 0) {
			i++;
			dram_ready_file = string(argv[i]);
		} else if (strcmp(argv[i], "--req-queue") == 0) {
			i++;
			request_queue_name = string(argv[i]);
		} else if (strcmp(argv[i], "--res-queue") == 0) {
			i++;
			response_queue_name = string(argv[i]);
		} else if (strcmp(argv[i], "--rf-path") == 0) {
			i++;
			rf_path = string(argv[i]);
		} else if (strcmp(argv[i], "-i") == 0) {
			i++;
			while (i < argc && argv[i][0] != '-') {
				pid_vec.push_back(atoi(argv[i]));
				BUG_ON(pid_vec.back() <= 0);
				i++;
			}
			i--;
		} else if (strcmp(argv[i], "-c") == 0) {
			i++;
			while (i < argc && argv[i][0] != '-') {
				core_str_vec.push_back(string(argv[i]));
				i++;
			}
			i--;
		} else if (strcmp(argv[i], "-v") == 0) {
			i++;
			while (i < argc && argv[i][0] != '-') {
				base_virt_addr_vec.push_back(atol(argv[i]));
				i++;
			}
			i--;
		} else if (strcmp(argv[i], "-g") == 0) {
			i++;
			while (i < argc && argv[i][0] != '-') {
				memory_size_MB_vec.push_back(atol(argv[i]));
				i++;
			}
			i--;
		} else {
			BUG_ON(true);
		}
	}
	BUG_ON(pid_vec.size() != core_str_vec.size());
	BUG_ON(base_virt_addr_vec.size() != pid_vec.size());
	BUG_ON(memory_size_MB_vec.size() != pid_vec.size());
	BUG_ON(!hot_cold_step_size_vec.empty() && hot_cold_step_size_vec.size() != pid_vec.size());
	if (hot_cold_step_size_vec.empty()) {
		for (int i = 0; i < pid_vec.size(); ++i)
			hot_cold_step_size_vec.push_back(0);
	}
	BUG_ON(hot_threshold < 0);
	BUG_ON(cold_threshold < 0);

	// Register SIGINT handler
	BUG_ON(signal(SIGINT, sigint_handler) == SIG_ERR);

	default_random_engine rand_eng(0xBEEF);

	// Set up perf counters and VM states
	vector<struct VM> vm_vec(pid_vec.size());
	for (int i = 0; i < pid_vec.size(); ++i) {
		setup_vm(&vm_vec[i], pid_vec[i], core_str_vec[i],
		         base_virt_addr_vec[i], memory_size_MB_vec[i],
		         rand_eng, is_1lm, dram_pct, hot_cold_step_size_vec[i],
		         karma_init_credit_factor);
	}

	if (is_1lm) {
		ofstream dram_ready_file_stream(dram_ready_file);
		dram_ready_file_stream << "1" << endl;
		dram_ready_file_stream.close();
	}

	ofstream perf_log_file(perf_log);
	ofstream metric_log_file(metric_log);
	ofstream policy_log_file(policy_log);
	ofstream exchange_log_file(exchange_log);
	ofstream vms_log_file(vms_log);
	ofstream damon_log_file(damon_log);
	ofstream hot_cold_exchange_log_file(hot_cold_exchange_log);

	// Open trace pipe
	FILE *trace_file = fopen(TRACE_PIPE_PATH, "r");
	BUG_ON(trace_file == NULL);

	// Open random forest
	Ort::Env ort_env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "memstrata");
	Ort::SessionOptions session_options;
	Ort::AllocatorWithDefaultOptions allocator;
	session_options.SetIntraOpNumThreads(1);
	std::vector<string> input_names_str;
	std::vector<string> output_names_str;
	std::vector<const char*> input_names;
	std::vector<const char*> output_names;
	std::vector<int64_t> input_dims;
	std::vector<int64_t> output_dims;
	Ort::Session *ort_session = NULL;
	if (!rf_path.empty()) {
		ort_session = new Ort::Session(ort_env, rf_path.c_str(), session_options);
		auto input_node_name = ort_session->GetInputNameAllocated(0, allocator);
		string input_name = input_node_name.get();
		input_names_str.push_back(input_name);
		input_names.push_back(input_names_str.back().c_str());
		auto output_node_name = ort_session->GetOutputNameAllocated(0, allocator);
		string output_name = output_node_name.get();
		output_names_str.push_back(output_name);
		output_names.push_back(output_names_str.back().c_str());

		Ort::TypeInfo inputTypeInfo = ort_session->GetInputTypeInfo(0);
		auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
		input_dims = inputTensorInfo.GetShape();

		Ort::TypeInfo outputTypeInfo = ort_session->GetOutputTypeInfo(0);
		auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
		output_dims = outputTensorInfo.GetShape();
	}

	// Main Loop
	unique_lock<mutex> lock(state_mutex);

	auto start_time = chrono::steady_clock::now();
	auto wakeup_time = start_time + chrono::seconds(operation_window_s);

	// Open message queue
	mqd_t request_queue;
	request_queue = mq_open(request_queue_name.c_str(), O_RDONLY | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH, NULL);
	BUG_ON(request_queue == (mqd_t) -1);
	mqd_t response_queue;
	response_queue = mq_open(response_queue_name.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH, NULL);
	BUG_ON(response_queue == (mqd_t) -1);
	thread mq_thread(mq_thread_fn, request_queue, response_queue, &vm_vec, is_1lm, dram_pct, karma_init_credit_factor, &exchange_log_file, start_time);

	for (; !has_signal; wakeup_time += chrono::seconds(operation_window_s), cout << endl) {
		// Sleep until the next operation window
		auto cur_time = chrono::steady_clock::now();
		if (cur_time < wakeup_time) {
			lock.unlock();
			sleep(chrono::duration_cast<chrono::seconds>(wakeup_time - cur_time).count());
			lock.lock();
		}
		cur_time = chrono::steady_clock::now();
		uint64_t epoch = chrono::duration_cast<chrono::seconds>(cur_time - start_time).count();

		if (state == STATE_PAUSED) {
			printf("Time: %ld seconds\n", epoch);
			printf("Execution paused\n");
			continue;
		}

		// Update DAMON regions
		update_damon_regions(trace_file, vm_vec, damon_log_file, epoch);

		// Read perf counters
		update_perf_counters(vm_vec, perf_log_file, epoch);
		update_metrics(vm_vec, metric_log_file, ewma_constant, epoch, operation_window_s);
		update_karma_avg_metrics(vm_vec, karma_avg_ewma_constant, karma_avg_size);
		if (ort_session != NULL)
			update_rf_prediction(vm_vec, *ort_session, input_names, output_names, input_dims, output_dims);

		// Log VMs
		log_and_print_vms(vm_vec, vms_log_file, epoch);

		// Exchange pages across VMs based on allocation policy
		switch(policy) {
		case POLICY_MIN_MISS:
			min_miss_exchange(vm_vec, step_ratio, policy_log_file, exchange_log_file, epoch, rand_eng);
			break;
		case POLICY_KARMA:
			karma_exchange(vm_vec, step_ratio, karma_unit, policy_log_file, exchange_log_file, epoch, rand_eng);
			break;
		default:
			// Not implemented
			BUG_ON(true);
		}

		// Hot-cold separation
		hot_cold_separation(vm_vec, hot_threshold, cold_threshold, random_hot_cold, hot_cold_exchange_log_file, epoch, rand_eng);
	}
	lock.unlock();

	mq_thread.join();
	BUG_ON(mq_close(request_queue) != 0);
	BUG_ON(mq_close(response_queue) != 0);

	// Terminate all the VMs
	for (int i = 0; i < vm_vec.size(); ++i) {
		struct VM *vm = &vm_vec[i];
		if (vm->pid < 0)
			continue;
		teardown_vm(vm);
	}

	// Teardown
	delete ort_session;
	BUG_ON(fclose(trace_file) != 0);

	hot_cold_exchange_log_file.close();
	damon_log_file.close();
	perf_log_file.close();
	exchange_log_file.close();
	policy_log_file.close();
	metric_log_file.close();
	vms_log_file.close();
	return 0;
}

#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <signal.h>
#include <math.h>
#include <chrono>
#include <thread>
#include <fstream>
#include <numeric>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/page_coloring.h>

// #undef COLOR_THP
#define COLOR_THP

using namespace std;

#define __NR_set_color 480
#define __NR_get_color 481
#define __NR_reserve_color 482

#define MADV_NO_PPOOL  40
#define MADV_PPOOL_0   41
// #define MADV_PPOOL_1   42

#define PHYS_TO_DRAM_PHYS(phys)	((phys) % (DRAM_SIZE_PER_NODE))

#define COLORINFO_PATH "/proc/colorinfo"
#define PAGE_SHIFT 12UL
#define PAGE_SIZE (1UL << PAGE_SHIFT)
#define HUGE_PAGE_SHIFT 21UL
#define HUGE_PAGE_SIZE (1UL << HUGE_PAGE_SHIFT)
#define MAX_PATH_LEN 512UL

#ifdef COLOR_THP
#define COLOR_PAGE_SHIFT HUGE_PAGE_SHIFT
#define COLOR_PAGE_SIZE HUGE_PAGE_SIZE
#else
#define COLOR_PAGE_SHIFT PAGE_SHIFT
#define COLOR_PAGE_SIZE PAGE_SIZE
#endif

#define PPOOL_PATH "/proc/color_ppool"

#define ROUND_DOWN(a, b) ((a) / (b) * (b))
#define ROUND_UP(a, b) (((a) + (b) - 1) / (b) * (b))

#define BUG_ON(cond)											\
	do {												\
		if (cond) {										\
			fprintf(stdout, "BUG_ON: %s (L%d) %s\n", __FILE__, __LINE__, __FUNCTION__);	\
			raise(SIGABRT);									\
		}											\
	} while (0)


int set_color_syscall(int len, void *buffer, int pid) {
	return syscall(__NR_set_color, pid, len, buffer);
}

int reserve_color_syscall(long nr_page) {
	return syscall(__NR_reserve_color, nr_page);
}

void print_help_message() {
	printf("Usage:\n"
	       "  colorctl set_color <color list (e.g., 0-7,8-15)> <pid> <command (optional)>\n"
	       "  colorctl fill_color <number of pages per color>\n"
	       "  colorctl set_ppool <pool> <pid> <command (optional)>\n"
	       "  colorctl unset_ppool <pid>\n"
	       "  colorctl fill_ppool <pool> <target number of pages> <nid> <color list (e.g., 0-7,8-15)>\n"
	       "  colorctl clean_ppool <pool> <current number of pages> <color filter list> <target number of pages> <target page degree>\n"
	       "  colorctl shuffle_ppool <pool> <number of pages> <first number of pages to shuffle (optional)>\n"
	       "  colorctl verify_ppool <pool> <total number of pages> <color filter list> <expected number of pages> <expected page degree>\n"
	       "  colorctl report_ppool <pool> <total number of pages> <color filter list>\n"
	       "  colorctl drain_lru\n"
	       "  colorctl verify_order <pool> <number of pages>\n"
	       "  colorctl organize_ppool <pool> <total number of pages> <target number of pages> <contention degree> <target number of uncontended pages (optional)>\n"
	       "  colorctl verify_organize <pool> <total number of pages> <target number of pages> <contention degree> <target number of uncontended pages (optional)>\n"
	       "  colorctl sort_ppool <pool> <pages per machine> <number of machines>\n"
		   "  colorctl verify_sort <pool> <pages per machine> <number of machines>\n"
	);
}

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

void convert_list_to_bitmap(char *list, uint64_t *bitmap) {
	parse_list_str(list, [&](long raw_color) {
		BUG_ON(raw_color < 0 || raw_color >= NR_COLORS);
		int color = (int) raw_color;
		int buffer_index = color / (sizeof(uint64_t) * 8);
		int bit_index = color % (sizeof(uint64_t) * 8);
		bitmap[buffer_index] = bitmap[buffer_index] | (1l << bit_index);
	});
}

int dram_addr_to_color(uint64_t dram_addr) {
	uint64_t dram_index = dram_addr >> COLOR_PAGE_SHIFT;

	uint64_t tmp_1 = dram_index % NR_COLORS;
	uint64_t tmp_2 = dram_index / NR_COLORS;

	return (int) ((tmp_1 + tmp_2) % NR_COLORS);
}

void set_color(int argc, char *argv[]) {
	if (argc < 2) {
		print_help_message();
		exit(1);
	}
	int pid = atoi(argv[1]);
	int buffer_len = (NR_COLORS + sizeof(uint64_t) * 8 - 1) / (sizeof(uint64_t) * 8);
	uint64_t *buffer = (uint64_t *) malloc(sizeof(*buffer) * buffer_len);
	BUG_ON(buffer == NULL);
	memset(buffer, 0, sizeof(*buffer) * buffer_len);
	convert_list_to_bitmap(argv[0], buffer);
	int ret = set_color_syscall(sizeof(*buffer) * buffer_len, buffer, pid);
	BUG_ON(ret != 0);
	free(buffer);

	if (argc > 2)
		execvp(argv[2], argv + 2);
}

void fill_color(int argc, char *argv[]) {
	if (argc != 1) {
		print_help_message();
		exit(1);
	}
	long num_pages_per_color = atol(argv[0]);
	int ret = reserve_color_syscall(num_pages_per_color);
	BUG_ON(ret != 0);
}

void set_ppool(int argc, char *argv[]) {
	if (argc < 2) {
		print_help_message();
		exit(1);
	}
	int pool = atoi(argv[0]);
	BUG_ON(pool < 0 || pool >= NR_PPOOLS);
	int pid = atoi(argv[1]);
	BUG_ON(pid < 0);

	struct ppool_enable_req req;
	req.pool = pool;
	req.pid = pid;

	int fd = open(PPOOL_PATH, O_RDONLY);
	BUG_ON(fd < 0);
	int ret = ioctl(fd, PPOOL_IOC_ENABLE, &req);
	BUG_ON(ret != 0);
	ret = close(fd);
	BUG_ON(ret != 0);

	if (argc > 2)
		execvp(argv[2], argv + 2);
}

void unset_ppool(int argc, char *argv[]) {
	if (argc != 1) {
		print_help_message();
		exit(1);
	}
	int pid = atoi(argv[0]);
	BUG_ON(pid < 0);
	int fd = open(PPOOL_PATH, O_RDONLY);
	BUG_ON(fd < 0);
	int ret = ioctl(fd, PPOOL_IOC_DISABLE, &pid);
	BUG_ON(ret != 0);
	ret = close(fd);
	BUG_ON(ret != 0);
}

void fill_ppool(int argc, char *argv[]) {
	if (argc != 4) {
		print_help_message();
		exit(1);
	}
	int pool = atoi(argv[0]);
	uint64_t target_num_pages = atol(argv[1]);
	int nid = atoi(argv[2]);
	BUG_ON(nid < 0);
	int buffer_len = (NR_COLORS + sizeof(uint64_t) * 8 - 1) / (sizeof(uint64_t) * 8);
	uint64_t *buffer = (uint64_t *) malloc(sizeof(*buffer) * buffer_len);
	BUG_ON(buffer == NULL);
	memset(buffer, 0, sizeof(*buffer) * buffer_len);
	convert_list_to_bitmap(argv[3], buffer);

	struct ppool_fill_req req;
	req.pool = pool;
	req.target_num_pages = target_num_pages;
	req.nid = nid;
	req.user_mask_ptr = buffer;

	int fd = open(PPOOL_PATH, O_RDONLY);
	BUG_ON(fd < 0);
	int ret = ioctl(fd, PPOOL_IOC_FILL, &req);
	BUG_ON(ret != 0);
	ret = close(fd);
	BUG_ON(ret != 0);
	free(buffer);
}

void clean_ppool(int argc, char *argv[]) {
	if (argc != 5) {
		print_help_message();
		exit(1);
	}
	int pool = atoi(argv[0]);
	BUG_ON(pool < 0 || pool >= NR_PPOOLS);
	uint64_t cur_num_pages = atol(argv[1]);
	char *color_filter_list_str = argv[2];
	uint64_t target_num_pages = atol(argv[3]);
	BUG_ON(target_num_pages > cur_num_pages);
	int target_page_degree = atoi(argv[4]);
	BUG_ON(target_page_degree < 1);
	BUG_ON(target_num_pages % target_page_degree != 0);

	unordered_set<int> color_filter_set;
	parse_list_str(color_filter_list_str, [&](long color) {
		BUG_ON(color < 0 || color >= NR_COLORS);
		color_filter_set.insert((int) color);
	});

	char *buffer = (char *) aligned_alloc(COLOR_PAGE_SIZE, cur_num_pages * COLOR_PAGE_SIZE);
	BUG_ON(buffer == NULL);
	BUG_ON(((uint64_t) buffer) % COLOR_PAGE_SIZE != 0);
	int madvise_ret;
#ifdef COLOR_THP
	madvise_ret = madvise(buffer, cur_num_pages * COLOR_PAGE_SIZE, MADV_HUGEPAGE);
	BUG_ON(madvise_ret != 0);
#endif
	madvise_ret = madvise(buffer, cur_num_pages * COLOR_PAGE_SIZE, MADV_PPOOL_0);
	BUG_ON(madvise_ret != 0);
	for (uint64_t i = 0; i < cur_num_pages; ++i)
		memset(buffer + i * COLOR_PAGE_SIZE, 0, 64);

	int pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
	BUG_ON(pagemap_fd < 0);
	uint64_t *pagemap = (uint64_t *) malloc(sizeof(*pagemap) * cur_num_pages);
	BUG_ON(pagemap == NULL);
	for (uint64_t i = 0; i < cur_num_pages; ++i) {
		int read_ret = pread(pagemap_fd, pagemap + i, sizeof(*pagemap),
		                     (((uint64_t) buffer + i * COLOR_PAGE_SIZE) >> PAGE_SHIFT) * sizeof(*pagemap));
		BUG_ON(read_ret != sizeof(*pagemap));
	}
	close(pagemap_fd);

	unordered_map<uint64_t, vector<uint64_t>> dram_addr_to_phys_addr;
	for (uint64_t i = 0; i < cur_num_pages; ++i) {
		BUG_ON((pagemap[i] & (1UL << 63)) == 0);  // Not present
		BUG_ON((pagemap[i] & (1UL << 62)) != 0);  // Swapped
		uint64_t phys_addr = (pagemap[i] & ((1UL << 55) - 1)) << PAGE_SHIFT;
		BUG_ON(phys_addr == 0);

		uint64_t dram_addr = PHYS_TO_DRAM_PHYS(phys_addr);
		int color = dram_addr_to_color(dram_addr);
		if (color_filter_set.find(color) == color_filter_set.end())
			continue;

		dram_addr_to_phys_addr[dram_addr].push_back(phys_addr);
	}
	free(pagemap);

	vector<uint64_t> phys_addr_to_release_vec;
	uint64_t kept_num_pages = 0;
	for (auto &pair : dram_addr_to_phys_addr) {
		uint64_t dram_addr = pair.first;
		vector<uint64_t> &phys_addr_vec = pair.second;
		if (phys_addr_vec.size() < target_page_degree) {
			for (uint64_t phys_addr : phys_addr_vec)
				phys_addr_to_release_vec.push_back(phys_addr);
		} else {
			uint64_t cur_dram_kept_num_pages = 0;
			for (uint64_t phys_addr : phys_addr_vec) {
				if (kept_num_pages >= target_num_pages || cur_dram_kept_num_pages >= target_page_degree) {
					phys_addr_to_release_vec.push_back(phys_addr);
				} else {
					++kept_num_pages;
					++cur_dram_kept_num_pages;
				}
			}
		}
	}

	free(buffer);

	struct ppool_release_req req;
	req.pool = pool;
	req.num_pages = phys_addr_to_release_vec.size();
	req.phys_addr_arr = (__u64 *) phys_addr_to_release_vec.data();

	int fd = open(PPOOL_PATH, O_RDONLY);
	BUG_ON(fd < 0);
	long ioctl_ret = ioctl(fd, PPOOL_IOC_RELEASE, &req);
	if (ioctl_ret != phys_addr_to_release_vec.size())
		printf("ioctl_ret = %ld, expected = %ld\n", ioctl_ret, phys_addr_to_release_vec.size());
	BUG_ON(ioctl_ret != phys_addr_to_release_vec.size());
	close(fd);
}

void shuffle_ppool(int argc, char *argv[]) {
	if (argc != 2 && argc != 3) {
		print_help_message();
		exit(1);
	}
	int pool = atoi(argv[0]);
	BUG_ON(pool < 0 || pool >= NR_PPOOLS);
	uint64_t num_pages = atol(argv[1]);
	if (num_pages == 0)
		return;
	uint64_t num_pages_to_shuffle = num_pages;
	if (argc == 3)
		num_pages_to_shuffle = atol(argv[2]);
	BUG_ON(num_pages_to_shuffle > num_pages);

	char *buffer = (char *) aligned_alloc(COLOR_PAGE_SIZE, num_pages * COLOR_PAGE_SIZE);
	BUG_ON(buffer == NULL);
	BUG_ON(((uint64_t) buffer) % COLOR_PAGE_SIZE != 0);
	int madvise_ret;
#ifdef COLOR_THP
	madvise_ret = madvise(buffer, num_pages * COLOR_PAGE_SIZE, MADV_HUGEPAGE);
	BUG_ON(madvise_ret != 0);
#endif
	madvise_ret = madvise(buffer, num_pages * COLOR_PAGE_SIZE, MADV_PPOOL_0);
	BUG_ON(madvise_ret != 0);

	uint64_t *index_arr = (uint64_t *) malloc(sizeof(*index_arr) * num_pages);
	BUG_ON(index_arr == NULL);
	iota(index_arr, index_arr + num_pages, 0);

	default_random_engine rand_eng(chrono::system_clock::now().time_since_epoch().count());
	shuffle(index_arr, index_arr + num_pages_to_shuffle, rand_eng);

	for (uint64_t i = 0; i < num_pages; ++i) {
		uint64_t index = index_arr[i];
		memset(buffer + index * COLOR_PAGE_SIZE, 0, 64);
	}
	free(index_arr);
	free(buffer);
}

void verify_ppool(int argc, char *argv[]) {
	if (argc != 5) {
		print_help_message();
		exit(1);
	}
	int pool = atoi(argv[0]);
	BUG_ON(pool < 0 || pool >= NR_PPOOLS);
	uint64_t total_num_pages = atol(argv[1]);
	char *color_filter_list_str = argv[2];
	uint64_t num_pages = atol(argv[3]);
	int page_degree = atoi(argv[4]);
	BUG_ON(num_pages > total_num_pages);
	BUG_ON(page_degree < 1);
	BUG_ON(num_pages % page_degree != 0);

	unordered_set<int> color_filter_set;
	parse_list_str(color_filter_list_str, [&](long color) {
		BUG_ON(color < 0 || color >= NR_COLORS);
		color_filter_set.insert((int) color);
	});

	char *buffer = (char *) aligned_alloc(COLOR_PAGE_SIZE, total_num_pages * COLOR_PAGE_SIZE);
	BUG_ON(buffer == NULL);
	BUG_ON(((uint64_t) buffer) % COLOR_PAGE_SIZE != 0);
	int madvise_ret;
#ifdef COLOR_THP
	madvise_ret = madvise(buffer, total_num_pages * COLOR_PAGE_SIZE, MADV_HUGEPAGE);
	BUG_ON(madvise_ret != 0);
#endif
	madvise_ret = madvise(buffer, total_num_pages * COLOR_PAGE_SIZE, MADV_PPOOL_0);
	BUG_ON(madvise_ret != 0);
	for (uint64_t i = 0; i < total_num_pages; ++i)
		memset(buffer + i * COLOR_PAGE_SIZE, 0, 64);

	int pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
	BUG_ON(pagemap_fd < 0);
	uint64_t *pagemap = (uint64_t *) malloc(sizeof(*pagemap) * total_num_pages);
	BUG_ON(pagemap == NULL);
	for (uint64_t i = 0; i < total_num_pages; ++i) {
		int read_ret = pread(pagemap_fd, pagemap + i, sizeof(*pagemap),
		                     (((uint64_t) buffer + i * COLOR_PAGE_SIZE) >> PAGE_SHIFT) * sizeof(*pagemap));
		BUG_ON(read_ret != sizeof(*pagemap));
	}
	close(pagemap_fd);

	unordered_map<uint64_t, vector<uint64_t>> dram_addr_to_phys_addr;
	for (uint64_t i = 0; i < total_num_pages; ++i) {
		BUG_ON((pagemap[i] & (1UL << 63)) == 0);  // Not present
		BUG_ON((pagemap[i] & (1UL << 62)) != 0);  // Swapped
		uint64_t phys_addr = (pagemap[i] & ((1UL << 55) - 1)) << PAGE_SHIFT;
		BUG_ON(phys_addr == 0);

		uint64_t dram_addr = PHYS_TO_DRAM_PHYS(phys_addr);
		int color = dram_addr_to_color(dram_addr);
		if (color_filter_set.find(color) == color_filter_set.end())
			continue;

		dram_addr_to_phys_addr[dram_addr].push_back(phys_addr);
	}
	free(pagemap);

	for (auto &pair : dram_addr_to_phys_addr) {
		vector<uint64_t> &phys_addr_vec = pair.second;
		if (phys_addr_vec.size() != page_degree) {
			printf("Error: %lu pages in the same DRAM page, expected %d\n", phys_addr_vec.size(),
			       page_degree);
			exit(1);
		}
	}
	if (dram_addr_to_phys_addr.size() != num_pages / page_degree) {
		printf("Error: %lu DRAM pages, expected %lu\n", dram_addr_to_phys_addr.size(),
		       num_pages / page_degree);
		exit(1);
	}

	free(buffer);
}

void report_ppool(int argc, char *argv[]) {
	if (argc != 3) {
		print_help_message();
		exit(1);
	}
	int pool = atoi(argv[0]);
	BUG_ON(pool < 0 || pool >= NR_PPOOLS);
	uint64_t total_num_pages = atol(argv[1]);
	char *color_filter_list_str = argv[2];

	unordered_set<int> color_filter_set;
	parse_list_str(color_filter_list_str, [&](long color) {
		BUG_ON(color < 0 || color >= NR_COLORS);
		color_filter_set.insert((int) color);
	});

	char *buffer = (char *) aligned_alloc(COLOR_PAGE_SIZE, total_num_pages * COLOR_PAGE_SIZE);
	BUG_ON(buffer == NULL);
	BUG_ON(((uint64_t) buffer) % COLOR_PAGE_SIZE != 0);
	int madvise_ret;
#ifdef COLOR_THP
	madvise_ret = madvise(buffer, total_num_pages * COLOR_PAGE_SIZE, MADV_HUGEPAGE);
	BUG_ON(madvise_ret != 0);
#endif
	madvise_ret = madvise(buffer, total_num_pages * COLOR_PAGE_SIZE, MADV_PPOOL_0);
	BUG_ON(madvise_ret != 0);
	for (uint64_t i = 0; i < total_num_pages; ++i)
		memset(buffer + i * COLOR_PAGE_SIZE, 0, 64);

	int pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
	BUG_ON(pagemap_fd < 0);
	uint64_t *pagemap = (uint64_t *) malloc(sizeof(*pagemap) * total_num_pages);
	BUG_ON(pagemap == NULL);
	for (uint64_t i = 0; i < total_num_pages; ++i) {
		int read_ret = pread(pagemap_fd, pagemap + i, sizeof(*pagemap),
		                     (((uint64_t) buffer + i * COLOR_PAGE_SIZE) >> PAGE_SHIFT) * sizeof(*pagemap));
		BUG_ON(read_ret != sizeof(*pagemap));
	}
	close(pagemap_fd);

	unordered_map<uint64_t, vector<uint64_t>> dram_addr_to_phys_addr;
	for (uint64_t i = 0; i < total_num_pages; ++i) {
		BUG_ON((pagemap[i] & (1UL << 63)) == 0);  // Not present
		BUG_ON((pagemap[i] & (1UL << 62)) != 0);  // Swapped
		uint64_t phys_addr = (pagemap[i] & ((1UL << 55) - 1)) << PAGE_SHIFT;
		BUG_ON(phys_addr == 0);

		uint64_t dram_addr = PHYS_TO_DRAM_PHYS(phys_addr);
		int color = dram_addr_to_color(dram_addr);
		if (color_filter_set.find(color) == color_filter_set.end())
			continue;

		dram_addr_to_phys_addr[dram_addr].push_back(phys_addr);
	}
	free(pagemap);

	unordered_map<uint64_t, uint64_t> page_count_map;
	for (auto &pair : dram_addr_to_phys_addr) {
		vector<uint64_t> &phys_addr_vec = pair.second;
		page_count_map[phys_addr_vec.size()] += phys_addr_vec.size();
	}
	for (auto &pair : page_count_map) {
		printf("%lu-degree pages: %lu\n", pair.first, pair.second);
	}

	free(buffer);
}

void verify_order(int argc, char *argv[]) {
	if (argc != 2) {
		print_help_message();
		exit(1);
	}
	int pool = atoi(argv[0]);
	BUG_ON(pool < 0 || pool >= NR_PPOOLS);
	uint64_t num_pages = atol(argv[1]);
	if (num_pages == 0)
		return;

	// First allocation
	char *buffer = (char *) aligned_alloc(COLOR_PAGE_SIZE, num_pages * COLOR_PAGE_SIZE);
	BUG_ON(buffer == NULL);
	BUG_ON(((uint64_t) buffer) % COLOR_PAGE_SIZE != 0);
	int madvise_ret;
#ifdef COLOR_THP
	madvise_ret = madvise(buffer, num_pages * COLOR_PAGE_SIZE, MADV_HUGEPAGE);
	BUG_ON(madvise_ret != 0);
#endif
	madvise_ret = madvise(buffer, num_pages * COLOR_PAGE_SIZE, MADV_PPOOL_0);
	BUG_ON(madvise_ret != 0);

	uint64_t *index_arr = (uint64_t *) malloc(sizeof(*index_arr) * num_pages);
	BUG_ON(index_arr == NULL);
	iota(index_arr, index_arr + num_pages, 0);
	default_random_engine rand_eng(chrono::system_clock::now().time_since_epoch().count());
	shuffle(index_arr, index_arr + num_pages, rand_eng);

	for (uint64_t i = 0; i < num_pages; ++i) {
		uint64_t index = index_arr[i];
		memset(buffer + index * COLOR_PAGE_SIZE, 0, 64);
	}

	int pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
	BUG_ON(pagemap_fd < 0);
	uint64_t *pagemap = (uint64_t *) malloc(sizeof(*pagemap) * num_pages);
	BUG_ON(pagemap == NULL);
	for (uint64_t i = 0; i < num_pages; ++i) {
		int read_ret = pread(pagemap_fd, pagemap + i, sizeof(*pagemap),
		                     (((uint64_t) buffer + i * COLOR_PAGE_SIZE) >> PAGE_SHIFT) * sizeof(*pagemap));
		BUG_ON(read_ret != sizeof(*pagemap));
	}
	close(pagemap_fd);
	free(index_arr);
	free(buffer);

	// Second allocation
	buffer = (char *) aligned_alloc(COLOR_PAGE_SIZE, num_pages * COLOR_PAGE_SIZE);
	BUG_ON(buffer == NULL);
	BUG_ON(((uint64_t) buffer) % COLOR_PAGE_SIZE != 0);
#ifdef COLOR_THP
	madvise_ret = madvise(buffer, num_pages * COLOR_PAGE_SIZE, MADV_HUGEPAGE);
	BUG_ON(madvise_ret != 0);
#endif
	madvise_ret = madvise(buffer, num_pages * COLOR_PAGE_SIZE, MADV_PPOOL_0);
	BUG_ON(madvise_ret != 0);
	for (uint64_t i = 0; i < num_pages; ++i) {
		memset(buffer + i * COLOR_PAGE_SIZE, 0, 64);
	}

	pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
	BUG_ON(pagemap_fd < 0);
	for (uint64_t i = 0; i < num_pages; ++i) {
		uint64_t second_pagemap;
		int read_ret = pread(pagemap_fd, &second_pagemap, sizeof(second_pagemap),
		                     (((uint64_t) buffer + i * COLOR_PAGE_SIZE) >> PAGE_SHIFT) * sizeof(second_pagemap));
		BUG_ON(read_ret != sizeof(second_pagemap));
		BUG_ON(pagemap[i] != second_pagemap);
	}
	close(pagemap_fd);
	free(buffer);
	free(pagemap);
}

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

void verify_organize(int argc, char *argv[]) {
	if (argc != 4 && argc != 5) {
		print_help_message();
		exit(1);
	}
	int pool = atoi(argv[0]);
	BUG_ON(pool < 0 || pool >= NR_PPOOLS);
	uint64_t total_num_pages = atol(argv[1]);
	uint64_t target_num_pages = atol(argv[2]);
	BUG_ON(target_num_pages > total_num_pages);
	uint64_t contention_degree = atol(argv[3]);
	BUG_ON(contention_degree <= 1);
	BUG_ON(total_num_pages % contention_degree != 0);
	BUG_ON(target_num_pages % contention_degree != 0);
	uint64_t target_num_uncontended_pages;
	if (argc == 5) {
		target_num_uncontended_pages = atol(argv[4]);
		BUG_ON(target_num_uncontended_pages > target_num_pages);
		BUG_ON(target_num_uncontended_pages % contention_degree != 0);
	}

	// Allocate all the pages in the pool
	char *buffer = (char *) aligned_alloc(COLOR_PAGE_SIZE, total_num_pages * COLOR_PAGE_SIZE);
	BUG_ON(buffer == NULL);
	BUG_ON(((uint64_t) buffer) % COLOR_PAGE_SIZE != 0);
	int madvise_ret;
#ifdef COLOR_THP
	madvise_ret = madvise(buffer, total_num_pages * COLOR_PAGE_SIZE, MADV_HUGEPAGE);
	BUG_ON(madvise_ret != 0);
#endif
	madvise_ret = madvise(buffer, total_num_pages * COLOR_PAGE_SIZE, MADV_PPOOL_0);
	BUG_ON(madvise_ret != 0);
	for (uint64_t i = 0; i < total_num_pages; ++i)
		memset(buffer + i * COLOR_PAGE_SIZE, 0, 64);

	// Retrieve the physcial addresses of all the pages
	int pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
	BUG_ON(pagemap_fd < 0);
	uint64_t *pagemap = (uint64_t *) malloc(sizeof(*pagemap) * total_num_pages);
	BUG_ON(pagemap == NULL);
	for (uint64_t i = 0; i < total_num_pages; ++i) {
		int read_ret = pread(pagemap_fd, pagemap + i, sizeof(*pagemap),
		                     (((uint64_t) buffer + i * COLOR_PAGE_SIZE) >> PAGE_SHIFT) * sizeof(*pagemap));
		BUG_ON(read_ret != sizeof(*pagemap));
	}
	close(pagemap_fd);

	// Identify contended pages and uncontended pages
	unordered_map<uint64_t, vector<uint64_t>> dram_addr_to_index;
	for (uint64_t i = 0; i < total_num_pages; ++i) {
		BUG_ON((pagemap[i] & (1UL << 63)) == 0);  // Not present
		BUG_ON((pagemap[i] & (1UL << 62)) != 0);  // Swapped
		uint64_t phys_addr = (pagemap[i] & ((1UL << 55) - 1)) << PAGE_SHIFT;
		BUG_ON(phys_addr == 0);

		uint64_t dram_addr = PHYS_TO_DRAM_PHYS(phys_addr);
		dram_addr_to_index[dram_addr].push_back(i);
	}

	// Verify
	uint64_t num_uncontended_pages = 0;
	for (uint64_t i = 0; i < target_num_pages; ++i) {
		uint64_t phys_addr = (pagemap[i] & ((1UL << 55) - 1)) << PAGE_SHIFT;
		uint64_t dram_addr = PHYS_TO_DRAM_PHYS(phys_addr);
		BUG_ON(dram_addr_to_index[dram_addr].size() != contention_degree && dram_addr_to_index[dram_addr].size() != 1);
		for (uint64_t j : dram_addr_to_index[dram_addr]) {
			BUG_ON(j >= target_num_pages);
		}
		if (dram_addr_to_index[dram_addr].size() == 1)
			++num_uncontended_pages;
	}
	if (argc == 5)
		BUG_ON(num_uncontended_pages != target_num_uncontended_pages);
	free(pagemap);
	free(buffer);
}

void organize_ppool(int argc, char *argv[], char **buffer_p) {
	if (argc != 4 && argc != 5) {
		print_help_message();
		exit(1);
	}
	int pool = atoi(argv[0]);
	BUG_ON(pool < 0 || pool >= NR_PPOOLS);
	uint64_t total_num_pages = atol(argv[1]);
	uint64_t target_num_pages = atol(argv[2]);
	BUG_ON(target_num_pages > total_num_pages);
	uint64_t contention_degree = atol(argv[3]);
	BUG_ON(contention_degree <= 1);
	BUG_ON(total_num_pages % contention_degree != 0);
	BUG_ON(target_num_pages % contention_degree != 0);

	// Only support 2-degree pages for now
	BUG_ON(contention_degree != 2);

	// Allocate all the pages in the pool
	char *buffer = (char *) aligned_alloc(COLOR_PAGE_SIZE, total_num_pages * COLOR_PAGE_SIZE);
	BUG_ON(buffer == NULL);
	BUG_ON(((uint64_t) buffer) % COLOR_PAGE_SIZE != 0);
	int madvise_ret;
#ifdef COLOR_THP
	madvise_ret = madvise(buffer, total_num_pages * COLOR_PAGE_SIZE, MADV_HUGEPAGE);
	BUG_ON(madvise_ret != 0);
#endif
	madvise_ret = madvise(buffer, total_num_pages * COLOR_PAGE_SIZE, MADV_PPOOL_0);
	BUG_ON(madvise_ret != 0);
	for (uint64_t i = 0; i < total_num_pages; ++i)
		memset(buffer + i * COLOR_PAGE_SIZE, 0, 64);
	if (argc == 5)
		*buffer_p = buffer;

	// Retrieve the physcial addresses of all the pages
	int pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
	BUG_ON(pagemap_fd < 0);
	uint64_t *pagemap = (uint64_t *) malloc(sizeof(*pagemap) * total_num_pages);
	BUG_ON(pagemap == NULL);
	for (uint64_t i = 0; i < total_num_pages; ++i) {
		int read_ret = pread(pagemap_fd, pagemap + i, sizeof(*pagemap),
		                     (((uint64_t) buffer + i * COLOR_PAGE_SIZE) >> PAGE_SHIFT) * sizeof(*pagemap));
		BUG_ON(read_ret != sizeof(*pagemap));
	}
	close(pagemap_fd);

	// Identify contended pages and uncontended pages
	unordered_map<uint64_t, vector<uint64_t>> dram_addr_to_index;
	for (uint64_t i = 0; i < total_num_pages; ++i) {
		BUG_ON((pagemap[i] & (1UL << 63)) == 0);  // Not present
		BUG_ON((pagemap[i] & (1UL << 62)) != 0);  // Swapped
		uint64_t phys_addr = (pagemap[i] & ((1UL << 55) - 1)) << PAGE_SHIFT;
		BUG_ON(phys_addr == 0);

		uint64_t dram_addr = PHYS_TO_DRAM_PHYS(phys_addr);
		dram_addr_to_index[dram_addr].push_back(i);
	}

	unordered_set<uint64_t> target_contended_index_set;
	for (uint64_t i = 0; i < target_num_pages; ++i) {
		uint64_t phys_addr = (pagemap[i] & ((1UL << 55) - 1)) << PAGE_SHIFT;
		uint64_t dram_addr = PHYS_TO_DRAM_PHYS(phys_addr);
		uint64_t page_degree = dram_addr_to_index[dram_addr].size();
		if (page_degree == contention_degree) {
			bool has_non_target = false;
			for (uint64_t j : dram_addr_to_index[dram_addr]) {
				if (j >= target_num_pages) {
					has_non_target = true;
					break;
				}
			}
			if (has_non_target)
				target_contended_index_set.insert(i);
		} else {
			BUG_ON(page_degree != 1);
		}
	}

	vector<uint64_t> non_target_uncontended_index_vec;
	for (uint64_t i = target_num_pages; i < total_num_pages; ++i) {
		if (non_target_uncontended_index_vec.size() >= contention_degree - 1)
			break;
		uint64_t phys_addr = (pagemap[i] & ((1UL << 55) - 1)) << PAGE_SHIFT;
		uint64_t dram_addr = PHYS_TO_DRAM_PHYS(phys_addr);
		uint64_t page_degree = dram_addr_to_index[dram_addr].size();
		if (page_degree == 1) {
			non_target_uncontended_index_vec.push_back(i);
		} else {
			BUG_ON(page_degree != contention_degree);
		}
	}

	vector<uint64_t> exchange_virt_addr_vec_1;
	vector<uint64_t> exchange_virt_addr_vec_2;
	while (!target_contended_index_set.empty()) {
		uint64_t i = *(target_contended_index_set.begin());
		uint64_t phys_addr = (pagemap[i] & ((1UL << 55) - 1)) << PAGE_SHIFT;
		uint64_t dram_addr = PHYS_TO_DRAM_PHYS(phys_addr);

		uint64_t count = 0;
		for (uint64_t j : dram_addr_to_index[dram_addr]) {
			if (j < target_num_pages) {
				BUG_ON(target_contended_index_set.find(j) == target_contended_index_set.end());
				++count;
			}
		}
		BUG_ON(count == 0);
		BUG_ON(count >= contention_degree);
		if (target_contended_index_set.size() - count >= contention_degree - count) {
			vector<uint64_t> index_to_exchange_vec;
			for (uint64_t j : dram_addr_to_index[dram_addr]) {
				if (j < target_num_pages)
					target_contended_index_set.erase(j);
				else
					index_to_exchange_vec.push_back(j);
			}
			for (uint64_t j : index_to_exchange_vec) {
				uint64_t k = *(target_contended_index_set.begin());
				target_contended_index_set.erase(k);

				exchange_virt_addr_vec_1.push_back(((uint64_t) buffer) + k * COLOR_PAGE_SIZE);
				exchange_virt_addr_vec_2.push_back(((uint64_t) buffer) + j * COLOR_PAGE_SIZE);

				uint64_t phys_addr_k = (pagemap[k] & ((1UL << 55) - 1)) << PAGE_SHIFT;
				uint64_t dram_addr_k = PHYS_TO_DRAM_PHYS(phys_addr_k);
				// No need to update dram_addr_to_index and pagemap because they won't be used
				for (uint64_t l : dram_addr_to_index[dram_addr_k]) {
					// Only true if contention degree is 2
					BUG_ON(target_contended_index_set.find(l) != target_contended_index_set.end());
				}
			}
		} else {
			break;
		}
	}
	if (!target_contended_index_set.empty()) {
		BUG_ON(target_contended_index_set.size() >= contention_degree);
		BUG_ON(target_contended_index_set.size() > non_target_uncontended_index_vec.size());
		uint64_t jj = 0;
		for (uint64_t i : target_contended_index_set) {
			uint64_t j = non_target_uncontended_index_vec[jj++];
			exchange_virt_addr_vec_1.push_back(((uint64_t) buffer) + i * COLOR_PAGE_SIZE);
			exchange_virt_addr_vec_2.push_back(((uint64_t) buffer) + j * COLOR_PAGE_SIZE);
		}
	}

	void **skipped_page_arr_1 = (void **) malloc(sizeof(*skipped_page_arr_1) * exchange_virt_addr_vec_1.size());
	BUG_ON(skipped_page_arr_1 == NULL);
	void **skipped_page_arr_2 = (void **) malloc(sizeof(*skipped_page_arr_2) * exchange_virt_addr_vec_2.size());
	BUG_ON(skipped_page_arr_2 == NULL);
	struct page_xchg_stats stats;
	// Might consider implementing a page exchange without data movement
	exchange_pages(0, 0, exchange_virt_addr_vec_1.size(), (void **) exchange_virt_addr_vec_1.data(),
	               (void **) exchange_virt_addr_vec_2.data(), skipped_page_arr_1, skipped_page_arr_2, &stats);
	BUG_ON(stats.num_succeeded != exchange_virt_addr_vec_1.size() << (COLOR_PAGE_SHIFT - PAGE_SHIFT));
	free(skipped_page_arr_1);
	free(skipped_page_arr_2);
	free(pagemap);
	if (argc != 5)
		free(buffer);
}

void organize_for_uncontended_pages(int argc, char *argv[], char *buffer) {
	BUG_ON(argc != 5);
	int pool = atoi(argv[0]);
	BUG_ON(pool < 0 || pool >= NR_PPOOLS);
	uint64_t total_num_pages = atol(argv[1]);
	uint64_t target_num_pages = atol(argv[2]);
	BUG_ON(target_num_pages > total_num_pages);
	uint64_t contention_degree = atol(argv[3]);
	BUG_ON(contention_degree <= 1);
	BUG_ON(total_num_pages % contention_degree != 0);
	BUG_ON(target_num_pages % contention_degree != 0);
	uint64_t target_num_uncontended_pages = atol(argv[4]);
	BUG_ON(target_num_uncontended_pages > target_num_pages);
	BUG_ON(target_num_uncontended_pages % contention_degree != 0);

// 	// Allocate all the pages in the pool
// 	char *buffer = (char *) aligned_alloc(COLOR_PAGE_SIZE, total_num_pages * COLOR_PAGE_SIZE);
// 	BUG_ON(buffer == NULL);
// 	BUG_ON(((uint64_t) buffer) % COLOR_PAGE_SIZE != 0);
// 	int madvise_ret;
// #ifdef COLOR_THP
// 	madvise_ret = madvise(buffer, total_num_pages * COLOR_PAGE_SIZE, MADV_HUGEPAGE);
// 	BUG_ON(madvise_ret != 0);
// #endif
// 	madvise_ret = madvise(buffer, total_num_pages * COLOR_PAGE_SIZE, MADV_PPOOL_0);
// 	BUG_ON(madvise_ret != 0);
// 	memset(buffer, 0, total_num_pages * COLOR_PAGE_SIZE);

	// Retrieve the physcial addresses of all the pages
	int pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
	BUG_ON(pagemap_fd < 0);
	uint64_t *pagemap = (uint64_t *) malloc(sizeof(*pagemap) * total_num_pages);
	BUG_ON(pagemap == NULL);
	for (uint64_t i = 0; i < total_num_pages; ++i) {
		int read_ret = pread(pagemap_fd, pagemap + i, sizeof(*pagemap),
		                     (((uint64_t) buffer + i * COLOR_PAGE_SIZE) >> PAGE_SHIFT) * sizeof(*pagemap));
		BUG_ON(read_ret != sizeof(*pagemap));
	}
	close(pagemap_fd);

	// Identify contended pages and uncontended pages
	unordered_map<uint64_t, vector<uint64_t>> dram_addr_to_index;
	for (uint64_t i = 0; i < total_num_pages; ++i) {
		BUG_ON((pagemap[i] & (1UL << 63)) == 0);  // Not present
		BUG_ON((pagemap[i] & (1UL << 62)) != 0);  // Swapped
		uint64_t phys_addr = (pagemap[i] & ((1UL << 55) - 1)) << PAGE_SHIFT;
		BUG_ON(phys_addr == 0);

		uint64_t dram_addr = PHYS_TO_DRAM_PHYS(phys_addr);
		dram_addr_to_index[dram_addr].push_back(i);
	}

	unordered_set<uint64_t> target_contended_dram_set;
	unordered_set<uint64_t> target_uncontended_dram_set;
	for (uint64_t i = 0; i < target_num_pages; ++i) {
		uint64_t phys_addr = (pagemap[i] & ((1UL << 55) - 1)) << PAGE_SHIFT;
		uint64_t dram_addr = PHYS_TO_DRAM_PHYS(phys_addr);
		uint64_t page_degree = dram_addr_to_index[dram_addr].size();
		if (page_degree == contention_degree) {
			target_contended_dram_set.insert(dram_addr);
			for (uint64_t j : dram_addr_to_index[dram_addr]) {
				BUG_ON(j >= target_num_pages);
			}
		} else {
			BUG_ON(page_degree != 1);
			target_uncontended_dram_set.insert(dram_addr);
		}
	}

	unordered_set<uint64_t> non_target_contended_dram_set;
	unordered_set<uint64_t> non_target_uncontended_dram_set;
	for (uint64_t i = target_num_pages; i < total_num_pages; ++i) {
		uint64_t phys_addr = (pagemap[i] & ((1UL << 55) - 1)) << PAGE_SHIFT;
		uint64_t dram_addr = PHYS_TO_DRAM_PHYS(phys_addr);
		uint64_t page_degree = dram_addr_to_index[dram_addr].size();
		if (page_degree == contention_degree) {
			non_target_contended_dram_set.insert(dram_addr);
		} else {
			BUG_ON(page_degree != 1);
			non_target_uncontended_dram_set.insert(dram_addr);
		}
	}

	vector<uint64_t> exchange_virt_addr_vec_1;
	vector<uint64_t> exchange_virt_addr_vec_2;
	if (target_num_uncontended_pages > target_uncontended_dram_set.size()) {
		uint64_t diff = target_num_uncontended_pages - target_uncontended_dram_set.size();
		BUG_ON(diff > non_target_uncontended_dram_set.size());
		BUG_ON(diff % contention_degree != 0);
		for (uint64_t i = 0; i < diff; ++i) {
			uint64_t dram_addr = *(non_target_uncontended_dram_set.begin());
			non_target_uncontended_dram_set.erase(dram_addr);
			uint64_t j = dram_addr_to_index[dram_addr][0];
			exchange_virt_addr_vec_1.push_back(((uint64_t) buffer) + j * COLOR_PAGE_SIZE);
		}
		for (uint64_t i = 0; i < diff / contention_degree; ++i) {
			uint64_t dram_addr = *(target_contended_dram_set.begin());
			target_contended_dram_set.erase(dram_addr);
			for (uint64_t j : dram_addr_to_index[dram_addr]) {
				exchange_virt_addr_vec_2.push_back(((uint64_t) buffer) + j * COLOR_PAGE_SIZE);
			}
		}
	} else if (target_num_uncontended_pages < target_uncontended_dram_set.size()) {
		uint64_t diff = target_uncontended_dram_set.size() - target_num_uncontended_pages;
		BUG_ON(diff > non_target_contended_dram_set.size());
		BUG_ON(diff % contention_degree != 0);
		for (uint64_t i = 0; i < diff; ++i) {
			uint64_t dram_addr = *(target_uncontended_dram_set.begin());
			target_uncontended_dram_set.erase(dram_addr);
			uint64_t j = dram_addr_to_index[dram_addr][0];
			exchange_virt_addr_vec_1.push_back(((uint64_t) buffer) + j * COLOR_PAGE_SIZE);
		}
		for (uint64_t i = 0; i < diff / contention_degree; ++i) {
			uint64_t dram_addr = *(non_target_contended_dram_set.begin());
			non_target_contended_dram_set.erase(dram_addr);
			for (uint64_t j : dram_addr_to_index[dram_addr]) {
				exchange_virt_addr_vec_2.push_back(((uint64_t) buffer) + j * COLOR_PAGE_SIZE);
			}
		}
	} else {
		free(pagemap);
		free(buffer);
		return;
	}
	BUG_ON(exchange_virt_addr_vec_1.size() != exchange_virt_addr_vec_2.size());

	void **skipped_page_arr_1 = (void **) malloc(sizeof(*skipped_page_arr_1) * exchange_virt_addr_vec_1.size());
	BUG_ON(skipped_page_arr_1 == NULL);
	void **skipped_page_arr_2 = (void **) malloc(sizeof(*skipped_page_arr_2) * exchange_virt_addr_vec_2.size());
	BUG_ON(skipped_page_arr_2 == NULL);
	struct page_xchg_stats stats;
	// Might consider implementing a page exchange without data movement
	exchange_pages(0, 0, exchange_virt_addr_vec_1.size(), (void **) exchange_virt_addr_vec_1.data(),
	               (void **) exchange_virt_addr_vec_2.data(), skipped_page_arr_1, skipped_page_arr_2, &stats);
	BUG_ON(stats.num_succeeded != exchange_virt_addr_vec_1.size() << (COLOR_PAGE_SHIFT - PAGE_SHIFT));
	free(skipped_page_arr_1);
	free(skipped_page_arr_2);
	free(pagemap);
	free(buffer);
}

void drain_lru(int argc, char *argv[]) {
	if (argc != 0) {
		print_help_message();
		exit(1);
	}
	int fd = open(COLORINFO_PATH, O_RDONLY);
	BUG_ON(fd < 0);
	long ioctl_ret = ioctl(fd, COLOR_IOC_PREP);
	BUG_ON(ioctl_ret != 0);
	close(fd);
}

/********* START TELEMETRY SECTION **********/

/*
 * Our goal is to organize a ppool such that it can be divided up among M
 * machines, each of which need N pages. The n'th page allocated to one machine
 * should conflict with the n'th page allocated to every other machine.
 *
 * @params argc[0]  pool ID
 *         argc[1]  number of pages allocated to each machine, i.e. N
 *         argc[2]  number of machines, i.e. M
 *
 * ----- START DESIGN NOTES -----
 *
 * APPROACH
 *
 * We want the pages at {i, i + N, i + 2N, ...} to conflict for all i. For each
 * i, choose some arbitrary DRAM address and swap each of the pages at
 * {i, i + N, i + 2N, ...} with a page currently mapped to that DRAM address.
 *
 * This yields at most O(N) exchange_page syscalls.
 *
 * OPTIMIZATIONS
 *
 * There are two optimizations we could implement.
 * 
 * (1) The approach above arbitrarily chooses the DRAM address assigned to the
 *     pages at {i, i + N, i + 2N, ...}. If some of these pages are already
 *     mapped to the same DRAM address, we could use that DRAM address instead.
 *     This would bring the total number of swaps down by some constant factor.
 *
 * (2) The approach above does one batch of swaps for each i. We could greedily
 *     select as many i as possible such that all swaps are non-overlapping.
 *     This would probably bring the total number of syscalls down to O(log N).
 *
 * Since the overhead of __telemetry_sort_ppool is only incurred once, we leave
 * these optimizations as future work.
 *
 * ----- END DESIGN NOTES -----
 */

namespace telemetry {

/*
 * Class to simplify passing of output parameters. This class should be passed
 * by REFERENCE. It is the user's responsibility to handle any memory associated
 * with pagemap and buffer.
 *
 *      dram_addr_to_idx    Contains a mapping from DRAM addresses to a list
 *                          of ppool pages mapped to each DRAM address. A ppool
 *                          page is represented using the index of the 
 *                          corresponding virtual page within <buffer>.
 *
 *      pagemap             Contains the contents of /proc/self/pagemap, i.e.
 *                          the PFNs corresponding to each virtual page.
 *
 *      buffer              Contains all pages currently in the ppool.
 */
class TelemetryParams {
  public:
    unordered_map<uint64_t, unordered_set<uint64_t>> dram_addr_to_idx;
    uint64_t *pagemap;
    char *buffer;
    uint64_t total_num_pages;
    uint64_t pages_per_machine;
    uint64_t num_machines;
};

/*
 * Helper function for inspecting the currrent ppool.
 */
void __get_current_ordering(TelemetryParams& params) {
    // Allocate all pages in the ppool to the buffer.
    int madvise_ret;
#ifdef COLOR_THP
    madvise_ret = madvise(params.buffer, params.total_num_pages * COLOR_PAGE_SIZE, MADV_HUGEPAGE);
    BUG_ON(madvise_ret != 0);
#endif
    madvise_ret = madvise(params.buffer, params.total_num_pages * COLOR_PAGE_SIZE, MADV_PPOOL_0);
    BUG_ON(madvise_ret != 0);
    memset(params.buffer, 0, params.total_num_pages * COLOR_PAGE_SIZE);

    // Retrieve the physical addresses of all the ppool pages.
    int pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
    BUG_ON(pagemap_fd < 0);
    for (uint64_t i = 0; i < params.total_num_pages; ++i) {
        int read_ret = pread(pagemap_fd, params.pagemap + i, sizeof(*params.pagemap),
                             (((uint64_t) params.buffer + i * COLOR_PAGE_SIZE) >> PAGE_SHIFT) * sizeof(*params.pagemap));
        BUG_ON(read_ret != sizeof(*params.pagemap));
    }
    close(pagemap_fd);

    // Identify which ppool pages conflict.
    for (uint64_t i = 0; i < params.total_num_pages; ++i) {
        BUG_ON((params.pagemap[i] & (1UL << 63)) == 0);  // Not present
        BUG_ON((params.pagemap[i] & (1UL << 62)) != 0);  // Swapped
        uint64_t phys_addr = (params.pagemap[i] & ((1UL << 55) - 1)) << PAGE_SHIFT;
        BUG_ON(phys_addr == 0);

        uint64_t dram_addr = PHYS_TO_DRAM_PHYS(phys_addr);
        if (params.dram_addr_to_idx.find(dram_addr) != params.dram_addr_to_idx.end()) {
            BUG_ON(params.dram_addr_to_idx[dram_addr].find(i) != params.dram_addr_to_idx[dram_addr].end());
            params.dram_addr_to_idx[dram_addr].emplace(i);
        } else {
            params.dram_addr_to_idx[dram_addr] = unordered_set<uint64_t>{i};
        }
    }
}

/*
 * Sorts the ppool.
 */
void __telemetry_sort_ppool(TelemetryParams& params) {
    // Store a reverse mapping from the index of a (virtual) page within the
    // buffer to its corresponding DRAM address.
    unordered_map<uint64_t, uint64_t> idx_to_dram_addr;

    for (const auto& [k, v] : params.dram_addr_to_idx) {
        for (const auto idx : v) {
            // The same virtual page shouldn't be mapped to two DRAM adddresses.
            BUG_ON(idx_to_dram_addr.find(idx) != idx_to_dram_addr.end());
            idx_to_dram_addr[idx] = k;
        }
    }

    uint64_t num_batches = 0, num_swaps = 0;
    for (uint64_t i = 0; i < params.pages_per_machine; ++i) {
        // Choose a random DRAM address.
        auto iter = params.dram_addr_to_idx.begin();
        uint64_t dram_addr = iter->first;

        unordered_set<uint64_t> indices_currently_assigned{move(iter->second)};
        unordered_set<uint64_t> indices_to_assign;
        for (uint64_t j = 0; j < params.num_machines; ++j) {
            indices_to_assign.emplace(i + j * params.pages_per_machine);
        }
        BUG_ON(indices_currently_assigned.size() != indices_to_assign.size());

        vector<uint64_t> to_swap_prev;  // Indices that were previously asigned
        vector<uint64_t> to_swap_new;   // Indices that we want to assign
        for (const auto i: indices_to_assign) {
            if (indices_currently_assigned.find(i) == indices_currently_assigned.end()) {
                to_swap_new.emplace_back(i);
            }
        }
        for (const auto i: indices_currently_assigned) {
            if (indices_to_assign.find(i) == indices_to_assign.end()) {
                to_swap_prev.emplace_back(i);
            }
        }
        BUG_ON(to_swap_prev.size() != to_swap_new.size());

        // Update stats
        num_swaps += (to_swap_prev.size());
        ++num_batches;

        // Get the virtual addresses of the pages to swap.
        vector<uint64_t> exchange_virt_addr_vec_1, exchange_virt_addr_vec_2;    
        for (uint64_t k = 0; k < to_swap_prev.size(); ++k) {
            exchange_virt_addr_vec_1.emplace_back(((uint64_t) params.buffer) + to_swap_prev[k] * COLOR_PAGE_SIZE);
            exchange_virt_addr_vec_2.emplace_back(((uint64_t) params.buffer) + to_swap_new[k] * COLOR_PAGE_SIZE);
        }
        BUG_ON(exchange_virt_addr_vec_1.size() != exchange_virt_addr_vec_2.size());

        // Perform the actual swap.
        void **skipped_page_arr_1 = (void **) malloc(sizeof(*skipped_page_arr_1) * exchange_virt_addr_vec_1.size());
        BUG_ON(skipped_page_arr_1 == NULL);
        void **skipped_page_arr_2 = (void **) malloc(sizeof(*skipped_page_arr_2) * exchange_virt_addr_vec_2.size());
        BUG_ON(skipped_page_arr_2 == NULL);
        struct page_xchg_stats stats;
        exchange_pages(0, 0, exchange_virt_addr_vec_1.size(), (void **) exchange_virt_addr_vec_1.data(),
                    (void **) exchange_virt_addr_vec_2.data(), skipped_page_arr_1, skipped_page_arr_2, &stats);
        BUG_ON(stats.num_succeeded != exchange_virt_addr_vec_1.size() << (COLOR_PAGE_SHIFT - PAGE_SHIFT));
        free(skipped_page_arr_1);
        free(skipped_page_arr_2);

        // Perform the swap within our internal data structures.
        for (uint64_t k = 0; k < to_swap_prev.size(); ++k) {
            uint64_t prev_idx = to_swap_prev[k];
            uint64_t new_idx = to_swap_new[k];
            // uint64_t prev_idx_prev_dram = dram_addr;
            BUG_ON(idx_to_dram_addr.find(new_idx) == idx_to_dram_addr.end());
            uint64_t new_idx_prev_dram = idx_to_dram_addr[new_idx];

            // Update dram_addr_to_idx. Note that we only need to erase 
            // dram_addr_to_idx[dram_addr] once.
            BUG_ON(params.dram_addr_to_idx.find(new_idx_prev_dram) == params.dram_addr_to_idx.end());
            params.dram_addr_to_idx[new_idx_prev_dram].erase(new_idx);
            params.dram_addr_to_idx[new_idx_prev_dram].emplace(prev_idx);

            // Update idx_to_dram_addr.
            idx_to_dram_addr[prev_idx] = new_idx_prev_dram;
            idx_to_dram_addr.erase(new_idx);
        }
        BUG_ON(params.dram_addr_to_idx.find(dram_addr) == params.dram_addr_to_idx.end());
        params.dram_addr_to_idx.erase(dram_addr);
    }
}

/*
 * Verifies that telemetry_sort_ppool ran successfully.
 */
void __telemetry_verify_sort(TelemetryParams& params) {
    // Check the number of pages allocated to each machine.
    BUG_ON(params.dram_addr_to_idx.size() != params.pages_per_machine);

    unordered_set<uint64_t> indices_seen;
    for (auto& [k, v] : params.dram_addr_to_idx) {
        vector<uint64_t> indices;
        for (const auto elem: v) {
            indices.emplace_back(elem);
        }

        // Check the number of conflicting pages for each DRAM addr.
        BUG_ON(indices.size() != params.num_machines);
        BUG_ON(indices.size() == 0);
       
        sort(indices.begin(), indices.end());

        // A single page shouldn't map to multiple DRAM addresses.
        BUG_ON(indices_seen.find(indices[0]) != indices_seen.end());
        indices_seen.emplace(indices[0]);

        // Verify that each conflicting page is in the same position for each
        // machine.
        for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
            BUG_ON(indices[i] != indices[0] + i * params.pages_per_machine);
        }
    }
}

/*
 * Enum class indicating which cxl-telemetry function is being called.
 */
enum class TelemetryFunction { sort, verify };

/*
 * Driver for telemetry_sort_ppool and telemetry_verify_sort
 */
void telemetry_driver(int argc, char *argv[], TelemetryFunction tf) {
    if (argc != 3) {
        print_help_message();
        exit(1);
    }

    TelemetryParams params{};

    // Read the function arguments.
    int pool = atoi(argv[0]);
    BUG_ON(pool < 0 || pool >= static_cast<int>(NR_PPOOLS));
    params.pages_per_machine = atol(argv[1]);
    params.num_machines = atol(argv[2]);

    // Get the current ordering of pages in the ppool.
    params.total_num_pages = params.pages_per_machine * params.num_machines;
    params.pagemap = (uint64_t *) malloc(sizeof(*params.pagemap) * params.total_num_pages);
    BUG_ON(params.pagemap == NULL);
    params.buffer = (char *) aligned_alloc(COLOR_PAGE_SIZE, params.total_num_pages * COLOR_PAGE_SIZE);
    BUG_ON(params.buffer == NULL);
    BUG_ON(((uint64_t) params.buffer) % COLOR_PAGE_SIZE != 0);
    __get_current_ordering(params);

    // Call the actual function.
    switch (tf) {
        case TelemetryFunction::sort:
            __telemetry_sort_ppool(params);
            break;
        case TelemetryFunction::verify:
            __telemetry_verify_sort(params);
            break;
    }

    free(params.buffer);
    free(params.pagemap);
}

} // namespace telemetry

/********* END TELEMETRY SECTION **********/

int main(int argc, char *argv[]) {
	if (argc == 1) {
		print_help_message();
		exit(1);
	}
	char *op_type_str = argv[1];
	if (strcmp(op_type_str, "set_color") == 0) {
		set_color(argc - 2, argv + 2);
	} else if (strcmp(op_type_str, "fill_color") == 0) {
		fill_color(argc - 2, argv + 2);
	} else if (strcmp(op_type_str, "set_ppool") == 0) {
		set_ppool(argc - 2, argv + 2);
	} else if (strcmp(op_type_str, "unset_ppool") == 0) {
		unset_ppool(argc - 2, argv + 2);
	} else if (strcmp(op_type_str, "fill_ppool") == 0) {
		fill_ppool(argc - 2, argv + 2);
	} else if (strcmp(op_type_str, "clean_ppool") == 0) {
		clean_ppool(argc - 2, argv + 2);
	} else if (strcmp(op_type_str, "shuffle_ppool") == 0) {
		shuffle_ppool(argc - 2, argv + 2);
	} else if (strcmp(op_type_str, "verify_ppool") == 0) {
		verify_ppool(argc - 2, argv + 2);
	} else if (strcmp(op_type_str, "report_ppool") == 0) {
		report_ppool(argc - 2, argv + 2);
	} else if (strcmp(op_type_str, "drain_lru") == 0) {
		drain_lru(argc - 2, argv + 2);
	} else if (strcmp(op_type_str, "verify_order") == 0) {
		verify_order(argc - 2, argv + 2);
	} else if (strcmp(op_type_str, "organize_ppool") == 0) {
		char *buffer;
		organize_ppool(argc - 2, argv + 2, &buffer);
		if (argc - 2 == 5)
			organize_for_uncontended_pages(argc - 2, argv + 2, buffer);
	} else if (strcmp(op_type_str, "verify_organize") == 0) {
		verify_organize(argc - 2, argv + 2);
	} else if (strcmp(op_type_str, "sort_ppool") == 0 || strcmp(op_type_str, "telemetry_sort_ppool") == 0) {
		telemetry::telemetry_driver(argc - 2, argv + 2, telemetry::TelemetryFunction::sort);
	} else if (strcmp(op_type_str, "verify_sort") == 0 || strcmp(op_type_str, "telemetry_verify_sort") == 0) {
		telemetry::telemetry_driver(argc - 2, argv + 2, telemetry::TelemetryFunction::verify);
	} else {
		// Default: set_color
		set_color(argc - 1, argv + 1);
	}
	return 0;
}

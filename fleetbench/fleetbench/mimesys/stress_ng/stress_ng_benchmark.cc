// Copyright 2022 The Fleetbench Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>
#include <unistd.h>
#include <ctype.h>
#include <sched.h>

#include "benchmark/benchmark.h"
#include "fleetbench/common/common.h"
#include "fleetbench/dynamic_registrar.h"
#include "fleetbench/mimesys/stress_ng/stress_ng_benchmark.h"

#include "fleetbench/mimesys/stress_ng/stress-ng.h"
#include "fleetbench/mimesys/stress_ng/core-pragma.h"
#include "fleetbench/mimesys/stress_ng/core-time.h"

#include "fleetbench/mimesys/stress_ng/core-affinity.h"
#include "fleetbench/mimesys/stress_ng/core-builtin.h"
#include "fleetbench/mimesys/stress_ng/core-cpu-cache.h"
#include "fleetbench/mimesys/stress_ng/core-killpid.h"
#include "fleetbench/mimesys/stress_ng/core-out-of-memory.h"
#include "fleetbench/mimesys/stress_ng/core-pragma.h"
#include "fleetbench/mimesys/stress_ng/core-mmap.h"
#include "fleetbench/mimesys/stress_ng/core-numa.h"
#include "fleetbench/mimesys/stress_ng/core-asm-x86.h"
#include "fleetbench/mimesys/stress_ng/core-asm-riscv.h"
#include "fleetbench/mimesys/stress_ng/core-put.h"

#include "fleetbench/mimesys/stress_ng/core-cpu.h"
#include "fleetbench/mimesys/stress_ng/core-nt-store.h"
#include "fleetbench/mimesys/stress_ng/core-target-clones.h"

#include <math.h>

#if defined(HAVE_SYS_SENDFILE_H)
#include <sys/sendfile.h>
#else
UNEXPECTED
#endif

// All benchmarks in this file are for cold lookups.
namespace fleetbench {
namespace stress_ng {
namespace readahead_ {
/*
 * Copyright (C) 2013-2021 Canonical, Ltd.
 * Copyright (C) 2021-2025 Colin Ian King.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 */

#define MIN_READAHEAD_BYTES	(1 * MB)
#define MAX_READAHEAD_BYTES	(MAX_FILE_LIMIT)
#define DEFAULT_READAHEAD_BYTES	(64 * MB)

#define BUF_ALIGNMENT		(4096)
#define BUF_SIZE		(4096)
#define MAX_OFFSETS		(16)

typedef uint64_t	buffer_t;

static void OPTIMIZE3 stress_readahead_generate_offsets(
	off_t *offsets,
	const uint64_t rounded_readahead_bytes)
{
	size_t i;

	for (i = 0; i < MAX_OFFSETS; i++)
		offsets[i] = (off_t)stress_mwc64modn(rounded_readahead_bytes - BUF_SIZE) & ~(BUF_SIZE - 1);
}

static void OPTIMIZE3 stress_readahead_modify_offsets(off_t *offsets)
{
	size_t i;

	for (i = 0; i < MAX_OFFSETS; i++)
		offsets[i] = ((offsets[i] * 31) >> 5) & ~(BUF_SIZE - 1);
}

static int do_readahead(
	stress_args_t *args,
	const int fd,
	const char *fs_type,
	off_t *offsets)
{
	size_t i;

	for (i = 0; i < MAX_OFFSETS; i++) {
		if (readahead(fd, offsets[i], BUF_SIZE) < 0) {
			pr_fail("%s: ftruncate failed, errno=%d (%s)%s\n",
				args->name, errno, strerror(errno), fs_type);
			return -1;
		}
	}
	return 0;
}

/*
 *  stress_readahead
 *	stress file system cache via readahead calls
 */
static int stress_readahead(stress_args_t *args)
{
	buffer_t *buf = NULL;
	uint64_t rounded_readahead_bytes, i;
	uint64_t readahead_bytes = DEFAULT_READAHEAD_BYTES;
	uint64_t misreads = 0;
	uint64_t baddata = 0;
	int ret, rc = EXIT_FAILURE;
	char filename[PATH_MAX];
	int flags = O_CREAT | O_RDWR | O_TRUNC;
	int fd, fd_wr;
	struct stat statbuf;
	const char *fs_type;
	off_t offsets[MAX_OFFSETS] ALIGN64;
	int generate_offsets = 0;
	const bool verify = !!(g_opt_flags & OPT_FLAGS_VERIFY);

	if (!stress_get_setting("readahead-bytes", &readahead_bytes)) {
		if (g_opt_flags & OPT_FLAGS_MAXIMIZE)
			readahead_bytes = MAX_32;
		if (g_opt_flags & OPT_FLAGS_MINIMIZE)
			readahead_bytes = MIN_READAHEAD_BYTES;
	}
	readahead_bytes /= args->instances;
	if (readahead_bytes < MIN_READAHEAD_BYTES)
		readahead_bytes = MIN_READAHEAD_BYTES;

	ret = stress_temp_dir_mk_args(args);
	if (ret < 0)
		return stress_exit_status(-rc);

	ret = posix_memalign((void **)&buf, BUF_ALIGNMENT, BUF_SIZE);
	if (ret || !buf) {
		rc = stress_exit_status(errno);
		pr_err("%s: cannot allocate buffer\n", args->name);
		(void)stress_temp_dir_rm_args(args);
		return rc;
	}

	(void)stress_temp_filename_args(args,
		filename, sizeof(filename), stress_mwc32());

	fd = open(filename, flags, S_IRUSR | S_IWUSR);
	if (fd < 0) {
		rc = stress_exit_status(errno);
		pr_fail("%s: open %s failed, errno=%d (%s)\n",
			args->name, filename, errno, strerror(errno));
		goto finish;
	}
	fs_type = stress_get_fs_type(filename);

	/* write-only open, ignore failure */
	fd_wr = open(filename, O_WRONLY, S_IRUSR | S_IWUSR);

	if (ftruncate(fd, (off_t)0) < 0) {
		rc = stress_exit_status(errno);
		pr_fail("%s: ftruncate failed, errno=%d (%s)%s\n",
			args->name, errno, strerror(errno), fs_type);
		goto close_finish;
	}
	(void)shim_unlink(filename);

#if defined(HAVE_POSIX_FADVISE) &&	\
    defined(POSIX_FADV_DONTNEED)
	if (posix_fadvise(fd, 0, (off_t)readahead_bytes, POSIX_FADV_DONTNEED) < 0) {
		pr_fail("%s: posix_fadvise failed, errno=%d (%s)%s\n",
			args->name, errno, strerror(errno), fs_type);
		goto close_finish;
	}

	/* Invalid lengths */
	(void)posix_fadvise(fd, 0, (off_t)~0, POSIX_FADV_DONTNEED);
	(void)posix_fadvise(fd, 0, (off_t)-1, POSIX_FADV_DONTNEED);
	/* Invalid offset */
	(void)posix_fadvise(fd, (off_t)-1, 1, POSIX_FADV_DONTNEED);
#endif

	/* Sequential Write */
	for (i = 0; i < readahead_bytes; i += BUF_SIZE) {
		ssize_t pret;
		size_t j;
		const off_t o = i / BUF_SIZE;
seq_wr_retry:
		if (UNLIKELY(!stress_continue_flag())) {
			pr_inf("%s: test expired during test setup "
				"(writing of data file)\n", args->name);
			rc = EXIT_SUCCESS;
			goto close_finish;
		}

PRAGMA_UNROLL_N(8)
		for (j = 0; j < (BUF_SIZE / sizeof(*buf)); j++)
			buf[j] = (buffer_t)o + j;

		pret = pwrite(fd, buf, BUF_SIZE, (off_t)i);
		if (pret <= 0) {
			if ((errno == EAGAIN) || (errno == EINTR))
				goto seq_wr_retry;
			if (errno == ENOSPC)
				break;
			if (errno) {
				pr_fail("%s: pwrite failed, errno=%d (%s)%s\n",
					args->name, errno, strerror(errno), fs_type);
				goto close_finish;
			}
		}
	}

	if (shim_fstat(fd, &statbuf) < 0) {
		pr_fail("%s: fstat failed, errno=%d (%s)%s\n",
			args->name, errno, strerror(errno), fs_type);
		goto close_finish;
	}

	/* Round to write size to get no partial reads */
	rounded_readahead_bytes = (uint64_t)statbuf.st_size -
		(uint64_t)(statbuf.st_size % BUF_SIZE);

	stress_readahead_generate_offsets(offsets, rounded_readahead_bytes);

	do {
		if (UNLIKELY(do_readahead(args, fd, fs_type, offsets) < 0))
			goto close_finish;

		for (i = 0; i < MAX_OFFSETS; i++) {
			ssize_t pret;
rnd_rd_retry:
			if (UNLIKELY(!stress_continue(args)))
				break;

			pret = pread(fd, buf, BUF_SIZE, offsets[i]);
			if (UNLIKELY(pret <= 0)) {
				if ((errno == EAGAIN) || (errno == EINTR))
					goto rnd_rd_retry;
				if (errno) {
					pr_fail("%s: read failed, errno=%d (%s)%s\n",
						args->name, errno, strerror(errno), fs_type);
					goto close_finish;
				}
				continue;
			}
			if (UNLIKELY(pret != BUF_SIZE))
				misreads++;

			if (verify) {
				size_t j;
				const off_t o = offsets[i] / BUF_SIZE;

PRAGMA_UNROLL_N(8)
				for (j = 0; j < (BUF_SIZE / sizeof(*buf)); j++) {
					const buffer_t v = (buffer_t)o + j;

					if (UNLIKELY(buf[j] != v))
						baddata++;
				}
				if (UNLIKELY(baddata)) {
					pr_fail("%s: error in data between 0x%" PRIxMAX " and 0x%" PRIxMAX "\n",
						args->name,
						(intmax_t)offsets[i],
						(intmax_t)offsets[i] + BUF_SIZE - 1);
				}
			}
      args->ci.counter++;
		}

#if defined(HAVE_POSIX_FADVISE) &&	\
    defined(POSIX_FADV_DONTNEED)
		VOID_RET(int, posix_fadvise(fd, 0, (off_t)readahead_bytes, POSIX_FADV_DONTNEED));
#endif

                /* Exercise illegal fd */
                VOID_RET(ssize_t, readahead(~0, 0, 512));

		/* Exercise zero size readahead */
                VOID_RET(ssize_t, readahead(fd, 0, 0));

		/* Exercise invalid readahead on write-only file, EBADF */
		if (fd_wr >= 0) {
			VOID_RET(ssize_t, readahead(fd_wr, 0, 512));
		}

                /* Exercise large sizes and illegal sizes */
		for (i = 15; i < sizeof(size_t) * 8; i += 4) {
			VOID_RET(ssize_t, readahead(fd, 0, 1ULL << i));
		}

		if (LIKELY(generate_offsets++ < 16)) {
			stress_readahead_modify_offsets(offsets);
		} else {
			stress_readahead_generate_offsets(offsets, rounded_readahead_bytes);
			generate_offsets = 0;
		}

	} while (stress_continue(args));

	rc = EXIT_SUCCESS;
close_finish:
	stress_set_proc_state(args->name, STRESS_STATE_DEINIT);
	if (fd_wr >= 0)
		(void)close(fd_wr);
	(void)close(fd);
finish:
	stress_set_proc_state(args->name, STRESS_STATE_DEINIT);
	free(buf);
	(void)stress_temp_dir_rm_args(args);

	if (misreads)
		pr_dbg("%s: %" PRIu64 " incomplete random reads\n",
			args->name, misreads);

	return rc;
}

static const stress_opt_t opts[] = {
	{ OPT_readahead_bytes, "readahead-bytes", TYPE_ID_UINT64_BYTES_FS, MIN_READAHEAD_BYTES, MAX_READAHEAD_BYTES, NULL },
	END_OPT,
};

static const stress_help_t help[] = {
	{ NULL,	"readahead N",		"start N workers exercising file readahead" },
	{ NULL,	"readahead-bytes N",	"size of file to readahead on (default is 1GB)" },
	{ NULL,	"readahead-ops N",	"stop after N readahead bogo operations" },
	{ NULL,	NULL,			NULL }
};

const stressor_info_t stress_readahead_info = {
	.stressor = stress_readahead,
	.cls = CLASS_IO | CLASS_OS,
	.opts = opts,
	.verify = VERIFY_OPTIONAL,
	.help = help
};

void BM_STRESS_NG_Readahead(benchmark::State& state) {
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;			/* shared memory */
  double g_opt_timeout = 1.0;
  const struct stressor_info *info = &stress_readahead_info;
  stats.start = stress_time_now();

  args.stats = &stats;
  args.name = "readahead";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  // run
  for (auto _ : state) {
    // Reset the counter for each iteration
    args.ci.counter = 0;

    // Call the stress_readahead function
    int rc = stress_readahead(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Readahead stress test failed");
      return;
    }
  }
}
} // namespace readahead

namespace readahead_1MB_ {

extern "C" {
    extern const stressor_info_t stress_readahead_info;
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Readahead_1MB(benchmark::State& state) {
  uint64_t ra_bytes = 1048576ULL;
  stress_set_setting_global("readahead-bytes", TYPE_ID_UINT64_BYTES_FS, &ra_bytes);
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 60.0;
  const struct stressor_info *info = &stress_readahead_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "readahead";
  args.max_ops = 16;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Readahead 1MB stress test failed");
      return;
    }
  }
}
} // namespace readahead_1MB_

namespace readahead_4MB_ {

extern "C" {
    extern const stressor_info_t stress_readahead_info;
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Readahead_4MB(benchmark::State& state) {
  uint64_t ra_bytes = 4194304ULL;
  stress_set_setting_global("readahead-bytes", TYPE_ID_UINT64_BYTES_FS, &ra_bytes);
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 60.0;
  const struct stressor_info *info = &stress_readahead_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "readahead";
  args.max_ops = 16;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Readahead 4MB stress test failed");
      return;
    }
  }
}
} // namespace readahead_4MB_

namespace readahead_8MB_ {

extern "C" {
    extern const stressor_info_t stress_readahead_info;
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Readahead_8MB(benchmark::State& state) {
  uint64_t ra_bytes = 8388608ULL;
  stress_set_setting_global("readahead-bytes", TYPE_ID_UINT64_BYTES_FS, &ra_bytes);
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 60.0;
  const struct stressor_info *info = &stress_readahead_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "readahead";
  args.max_ops = 16;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Readahead 8MB stress test failed");
      return;
    }
  }
}
} // namespace readahead_8MB_

namespace readahead_16MB_ {

extern "C" {
    extern const stressor_info_t stress_readahead_info;
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Readahead_16MB(benchmark::State& state) {
  uint64_t ra_bytes = 16777216ULL;
  stress_set_setting_global("readahead-bytes", TYPE_ID_UINT64_BYTES_FS, &ra_bytes);
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 60.0;
  const struct stressor_info *info = &stress_readahead_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "readahead";
  args.max_ops = 16;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Readahead 16MB stress test failed");
      return;
    }
  }
}
} // namespace readahead_16MB_

namespace tlb_shootdown_ {

#include "stress-ng.h"
#include "core-affinity.h"
#include "core-builtin.h"
#include "core-cpu-cache.h"
#include "core-killpid.h"
#include "core-out-of-memory.h"
#include "core-pragma.h"

#include <ctype.h>
#include <sched.h>

static const stress_help_t help[] = {
	{ NULL,	"tlb-shootdown N",	"start N workers that force TLB shootdowns" },
	{ NULL,	"tlb-shootdown-ops N",	"stop after N TLB shootdown bogo ops" },
	{ NULL,	NULL,			NULL }
};

#if defined(HAVE_SCHED_GETAFFINITY) && 	\
    defined(HAVE_MPROTECT)

#define MAX_TLB_PROCS		(8)
#define MIN_TLB_PROCS		(2)
#define MMAP_PAGES		(512)
#define MMAP_FD_PAGES		(4)
#define STRESS_CACHE_LINE_SHIFT	(6)	/* Typical 64 byte size */
#define STRESS_CACHE_LINE_SIZE	(1 << STRESS_CACHE_LINE_SHIFT)

/*
 * stress_tlb_interrupts()
 *	parse /proc/interrupts for per CPU TLB shootdown count
 */
static uint64_t stress_tlb_interrupts(void)
{
#if defined(__linux__)
	FILE *fp;
	char buffer[8192];
	uint64_t total = 0;

	fp = fopen("/proc/interrupts", "r");
	if (!fp)
		return 0ULL;

	(void)shim_memset(buffer, 0, sizeof(buffer));
	while (fgets(buffer, sizeof(buffer), fp) != NULL) {
		char *ptr;
		char *eptr;
		long long val;

		ptr = strstr(buffer, "TLB:");
		if (!ptr)
			continue;

		ptr += 4; /* skip over TLB: */
		while (*ptr) {
			/* skip over spaces */
			while (*ptr == ' ')
				ptr++;
			/* end of string? */
			if (!*ptr)
				break;
			/* not a digit? */
			if (!isdigit((int)*ptr))
				break;

			eptr = NULL;
			val = strtoll(ptr, &eptr, 10);
			/* no number parsed? */
			if (!eptr)
				break;
			/* should be positive */
			if (val < 0)
				break;
			/* sum per CPU TLB shootdown count */
			total += (uint64_t)val;
			ptr = eptr;
		}
		break;
	}
	(void)fclose(fp);

	return total;
#else
	return 0ULL;
#endif
}

/*
 *  stress_tlb_shootdown_read_mem()
 *	read from every cache line in mem
 */
static inline void OPTIMIZE3 stress_tlb_shootdown_read_mem(
	const uint8_t *mem,
	const size_t size,
	const size_t page_size)
{
	const volatile uint8_t *vmem;

	for (vmem = mem; vmem < mem + size; vmem += page_size) {
		size_t m;

		for (m = 0; m < page_size; ) {
			(void)vmem[m];
			m += STRESS_CACHE_LINE_SIZE;
			(void)vmem[m];
			m += STRESS_CACHE_LINE_SIZE;
			(void)vmem[m];
			m += STRESS_CACHE_LINE_SIZE;
			(void)vmem[m];
			m += STRESS_CACHE_LINE_SIZE;
			(void)vmem[m];
			m += STRESS_CACHE_LINE_SIZE;
			(void)vmem[m];
			m += STRESS_CACHE_LINE_SIZE;
			(void)vmem[m];
			m += STRESS_CACHE_LINE_SIZE;
			(void)vmem[m];
			m += STRESS_CACHE_LINE_SIZE;
		}
	}
}

/*
 *  stress_tlb_shootdown_read_mem()
 *	write to every cache line in mem
 */
static inline void OPTIMIZE3 stress_tlb_shootdown_write_mem(
	uint8_t *mem,
	const size_t size,
	const size_t page_size)
{
	volatile uint8_t *vmem;
	const uint8_t rnd8 = stress_mwc8();

	for (vmem = mem; vmem < mem + size; vmem += page_size) {
		size_t m;

		for (m = 0; m < page_size; ) {
			vmem[m] = m + rnd8;
			m += STRESS_CACHE_LINE_SIZE;
			vmem[m] = m + rnd8;
			m += STRESS_CACHE_LINE_SIZE;
			vmem[m] = m + rnd8;
			m += STRESS_CACHE_LINE_SIZE;
			vmem[m] = m + rnd8;
			m += STRESS_CACHE_LINE_SIZE;
			vmem[m] = m + rnd8;
			m += STRESS_CACHE_LINE_SIZE;
			vmem[m] = m + rnd8;
			m += STRESS_CACHE_LINE_SIZE;
			vmem[m] = m + rnd8;
			m += STRESS_CACHE_LINE_SIZE;
			vmem[m] = m + rnd8;
			m += STRESS_CACHE_LINE_SIZE;
		}
	}
	stress_cpu_data_cache_flush((void *)mem, size);
}

/*
 *  stress_tlb_shootdown_read_mem()
 *	mmap with retries
 */
static void *stress_tlb_shootdown_mmap(
	stress_args_t *args,
	void *addr,
	size_t length,
	int prot,
	int flags,
	int fd,
	off_t offset)
{
	int retry = 128;
	void *mem;

	do {
		mem = mmap(addr, length, prot, flags, fd, offset);
		if (LIKELY((void *)mem != MAP_FAILED))
			return mem;
		if ((errno == EAGAIN) ||
		    (errno == ENOMEM) ||
		    (errno == ENFILE)) {
			retry--;
		} else {
			break;
		}
	} while (retry > 0);

	pr_inf_skip("%s: mmap failed, errno=%d (%s), skipping stressor\n",
		args->name, errno, strerror(errno));
	return mem;
}

/*
 *  stress_tlb_shootdown()
 *	stress out TLB shootdowns
 */
static int stress_tlb_shootdown(stress_args_t *args)
{
	double rate, t_begin, duration;
	uint64_t tlb_begin, tlb_end;
	const size_t page_size = args->page_size;
	const size_t page_mask = ~(page_size - 1);
	const size_t mmap_size = page_size * MMAP_PAGES;
	const size_t mmap_mask = mmap_size - 1;
	const size_t cache_lines = mmap_size >> STRESS_CACHE_LINE_SHIFT;
	size_t offset;
	uint32_t *cpus;
	const uint32_t n_cpus = stress_get_usable_cpus(&cpus, true);
	stress_pid_t *s_pids, *s_pids_head = NULL;
	const pid_t pid = getpid();
	int rc = EXIT_SUCCESS;
	uint32_t tlb_procs, i;
	uint8_t *mem;
#if defined(HAVE_MADVISE) &&	\
    defined(MADV_DONTNEED)
	int fd, ret;
	uint8_t *memfd;
	const size_t mmapfd_size = page_size * MMAP_FD_PAGES;
	const size_t mmapfd_mask = mmapfd_size - 1;
	char filename[PATH_MAX];
#endif

	s_pids = stress_s_pids_mmap(MAX_TLB_PROCS);
	if (s_pids == MAP_FAILED) {
		pr_inf_skip("%s: failed to mmap %d PIDs, skipping stressor\n", args->name, MAX_TLB_PROCS);
		rc = EXIT_NO_RESOURCE;
		goto err_free_cpus;
	}

#if defined(HAVE_MADVISE) &&	\
    defined(MADV_DONTNEED)
	ret = stress_temp_dir_mk_args(args);
	if (ret < 0) {
		rc = stress_exit_status(-ret);
		goto err_s_pids;
	}
	(void)stress_temp_filename_args(args,
		filename, sizeof(filename), stress_mwc32());
	if ((fd = open(filename, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)) < 0) {
		ret = stress_exit_status(errno);
		pr_fail("%s: open on %s failed, errno=%d (%s)\n",
			args->name, filename, errno, strerror(errno));
		rc = ret;
		goto err_rmdir;
	}
	(void)shim_unlink(filename);
	if (ftruncate(fd, mmapfd_size) < 0) {
		pr_fail("%s: ftruncate to %zu bytes on %s failed, errno=%d (%s)\n",
			args->name, mmapfd_size, filename, errno, strerror(errno));
		rc = EXIT_NO_RESOURCE;
		goto err_close;
	}
	memfd = (uint8_t*)stress_tlb_shootdown_mmap(args, NULL, mmapfd_size,
			PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
	if ((void *)memfd == MAP_FAILED) {
		rc = EXIT_NO_RESOURCE;
		goto err_close;
	}
#if defined(MADV_NOHUGEPAGE)
	(void)shim_madvise(memfd, mmapfd_size, MADV_NOHUGEPAGE);
#endif
#endif

	mem = (uint8_t*)stress_tlb_shootdown_mmap(args, NULL, mmap_size,
			PROT_WRITE | PROT_READ,
			MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	if ((void *)mem == MAP_FAILED) {
		rc = EXIT_NO_RESOURCE;
		goto err_munmap_memfd;
	}
#if defined(MADV_NOHUGEPAGE)
	(void)shim_madvise(mem, mmap_size, MADV_NOHUGEPAGE);
#endif
	stress_set_vma_anon_name(mem, mmap_size, "tlb-shootdown-buffer");
	(void)shim_memset(mem, 0xff, mmap_size);

	tlb_procs = n_cpus;
	if (tlb_procs > MAX_TLB_PROCS)
		tlb_procs = MAX_TLB_PROCS;
	if (tlb_procs < MIN_TLB_PROCS)
		tlb_procs = MIN_TLB_PROCS;

	t_begin = stress_time_now();
	tlb_begin = stress_tlb_interrupts();

	for (i = 0; i < tlb_procs; i++)
		stress_sync_start_init(&s_pids[i]);

	for (i = 0; i < tlb_procs; i++) {
		uint32_t cpu_idx = 0;
		const size_t stride = (137 + (size_t)stress_get_next_prime64((uint64_t)cache_lines)) << STRESS_CACHE_LINE_SHIFT;

		s_pids[i].pid = fork();
		if (s_pids[i].pid < 0) {
			continue;
		} else if (s_pids[i].pid == 0) {
			cpu_set_t mask;
			double t_start, t_next;

			s_pids[i].pid = getpid();

			stress_parent_died_alarm();
			(void)sched_settings_apply(true);

			/* Make sure this is killable by OOM killer */
			stress_set_oom_adjustment(args, true);

                        stress_sync_start_wait_s_pid(&s_pids[i]);

			if (LIKELY(n_cpus > 0)) {
				CPU_ZERO(&mask);
				CPU_SET((int)cpus[cpu_idx], &mask);
				(void)sched_setaffinity(args->pid, sizeof(mask), &mask);
			}

			t_start = stress_time_now();
			t_next = t_start + 1.0;

			do {
				size_t l;
				size_t k = stress_mwc32() & mmap_mask;
				const uint8_t rnd8 = stress_mwc8();
				volatile uint8_t *vmem;

				offset = (stress_mwc32() & mmap_mask) & page_mask;
				(void)mprotect(mem + offset, page_size, PROT_READ);
				stress_tlb_shootdown_read_mem(mem + offset, page_size, page_size);

				(void)mprotect(mem + offset, page_size, PROT_WRITE);
				stress_tlb_shootdown_write_mem(mem + offset, page_size, page_size);

				vmem = mem;
				(void)mprotect(mem, mmap_size, PROT_READ);
PRAGMA_UNROLL_N(8)
				for (l = 0; l < cache_lines; l++) {
					(void)vmem[k];
					k = (k + stride) & mmap_mask;
				}
				(void)mprotect(mem, mmap_size, PROT_WRITE);
PRAGMA_UNROLL_N(8)
				for (l = 0; l < cache_lines; l++) {
					vmem[k] = (uint8_t)(k + rnd8);
					k = (k + stride) & mmap_mask;
				}
				(void)mprotect(mem, mmap_size, PROT_READ | PROT_WRITE);
#if defined(SHIM_MADV_DONTNEED)
				offset = (stress_mwc32() & mmapfd_mask) & mmap_mask;

				(void)shim_madvise(mem + offset, page_size, SHIM_MADV_DONTNEED);
				stress_tlb_shootdown_read_mem(mem + offset, page_size, page_size);

				(void)shim_madvise(memfd + offset, page_size, SHIM_MADV_DONTNEED);
				stress_tlb_shootdown_write_mem(memfd, page_size, page_size);
				shim_msync(memfd, mmapfd_size, MS_ASYNC);
#endif
				stress_bogo_inc(args);

				/*
				 *  periodically change cpu affinity
				 */
				if (UNLIKELY((stress_time_now() >= t_next) && (n_cpus > 0))) {
					cpu_idx++;
					cpu_idx = (cpu_idx >= n_cpus) ? 0 : cpu_idx;

					CPU_ZERO(&mask);
					CPU_SET(cpus[cpu_idx], &mask);
					(void)sched_setaffinity(args->pid, sizeof(mask), &mask);
					t_next += 1.0;
				}
			} while (stress_continue(args));

			(void)shim_kill(pid, SIGALRM);
			_exit(0);
		} else {
			stress_sync_start_s_pid_list_add(&s_pids_head, &s_pids[i]);
		}
	}

	do {
#if defined(SHIM_MADV_DONTNEED)
		offset = (stress_mwc32() & mmapfd_mask) & page_mask;
		(void)shim_madvise(memfd + offset, page_size, SHIM_MADV_DONTNEED);
		stress_tlb_shootdown_write_mem(memfd, page_size, page_size);
		(void)shim_msync(memfd, mmapfd_size, MS_SYNC);

		(void)shim_madvise(memfd + offset, page_size, SHIM_MADV_DONTNEED);
		stress_tlb_shootdown_read_mem(memfd + offset, page_size, page_size);
		(void)shim_msync(memfd, mmapfd_size, MS_SYNC);

		offset = (stress_mwc32() & mmap_mask) & page_mask;
		(void)shim_madvise(mem + offset, page_size, SHIM_MADV_DONTNEED);
		stress_tlb_shootdown_read_mem(mem + offset, page_size, page_size);

		(void)shim_madvise(mem + offset, page_size, SHIM_MADV_DONTNEED);
		stress_tlb_shootdown_write_mem(mem + offset, page_size, page_size);
#endif
#if defined(__linux__)
		{
			static const char flush_ceiling[] = "/sys/kernel/debug/x86/tlb_single_page_flush_ceiling";
			char buf[64];
			ssize_t rd_ret;

			rd_ret = stress_system_read(flush_ceiling, buf, sizeof(buf));
			if (rd_ret > 0)
				VOID_RET(ssize_t, stress_system_write(flush_ceiling, buf, rd_ret));
		}
#endif
#if defined(MADV_NOHUGEPAGE) && 	\
    defined(MADV_COLLAPSE)
		(void)shim_madvise(mem, mmap_size, MADV_COLLAPSE);
		(void)shim_madvise(mem, mmap_size, MADV_NOHUGEPAGE);

		(void)shim_madvise(memfd, mmapfd_size, MADV_COLLAPSE);
		(void)shim_madvise(memfd, mmapfd_size, MADV_NOHUGEPAGE);
		(void)shim_msync(memfd, mmapfd_size, MS_SYNC);
#endif

		stress_bogo_inc(args);
	} while (stress_continue(args));

	stress_set_proc_state(args->name, STRESS_STATE_DEINIT);
	tlb_end = stress_tlb_interrupts();
	duration = stress_time_now() - t_begin;

	rate = (duration > 0.0) ? (double)(tlb_end - tlb_begin) / duration : 0.0;
	if (rate > 0)
		stress_metrics_set(args, 0, "TLB shootdowns/sec", rate, STRESS_METRIC_GEOMETRIC_MEAN);

	stress_kill_and_wait_many(args, s_pids, tlb_procs, SIGALRM, true);

	(void)munmap((void *)mem, mmap_size);
err_munmap_memfd:
#if defined(HAVE_MADVISE) &&	\
    defined(MADV_DONTNEED)
	(void)munmap((void *)memfd, mmapfd_size);
err_close:
	(void)close(fd);
err_rmdir:
	(void)stress_temp_dir_rm_args(args);
err_s_pids:
	(void)stress_s_pids_munmap(s_pids, MAX_TLB_PROCS);
#endif
err_free_cpus:
	stress_free_usable_cpus(&cpus);

	return rc;
}

const stressor_info_t stress_tlb_shootdown_info = {
	.stressor = stress_tlb_shootdown,
	.cls = CLASS_OS | CLASS_MEMORY,
	.help = help
};
#else
const stressor_info_t stress_tlb_shootdown_info = {
	.stressor = stress_unimplemented,
	.cls = CLASS_OS | CLASS_MEMORY,
	.help = help,
	.unimplemented_reason = "built without sched_getaffinity() or mprotect() system calls"
};
#endif

void BM_STRESS_NG_TLB_Shootdown(benchmark::State& state) {
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;			/* shared memory */
  double g_opt_timeout = 1.0;
  const struct stressor_info *info = &stress_tlb_shootdown_info;
  stats.start = stress_time_now();

  args.stats = &stats;
  args.name = "tlb_shootdown";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  // run
  for (auto _ : state) {
    // Reset the counter for each iteration
    args.ci.counter = 0;

    // Call the stress_readahead function
    int rc = stress_tlb_shootdown(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("TLB shootdown stress test failed");
      return;
    }
  }
}

} // namespace tlb_shootdown_

namespace radixsort_ {
/*
 * Copyright (C) 2013-2021 Canonical, Ltd.
 * Copyright (C) 2022-2025 Colin Ian King.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 */

#define MIN_RADIXSORT_SIZE	(1 * KB)
#define MAX_RADIXSORT_SIZE	(4 * MB)
#define DEFAULT_RADIXSORT_SIZE	(256 * KB)

static const stress_help_t help[] = {
	{ NULL,	"radixsort N",		"start N workers radix sorting random strings" },
	{ NULL,	"radixsort-method M",	"select sort method [ radixsort-libc | radixsort-nonlibc]" },
	{ NULL,	"radixsort-ops N",	"stop after N radixsort bogo operations" },
	{ NULL,	"radixsort-size N",	"number of strings to sort" },
	{ NULL,	NULL,			NULL }
};

typedef int (*radixsort_func_t)(const unsigned char **base, int nmemb, const unsigned char *table, unsigned endbyte);

typedef struct {
	const char *name;
	const radixsort_func_t radixsort_func;
} stress_radixsort_method_t;

#define STR_SIZE	(8)

static volatile bool do_jmp = true;
static sigjmp_buf jmp_env;

#define IDX(base, i, k) 	(1U + base[(i)][(k)])
#define IDX_T(base, i, k)	(1U + table[base[(i)][(k)]])

static inline void ALWAYS_INLINE radix_count_sort(
	const int size,
	const unsigned short int k,
	const unsigned char *base[],
	const unsigned char *b[],
	const unsigned short int lengths[],
	const unsigned char table[])
{
	int i;
	unsigned int c[257];

	(void)shim_memset(c, 0, sizeof(c));

	if (table) {
		for (i = 0; i < size; i++)
			c[(k < lengths[i]) ? IDX_T(base, i, k) : 0]++;

		for (i = 1; i < 257; i++)
			c[i] += c[i - 1];

		for (i = size - 1; i >= 0; i--) {
			const bool lt = k < lengths[i];
			const int j = IDX_T(base, i, k);
			const int l = lt ? j : 0;

			c[l]--;
			b[c[l]] = base[i];
		}
	} else {
		for (i = 0; i < size; i++)
			c[(k < lengths[i]) ? IDX(base, i, k) : 0]++;

		for (i = 1; i < 257; i++)
			c[i] += c[i - 1];

		for (i = size - 1; i >= 0; i--) {
			const bool lt = k < lengths[i];
			const int j = IDX(base, i, k);
			const int l = lt ? j : 0;

			c[l]--;
			b[c[l]] = base[i];
		}
	}
	(void)shim_memcpy((void *)base, (void *)b, sizeof(*base) * size);
}

static inline ALWAYS_INLINE int radix_strlen(const unsigned char *str, unsigned char endbyte)
{
	const unsigned char *ptr = str;

	while (*ptr != endbyte)
		ptr++;

	return ptr - str;
}

static int radixsort_nonlibc(
	const unsigned char **base,
	int nmemb,
	const unsigned char *table,
	unsigned int endbyte)
{
	const unsigned char **b;
	int digit;
	unsigned short int *lengths, max;
	int i;
	unsigned char endchar;

	if (nmemb < 2)
		return 0;

	b = (const unsigned char **)malloc(sizeof(*b) * nmemb);
	if (!b) {
		errno = ENOMEM;
		return -1;
	}
	lengths = (unsigned short int *)malloc(sizeof(*lengths) * nmemb);
	if (!lengths) {
		free(b);
		errno = ENOMEM;
		return -1;
	}

	endchar = (unsigned char)endbyte;
	max = radix_strlen(base[0], endchar);
	lengths[0] = max;
	for (i = 1; i < nmemb; i++) {
		const short int len = radix_strlen(base[i], endchar);

		lengths[i] = len;
		if (len > max)
			max = len;
	}

	for (digit = max - 1; digit >= 0; digit--)
		radix_count_sort(nmemb, digit, base, b, lengths, table);

	free(lengths);
	free(b);
	return 0;
}

static const stress_radixsort_method_t stress_radixsort_methods[] = {
#if defined(HAVE_LIB_BSD)
	{ "radixsort-libc",	radixsort },
#endif
	{ "radixsort-nonlibc",	radixsort_nonlibc },
};

/*
 *  stress_radixsort_handler()
 *	SIGALRM generic handler
 */
static void MLOCKED_TEXT stress_radixsort_handler(int signum)
{
	(void)signum;

	if (do_jmp) {
		do_jmp = false;
		siglongjmp(jmp_env, 1);		/* Ugly, bounce back */
	}
}

static const char *stress_radixsort_method(const size_t i)
{
	return (i < SIZEOF_ARRAY(stress_radixsort_methods)) ? stress_radixsort_methods[i].name : NULL;
}

static const stress_opt_t opts[] = {
	{ OPT_radixsort_method,	"radixsort-method", TYPE_ID_SIZE_T_METHOD, 0, 0, (void*)stress_radixsort_method },
	{ OPT_radixsort_size,	"radixsort-size",   TYPE_ID_UINT64, MIN_RADIXSORT_SIZE, MAX_RADIXSORT_SIZE, NULL },
	END_OPT,
};

/*
 *  stress_radixsort()
 *	stress radixsort
 */
static int stress_radixsort(stress_args_t *args)
{
	uint64_t radixsort_size = DEFAULT_RADIXSORT_SIZE;
	const unsigned char **data;
	unsigned char *text, *ptr;
	int n, i;
	struct sigaction old_action;
	int ret;
	unsigned char revtable[256];
	size_t radixsort_method = 0;
	NOCLOBBER int rc = EXIT_SUCCESS;

	radixsort_func_t radixsort_func;

	(void)stress_get_setting("radixsort-method", &radixsort_method);

	radixsort_func = stress_radixsort_methods[radixsort_method].radixsort_func;
	if (args->instance == 0)
		pr_inf("%s: using method '%s'\n",
			args->name, stress_radixsort_methods[radixsort_method].name);

	if (!stress_get_setting("radixsort-size", &radixsort_size)) {
		if (g_opt_flags & OPT_FLAGS_MAXIMIZE)
			radixsort_size = MAX_RADIXSORT_SIZE;
		if (g_opt_flags & OPT_FLAGS_MINIMIZE)
			radixsort_size = MIN_RADIXSORT_SIZE;
	}
	n = (int)radixsort_size;

	text = (unsigned char *)calloc((size_t)n, STR_SIZE);
	if (!text) {
		pr_inf_skip("%s: calloc failed allocating %d strings, "
			"skipping stressor\n", args->name, n);
		return EXIT_NO_RESOURCE;
	}
	data = (const unsigned char **)calloc((size_t)n, sizeof(*data));
	if (!data) {
		pr_inf_skip("%s: calloc failed allocating %d string pointers, "
			"skipping stressor\n", args->name, n);
		free(text);
		return EXIT_NO_RESOURCE;
	}

	ret = sigsetjmp(jmp_env, 1);
	if (ret) {
		/*
		 * We return here if SIGALRM jmp'd back
		 */
		(void)stress_sigrestore(args->name, SIGALRM, &old_action);
		goto tidy;
	}

	if (stress_sighandler(args->name, SIGALRM, stress_radixsort_handler, &old_action) < 0) {
		free(data);
		free(text);
		return EXIT_FAILURE;
	}

	for (i = 0; i < 256; i++)
		revtable[i] = (unsigned char)(255 - i);

	/* This is very expensive, do it once */
	for (ptr = text, i = 0; i < n; i++, ptr += STR_SIZE) {
		data[i] = ptr;
		stress_rndstr((char *)ptr, STR_SIZE);
	}

	do {
		/* Sort "random" data */
		(void)radixsort_func(data, n, NULL, 0);
		if (UNLIKELY(!stress_continue_flag()))
			break;

		if (g_opt_flags & OPT_FLAGS_VERIFY) {
			for (i = 0; i < n - 1; i++) {
				if (strcmp((const char *)data[i], (const char *)data[i + 1]) > 0) {
					pr_fail("%s: sort error "
						"detected, incorrect ordering "
						"found\n", args->name);
					rc = EXIT_FAILURE;
					break;
				}
			}
		}

		/* Reverse sort */
		(void)radixsort_func(data, n, revtable, 0);

		if (g_opt_flags & OPT_FLAGS_VERIFY) {
			for (i = 0; i < n - 1; i++) {
				if (strcmp((const char *)data[i], (const char *)data[i + 1]) < 0) {
					pr_fail("%s: sort error "
						"detected, incorrect ordering "
						"found\n", args->name);
					rc = EXIT_FAILURE;
					break;
				}
			}
		}

		/* Randomize first char */
		for (ptr = text, i = 0; i < n; i++, ptr += STR_SIZE)
			*ptr = 'a' + stress_mwc8modn(26);

		stress_bogo_inc(args);
	} while ((rc == EXIT_SUCCESS) && stress_continue(args));

	do_jmp = false;
	(void)stress_sigrestore(args->name, SIGALRM, &old_action);
tidy:
	stress_set_proc_state(args->name, STRESS_STATE_DEINIT);

	free(data);
	free(text);

	return rc;
}

const stressor_info_t stress_radixsort_info = {
	.stressor = stress_radixsort,
	.cls = CLASS_CPU_CACHE | CLASS_CPU | CLASS_MEMORY | CLASS_SORT,
	.opts = opts,
	.verify = VERIFY_OPTIONAL,
	.help = help
};

void BM_STRESS_NG_Radixsort(benchmark::State& state) {
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;			/* shared memory */
  double g_opt_timeout = 1.0;
  const struct stressor_info *info = &stress_radixsort_info;
  stats.start = stress_time_now();

  args.stats = &stats;
  args.name = "radixsort";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  // run
  for (auto _ : state) {
    // Reset the counter for each iteration
    args.ci.counter = 0;

    // Call the stress_readahead function
    int rc = stress_radixsort(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Radix sort stress test failed");
      return;
    }
  }
}
}

namespace fallocate_ {
/*
 * Copyright (C) 2013-2021 Canonical, Ltd.
 * Copyright (C) 2021-2025 Colin Ian King.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 */
#define MIN_FALLOCATE_BYTES	(1 * MB)
#define MAX_FALLOCATE_BYTES	(MAX_FILE_LIMIT)
#define DEFAULT_FALLOCATE_BYTES	(8 * MB)

static const stress_help_t help[] = {
	{ NULL,	"fallocate N",		"start N workers fallocating 16MB files" },
	{ NULL,	"fallocate-bytes N",	"specify size of file to allocate" },
	{ NULL,	"fallocate-ops N",	"stop after N fallocate bogo operations" },
	{ NULL,	NULL,			NULL }
};

static const stress_opt_t opts[] = {
	{ OPT_fallocate_bytes, "fallocate-bytes", TYPE_ID_OFF_T, MIN_FALLOCATE_BYTES, MAX_FALLOCATE_BYTES, NULL },
	END_OPT,
};

#if defined(HAVE_FALLOCATE)

static const int modes[] = {
	0,
#if defined(FALLOC_FL_KEEP_SIZE)
	FALLOC_FL_KEEP_SIZE,
#endif
#if defined(FALLOC_FL_KEEP_SIZE) &&	\
    defined(FALLOC_FL_PUNCH_HOLE)
	FALLOC_FL_KEEP_SIZE | FALLOC_FL_PUNCH_HOLE,
#endif
#if defined(FALLOC_FL_ZERO_RANGE)
	FALLOC_FL_ZERO_RANGE,
#endif
#if defined(FALLOC_FL_COLLAPSE_RANGE)
	FALLOC_FL_COLLAPSE_RANGE,
#endif
#if defined(FALLOC_FL_INSERT_RANGE)
	FALLOC_FL_INSERT_RANGE,
#endif
};

/*
 *  illegal mode flags mixes
 */
static const int illegal_modes[] = {
	~0,
#if defined(FALLOC_FL_PUNCH_HOLE) &&	\
    defined(FALLOC_FL_ZERO_RANGE)
	FALLOC_FL_PUNCH_HOLE | FALLOC_FL_ZERO_RANGE,
#endif
#if defined(FALLOC_FL_PUNCH_HOLE)
	FALLOC_FL_PUNCH_HOLE,
#endif
#if defined(FALLOC_FL_COLLAPSE_RANGE) &&	\
    defined(FALLOC_FL_ZERO_RANGE)
	FALLOC_FL_COLLAPSE_RANGE | FALLOC_FL_ZERO_RANGE,
#endif
#if defined(FALLOC_FL_INSERT_RANGE) &&	\
    defined(FALLOC_FL_ZERO_RANGE)
	FALLOC_FL_INSERT_RANGE | FALLOC_FL_ZERO_RANGE,
#endif
#if defined(FALLOC_FL_UNSHARE_RANGE) && \
    defined(FALLOC_FL_KEEP_SIZE)
	FALLOC_FL_UNSHARE_RANGE | FALLOC_FL_KEEP_SIZE,
#endif
};

/*
 *  stress_fallocate
 *	stress I/O via fallocate and ftruncate
 */
static int stress_fallocate(stress_args_t *args)
{
	int fd_async = -1, ret, pipe_ret = -1, pipe_fds[2] = { -1, -1 };
#if defined(O_SYNC)
	int fd_sync = -1;
#endif
	const int bad_fd = stress_get_bad_fd();
	char filename[PATH_MAX];
	uint64_t ftrunc_errs = 0;
	off_t fallocate_bytes = DEFAULT_FALLOCATE_BYTES;
	int *mode_perms = NULL, all_modes;
	size_t i, mode_count;
	const char *fs_type;
	int count = 0, rc = EXIT_SUCCESS;

	for (all_modes = 0, i = 0; i < SIZEOF_ARRAY(modes); i++)
		all_modes |= modes[i];
	mode_count = stress_flag_permutation(all_modes, &mode_perms);

	if (!stress_get_setting("fallocate-bytes", &fallocate_bytes)) {
		if (g_opt_flags & OPT_FLAGS_MAXIMIZE)
			fallocate_bytes = MAXIMIZED_FILE_SIZE;
		if (g_opt_flags & OPT_FLAGS_MINIMIZE)
			fallocate_bytes = MIN_FALLOCATE_BYTES;
	}

	fallocate_bytes /= args->instances;
	if (fallocate_bytes < (off_t)MIN_FALLOCATE_BYTES)
		fallocate_bytes = (off_t)MIN_FALLOCATE_BYTES;
	ret = stress_temp_dir_mk_args(args);
	if (ret < 0) {
		free(mode_perms);
		return stress_exit_status(-ret);
	}

	(void)stress_temp_filename_args(args,
		filename, sizeof(filename), stress_mwc32());
	if ((fd_async = open(filename, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)) < 0) {
		ret = stress_exit_status(errno);
		pr_fail("%s: open %s failed, errno=%d (%s)\n",
			args->name, filename, errno, strerror(errno));
		(void)stress_temp_dir_rm_args(args);
		free(mode_perms);
		return ret;
	}
#if defined(O_SYNC)
	/* don't worry if this fails, we won't use it fails */
	fd_sync = open(filename, O_RDWR | O_SYNC);
#endif
	fs_type = stress_get_fs_type(filename);
#if defined(HAVE_PATHCONF)
#if defined(_PC_ALLOC_SIZE_MIN)
	VOID_RET(long int, pathconf(filename, _PC_ALLOC_SIZE_MIN));
#endif
#if defined(_PC_FILESIZEBITS)
	VOID_RET(long int, pathconf(filename, _PC_FILESIZEBITS));
#endif
#endif
	(void)shim_unlink(filename);

	pipe_ret = pipe(pipe_fds);

	do {
		const bool use_sync = (fd_sync != -1) && ((count++ & 15) == 15);
#if defined(O_SYNC)
		const int fd = use_sync ? fd_sync : fd_async;
#else
		const int fd = fd_async;
#endif

#if defined(HAVE_POSIX_FALLOCATE)
		ret = shim_posix_fallocate(fd_async, (off_t)0, fallocate_bytes);
#else
		ret = shim_fallocate(fd_async, 0, (off_t)0, fallocate_bytes);
#endif
		if (UNLIKELY(!stress_continue_flag()))
			break;
		(void)shim_fsync(fd);
		if ((ret == 0) && (g_opt_flags & OPT_FLAGS_VERIFY)) {
			struct stat buf;

			if (shim_fstat(fd, &buf) < 0) {
				pr_fail("%s: fstat failed, errno=%d (%s)%s\n",
					args->name, errno, strerror(errno), fs_type);
				rc = EXIT_FAILURE;
			}
			else if (buf.st_size != fallocate_bytes) {
				pr_fail("%s: file size %" PRIdMAX " does not match "
					"the expected file size of %" PRIdMAX "\n",
					args->name, (intmax_t)buf.st_size,
					(intmax_t)fallocate_bytes);
				rc = EXIT_FAILURE;
			}
		}

		if (ftruncate(fd, 0) < 0)
			ftrunc_errs++;
		if (UNLIKELY(!stress_continue_flag()))
			break;
		(void)shim_fsync(fd);
		if (UNLIKELY(!stress_continue_flag()))
			break;

		if (g_opt_flags & OPT_FLAGS_VERIFY) {
			struct stat buf;

			if (shim_fstat(fd, &buf) < 0) {
				pr_fail("%s: fstat failed, errno=%d (%s)\n",
					args->name, errno, strerror(errno));
				rc = EXIT_FAILURE;
			}
			else if (buf.st_size != (off_t)0) {
				pr_fail("%s: file size %" PRIdMAX " does not match "
					"the expected file size " "of 0\n",
					args->name, (intmax_t)buf.st_size);
				rc = EXIT_FAILURE;
			}
		}

		if (ftruncate(fd, fallocate_bytes) < 0)
			ftrunc_errs++;
		(void)shim_fsync(fd);
		if (UNLIKELY(!stress_continue_flag()))
			break;
		if (ftruncate(fd, 0) < 0)
			ftrunc_errs++;
		if (UNLIKELY(!stress_continue_flag()))
			break;
		(void)shim_fsync(fd);
		if (UNLIKELY(!stress_continue_flag()))
			break;

		if (SIZEOF_ARRAY(modes) > 1) {
			/*
			 *  non-portable Linux fallocate()
			 */
			(void)shim_fallocate(fd, 0, (off_t)0, fallocate_bytes);
			if (UNLIKELY(!stress_continue_flag()))
				break;
			(void)shim_fsync(fd);
			if (UNLIKELY(!stress_continue_flag()))
				break;

			for (i = 0; i < 32; i++) {
				const size_t j = stress_mwc32modn((uint32_t)SIZEOF_ARRAY(modes));
				const off_t offset = (off_t)stress_mwc64modn((uint64_t)fallocate_bytes) & ~0xfff;

				if (shim_fallocate(fd, modes[j], offset, 64 * KB) == 0)
					(void)shim_fsync(fd);
				if (UNLIKELY(!stress_continue_flag()))
					break;
			}
			if (ftruncate(fd, 0) < 0)
				ftrunc_errs++;
			if (UNLIKELY(!stress_continue_flag()))
				break;
			(void)shim_fsync(fd);
			if (UNLIKELY(!stress_continue_flag()))
				break;
		}

		stress_bogo_inc(args);
	} while ((rc == EXIT_SUCCESS) && stress_continue(args));

	if (ftrunc_errs)
		pr_dbg("%s: %" PRIu64
			" ftruncate errors occurred.\n", args->name, ftrunc_errs);
	if (pipe_ret == 0) {
		(void)close(pipe_fds[0]);
		(void)close(pipe_fds[1]);
	}
	stress_set_proc_state(args->name, STRESS_STATE_DEINIT);

#if defined(O_SYNC)
	if (fd_sync != -1)
		(void)close(fd_sync);
#endif
	if (fd_async != -1)
		(void)close(fd_async);

	(void)stress_temp_dir_rm_args(args);
	free(mode_perms);

	return rc;
}

const stressor_info_t stress_fallocate_info = {
	.stressor = stress_fallocate,
	.cls = CLASS_FILESYSTEM | CLASS_OS,
	.opts = opts,
	.verify = VERIFY_OPTIONAL,
	.help = help
};
#else
const stressor_info_t stress_fallocate_info = {
	.stressor = stress_unimplemented,
	.cls = CLASS_FILESYSTEM | CLASS_OS,
	.opts = opts,
	.verify = VERIFY_OPTIONAL,
	.help = help,
	.unimplemented_reason = "built without fallocate() system call"
};
#endif

void BM_STRESS_NG_Fallocate(benchmark::State& state) {
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;			/* shared memory */
  double g_opt_timeout = 1.0;
  const struct stressor_info *info = &stress_fallocate_info;
  stats.start = stress_time_now();

  args.stats = &stats;
  args.name = "fallocate";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  // run
  for (auto _ : state) {
    // Reset the counter for each iteration
    args.ci.counter = 0;

    // Call the stress_fallocate function
    int rc = stress_fallocate(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Fallocate stress test failed");
      return;
    }
  }
}
}
namespace fallocate_1MB_ {

extern "C" {
    extern const stressor_info_t stress_fallocate_info;
    int stress_fallocate(stress_args_t *args);
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Fallocate_1MB(benchmark::State& state) {
  off_t fa_bytes = 1048576LL;
  stress_set_setting_global("fallocate-bytes", TYPE_ID_OFF_T, &fa_bytes);
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 5.0;
  const struct stressor_info *info = &stress_fallocate_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "fallocate";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Fallocate 1MB stress test failed");
      return;
    }
  }
}
} // namespace fallocate_1MB_
namespace fallocate_32MB_ {

extern "C" {
    extern const stressor_info_t stress_fallocate_info;
    int stress_fallocate(stress_args_t *args);
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Fallocate_32MB(benchmark::State& state) {
  off_t fa_bytes = 33554432LL;
  stress_set_setting_global("fallocate-bytes", TYPE_ID_OFF_T, &fa_bytes);
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 5.0;
  const struct stressor_info *info = &stress_fallocate_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "fallocate";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Fallocate 32MB stress test failed");
      return;
    }
  }
}
} // namespace fallocate_32MB_
namespace fallocate_128MB_ {

extern "C" {
    extern const stressor_info_t stress_fallocate_info;
    int stress_fallocate(stress_args_t *args);
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Fallocate_128MB(benchmark::State& state) {
  off_t fa_bytes = 134217728LL;
  stress_set_setting_global("fallocate-bytes", TYPE_ID_OFF_T, &fa_bytes);
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 5.0;
  const struct stressor_info *info = &stress_fallocate_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "fallocate";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Fallocate 128MB stress test failed");
      return;
    }
  }
}
} // namespace fallocate_128MB_
namespace fallocate_512MB_ {

extern "C" {
    extern const stressor_info_t stress_fallocate_info;
    int stress_fallocate(stress_args_t *args);
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Fallocate_512MB(benchmark::State& state) {
  off_t fa_bytes = 536870912LL;
  stress_set_setting_global("fallocate-bytes", TYPE_ID_OFF_T, &fa_bytes);
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 5.0;
  const struct stressor_info *info = &stress_fallocate_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "fallocate";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Fallocate 512MB stress test failed");
      return;
    }
  }
}
} // namespace fallocate_512MB_
namespace fallocate_2GB_ {

extern "C" {
    extern const stressor_info_t stress_fallocate_info;
    int stress_fallocate(stress_args_t *args);
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Fallocate_2GB(benchmark::State& state) {
  off_t fa_bytes = 2147483648LL;
  stress_set_setting_global("fallocate-bytes", TYPE_ID_OFF_T, &fa_bytes);
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 5.0;
  const struct stressor_info *info = &stress_fallocate_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "fallocate";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Fallocate 2GB stress test failed");
      return;
    }
  }
}
} // namespace fallocate_2GB_

namespace fallocate_2MB_ {

extern "C" {
    extern const stressor_info_t stress_fallocate_info;
    int stress_fallocate(stress_args_t *args);
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Fallocate_2MB(benchmark::State& state) {
  off_t fa_bytes = 2097152LL;
  stress_set_setting_global("fallocate-bytes", TYPE_ID_OFF_T, &fa_bytes);
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 5.0;
  const struct stressor_info *info = &stress_fallocate_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "fallocate";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Fallocate 2MB stress test failed");
      return;
    }
  }
}
} // namespace fallocate_2MB_

namespace fallocate_4MB_ {

extern "C" {
    extern const stressor_info_t stress_fallocate_info;
    int stress_fallocate(stress_args_t *args);
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Fallocate_4MB(benchmark::State& state) {
  off_t fa_bytes = 4194304LL;
  stress_set_setting_global("fallocate-bytes", TYPE_ID_OFF_T, &fa_bytes);
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 5.0;
  const struct stressor_info *info = &stress_fallocate_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "fallocate";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Fallocate 4MB stress test failed");
      return;
    }
  }
}
} // namespace fallocate_4MB_

namespace fallocate_16MB_ {

extern "C" {
    extern const stressor_info_t stress_fallocate_info;
    int stress_fallocate(stress_args_t *args);
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Fallocate_16MB(benchmark::State& state) {
  off_t fa_bytes = 16777216LL;
  stress_set_setting_global("fallocate-bytes", TYPE_ID_OFF_T, &fa_bytes);
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 5.0;
  const struct stressor_info *info = &stress_fallocate_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "fallocate";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Fallocate 16MB stress test failed");
      return;
    }
  }
}
} // namespace fallocate_16MB_

namespace sendfile_ {
/*
 * Copyright (C) 2013-2021 Canonical, Ltd.
 * Copyright (C) 2022-2025 Colin Ian King.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 */
#define MIN_SENDFILE_SIZE	(1 * KB)
#define MAX_SENDFILE_SIZE	(1 * GB)
#define DEFAULT_SENDFILE_SIZE	(4 * MB)

static const stress_help_t help[] = {
	{ NULL,	"sendfile N",	   "start N workers exercising sendfile" },
	{ NULL,	"sendfile-ops N",  "stop after N bogo sendfile operations" },
	{ NULL,	"sendfile-size N", "size of data to be sent with sendfile" },
	{ NULL,	NULL,		   NULL }
};

static const stress_opt_t opts[] = {
	{ OPT_sendfile_size, "sendfile-size", TYPE_ID_UINT64_BYTES_VM, MIN_SENDFILE_SIZE, MAX_SENDFILE_SIZE, NULL },
	END_OPT,
};

#if defined(HAVE_SYS_SENDFILE_H) &&	\
    defined(HAVE_SENDFILE) &&		\
    NEED_GLIBC(2,1,0)

/*
 *  stress_sendfile
 *	stress reading of a temp file and writing to /dev/null via sendfile
 */
static int stress_sendfile(stress_args_t *args)
{
	char filename[PATH_MAX];
	int i = 0, fdin, fdout, ret, bad_fd, rc = EXIT_SUCCESS;
	size_t sz;
	int64_t sendfile_size = DEFAULT_SENDFILE_SIZE;
	double duration = 0.0, bytes = 0.0, rate;
	int metrics_count = 0;

	if (!stress_get_setting("sendfile-size", &sendfile_size)) {
		if (g_opt_flags & OPT_FLAGS_MAXIMIZE)
			sendfile_size = MAX_SENDFILE_SIZE;
		if (g_opt_flags & OPT_FLAGS_MINIMIZE)
			sendfile_size = MIN_SENDFILE_SIZE;
	}
	sz = (size_t)sendfile_size;

	ret = stress_temp_dir_mk_args(args);
	if (ret < 0)
		return stress_exit_status(-ret);

	(void)stress_temp_filename_args(args,
		filename, sizeof(filename), stress_mwc32());

	if ((fdin = open(filename, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)) < 0) {
		rc = stress_exit_status(errno);
		pr_err("%s: open %s failed, errno=%d (%s)\n",
			args->name, filename, errno, strerror(errno));
		goto dir_out;
	}
#if defined(HAVE_POSIX_FALLOCATE)
	ret = shim_posix_fallocate(fdin, (off_t)0, (off_t)sz);
	errno = ret;
#else
	ret = shim_fallocate(fdin, 0, (off_t)0, (off_t)sz);
#endif
	if (ret != 0) {
		rc = stress_exit_status(errno);
		pr_err("%s: fallocate failed, errno=%d (%s)\n",
			args->name, errno, strerror(errno));
		goto close_in;
	}
	(void)close(fdin);
	if ((fdin = open(filename, O_RDONLY)) < 0) {
		rc = stress_exit_status(errno);
		pr_err("%s: open %s failed, errno=%d (%s)\n",
			args->name, filename, errno, strerror(errno));
		goto dir_out;
	}

	if ((fdout = open("/dev/null", O_WRONLY)) < 0) {
		pr_err("%s: open /dev/null failed, errno=%d (%s)\n",
			args->name, errno, strerror(errno));
		rc = EXIT_FAILURE;
		goto close_in;
	}

	bad_fd = stress_get_bad_fd();

	do {
		off_t offset = 0;
		ssize_t nbytes;
		double t;

		if (LIKELY(metrics_count++ < 1000)) {
			/* fast non-metrics sendfile */
			nbytes = sendfile(fdout, fdin, &offset, sz);
			if (LIKELY(nbytes >= 0))
				goto sendfile_ok;
		} else {
			/* slow metrics sendfile */
			metrics_count = 0;
			t = stress_time_now();
			nbytes = sendfile(fdout, fdin, &offset, sz);
			if (LIKELY(nbytes >= 0)) {
				duration += stress_time_now() - t;
				bytes += (double)nbytes;
				goto sendfile_ok;
			}
		}

		if (errno == ENOSYS) {
			if (args->instance == 0)
				pr_inf_skip("%s: skipping stressor, sendfile not implemented\n",
					args->name);
			rc = EXIT_NOT_IMPLEMENTED;
			goto close_out;
		}
		if (errno == EINTR)
			continue;
		pr_fail("%s: sendfile failed, errno=%d (%s)\n",
			args->name, errno, strerror(errno));
		rc = EXIT_FAILURE;
		goto close_out;

sendfile_ok:
		/* Periodically perform some unusual sendfile calls */
		if (UNLIKELY((i++ & 0xff) == 0)) {
			/* Exercise with invalid destination fd */
			offset = 0;
			(void)sendfile(bad_fd, fdin, &offset, sz);

			/* Exercise with invalid source fd */
			offset = 0;
			(void)sendfile(fdout, bad_fd, &offset, sz);

			/* Exercise with invalid offset */
			offset = -1;
			(void)sendfile(fdout, fdin, &offset, sz);

			/* Exercise with invalid size */
			offset = 0;
			(void)sendfile(fdout, fdin, &offset, (size_t)-1);

			/* Exercise with zero size (should work, no-op) */
			offset = 0;
			(void)sendfile(fdout, fdin, &offset, 0);

			/* Exercise with read-only destination (EBADF) */
			offset = 0;
			(void)sendfile(fdin, fdin, &offset, sz);

			/* Exercise with write-only source (EBADF) */
			offset = 0;
			(void)sendfile(fdout, fdout, &offset, sz);

			/* Exercise truncated read */
			offset = (off_t)(sz - 1);
			(void)sendfile(fdout, fdin, &offset, sz);
		}
		stress_bogo_inc(args);
	} while (stress_continue(args));

	rate = (duration > 0.0) ? bytes / duration : 0.0;
	stress_metrics_set(args, 0, "MB per sec sent to /dev/null",
		rate / (double)MB, STRESS_METRIC_HARMONIC_MEAN);

close_out:
	stress_set_proc_state(args->name, STRESS_STATE_DEINIT);
	(void)close(fdout);
close_in:
	stress_set_proc_state(args->name, STRESS_STATE_DEINIT);
	(void)close(fdin);
	(void)shim_unlink(filename);
dir_out:
	stress_set_proc_state(args->name, STRESS_STATE_DEINIT);
	(void)stress_temp_dir_rm_args(args);

	return rc;
}

const stressor_info_t stress_sendfile_info = {
	.stressor = stress_sendfile,
	.cls = CLASS_PIPE_IO | CLASS_OS,
	.opts = opts,
	.verify = VERIFY_ALWAYS,
	.help = help
};
#else
const stressor_info_t stress_sendfile_info = {
	.stressor = stress_unimplemented,
	.cls = CLASS_PIPE_IO | CLASS_OS,
	.opts = opts,
	.verify = VERIFY_ALWAYS,
	.help = help,
	.unimplemented_reason = "built without sys/sendfile.h or sendfile() system call support"
};
#endif

void BM_STRESS_NG_Sendfile(benchmark::State& state) {
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;			/* shared memory */
  double g_opt_timeout = 1.0;
  const struct stressor_info *info = &stress_sendfile_info;
  stats.start = stress_time_now();

  args.stats = &stats;
  args.name = "sendfile";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  // run
  for (auto _ : state) {
    // Reset the counter for each iteration
    args.ci.counter = 0;

    // Call the stress_sendfile function
    int rc = stress_sendfile(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Sendfile stress test failed");
      return;
    }
  }
}
}
namespace sendfile_1MB_ {

extern "C" {
    extern const stressor_info_t stress_sendfile_info;
    int stress_sendfile(stress_args_t *args);
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Sendfile_1MB(benchmark::State& state) {
  uint64_t sf_bytes = 1048576ULL;
  stress_set_setting_global("sendfile-size", TYPE_ID_UINT64_BYTES_VM, &sf_bytes);
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 5.0;
  const struct stressor_info *info = &stress_sendfile_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "sendfile";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Sendfile 1MB stress test failed");
      return;
    }
  }
}
} // namespace sendfile_1MB_
namespace sendfile_32MB_ {

extern "C" {
    extern const stressor_info_t stress_sendfile_info;
    int stress_sendfile(stress_args_t *args);
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Sendfile_32MB(benchmark::State& state) {
  uint64_t sf_bytes = 33554432ULL;
  stress_set_setting_global("sendfile-size", TYPE_ID_UINT64_BYTES_VM, &sf_bytes);
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 5.0;
  const struct stressor_info *info = &stress_sendfile_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "sendfile";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Sendfile 32MB stress test failed");
      return;
    }
  }
}
} // namespace sendfile_32MB_
namespace sendfile_256MB_ {

extern "C" {
    extern const stressor_info_t stress_sendfile_info;
    int stress_sendfile(stress_args_t *args);
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Sendfile_256MB(benchmark::State& state) {
  uint64_t sf_bytes = 268435456ULL;
  stress_set_setting_global("sendfile-size", TYPE_ID_UINT64_BYTES_VM, &sf_bytes);
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 5.0;
  const struct stressor_info *info = &stress_sendfile_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "sendfile";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Sendfile 256MB stress test failed");
      return;
    }
  }
}
} // namespace sendfile_256MB_
namespace sendfile_1GB_ {

extern "C" {
    extern const stressor_info_t stress_sendfile_info;
    int stress_sendfile(stress_args_t *args);
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Sendfile_1GB(benchmark::State& state) {
  uint64_t sf_bytes = 1073741824ULL;
  stress_set_setting_global("sendfile-size", TYPE_ID_UINT64_BYTES_VM, &sf_bytes);
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 5.0;
  const struct stressor_info *info = &stress_sendfile_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "sendfile";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Sendfile 1GB stress test failed");
      return;
    }
  }
}
} // namespace sendfile_1GB_

namespace mmaphuge_ {
/*
 * Copyright (C) 2013-2021 Canonical, Ltd.
 * Copyright (C) 2022-2025 Colin Ian King.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 */


#define MIN_MMAPHUGE_MMAPS	(1)
#define MAX_MMAPHUGE_MMAPS	(65536)

static const stress_help_t help[] = {
	{ NULL,	"mmaphuge N",		"start N workers stressing mmap with huge mappings" },
	{ NULL, "mmaphuge-file",	"perform mappings on a temporary file" },
	{ NULL,	"mmaphuge-mlock",	"attempt to mlock pages into memory" },
	{ NULL, "mmaphuge-mmaps N",	"select number of memory mappings per iteration" },
	{ NULL, "mmaphuge-numa",	"bind memory mappings to randomly selected NUMA nodes" },
	{ NULL,	"mmaphuge-ops N",	"stop after N mmaphuge bogo operations" },
	{ NULL,	NULL,			NULL }
};

static const stress_opt_t opts[] = {
	{ OPT_mmaphuge_file,  "mmaphuge-file",  TYPE_ID_BOOL, 0, 1, NULL },
	{ OPT_mmaphuge_mlock, "mmaphuge-mlock", TYPE_ID_BOOL, 0, 1, NULL },
	{ OPT_mmaphuge_mmaps, "mmaphuge-mmaps", TYPE_ID_SIZE_T, MIN_MMAPHUGE_MMAPS, MAX_MMAPHUGE_MMAPS, NULL },
	{ OPT_mmaphuge_numa,  "mmaphuge-numa",  TYPE_ID_BOOL, 0, 1, NULL },
	END_OPT
};

#if defined(MAP_HUGETLB)

#define MAX_MMAP_BUFS	(8192)

#if !defined(MAP_HUGE_2MB) && defined(MAP_HUGE_SHIFT)
#define MAP_HUGE_2MB    (21 << MAP_HUGE_SHIFT)
#endif

#if !defined(MAP_HUGE_1GB) && defined(MAP_HUGE_SHIFT)
#define MAP_HUGE_1GB    (30 << MAP_HUGE_SHIFT)
#endif

typedef struct {
	uint8_t	*buf;		/* mapping start */
	size_t	sz;		/* mapping size */
} stress_mmaphuge_buf_t;

typedef struct {
	const int	flags;	/* MMAP flag */
	const size_t	sz;	/* MMAP size */
} stress_mmaphuge_setting_t;

typedef struct {
	stress_mmaphuge_buf_t	*bufs;	/* mmap'd buffers */
	size_t mmaphuge_mmaps;	/* number of mmap'd buffers */
	size_t sz;		/* size of mmap'd file */
	bool mmaphuge_file;	/* true if using mmap'd file */
	bool mmaphuge_mlock;	/* true if using mlocked mmaps */
	bool mmaphuge_numa;	/* true if using numa binding */
	int fd;
#if defined(HAVE_LINUX_MEMPOLICY_H)
	stress_numa_mask_t *numa_mask;
#endif
} stress_mmaphuge_context_t;

static const stress_mmaphuge_setting_t stress_mmap_settings[] =
{
#if defined(MAP_HUGE_2MB)
	{ MAP_HUGETLB | MAP_HUGE_2MB,	2 * MB },
#endif
#if defined(MAP_HUGE_1GB)
	{ MAP_HUGETLB | MAP_HUGE_1GB,	1 * GB },
#endif
	{ MAP_HUGETLB, 1 * GB },
	{ MAP_HUGETLB, 16 * MB },	/* ppc64 */
	{ MAP_HUGETLB, 2 * MB },
	{ 0, 1 * GB },			/* for THP */
	{ 0, 16 * MB },			/* for THP */
	{ 0, 2 * MB },			/* for THP */
};

static int stress_mmaphuge_child(stress_args_t *args, void *v_context)
{
	stress_mmaphuge_context_t *context = (stress_mmaphuge_context_t *)v_context;
	const size_t page_size = args->page_size;
	stress_mmaphuge_buf_t *bufs = (stress_mmaphuge_buf_t *)context->bufs;
	size_t idx = 0;
	int rc = EXIT_SUCCESS;

	do {
		size_t i;

		for (i = 0; i < context->mmaphuge_mmaps; i++)
			bufs[i].buf = (uint8_t *)MAP_FAILED;

		for (i = 0; LIKELY(stress_continue(args) && (i < context->mmaphuge_mmaps)); i++) {
			size_t shmall, freemem, totalmem, freeswap, totalswap, last_freeswap, last_totalswap;
			size_t j;

			stress_get_memlimits(&shmall, &freemem, &totalmem, &last_freeswap, &last_totalswap);

			for (j = 0; j < SIZEOF_ARRAY(stress_mmap_settings); j++) {
				uint8_t *buf = (uint8_t *)MAP_FAILED;
				const size_t sz = stress_mmap_settings[idx].sz;
				int flags = MAP_ANONYMOUS;

				flags |= (stress_mwc1() ? MAP_PRIVATE : MAP_SHARED);
				flags |= stress_mmap_settings[idx].flags;

				if ((g_opt_flags & OPT_FLAGS_OOM_AVOID) && stress_low_memory(page_size))
					break;

				bufs[i].sz = sz;
				/* If we're mapping onto a file, try it first */
				if (context->mmaphuge_file) {
					const off_t offset = 4096 * stress_mwc8modn(16);

					if (sz + offset < context->sz) {
						buf = (uint8_t *)mmap(NULL, sz,
								PROT_READ | PROT_WRITE,
								flags & ~MAP_ANONYMOUS, context->fd, offset);
						if (buf == MAP_FAILED)
							buf = (uint8_t *)mmap(NULL, sz,
								PROT_READ | PROT_WRITE,
								flags & ~MAP_ANONYMOUS, context->fd, 0);
					}
				}
				/* file mapping failed or not mapped yet, try anonymous map */
				if (buf == MAP_FAILED) {
					buf = (uint8_t *)mmap(NULL, sz,
							PROT_READ | PROT_WRITE,
							flags, -1, 0);
				}
				bufs[i].buf = buf;
				idx++;
				if (UNLIKELY(idx >= SIZEOF_ARRAY(stress_mmap_settings)))
					idx = 0;

				if (buf != MAP_FAILED) {
					const uint64_t rndval = stress_mwc64();
					const size_t stride = (page_size * 64) / sizeof(uint64_t);
					uint64_t *ptr, val;
					const uint64_t *buf_end = (uint64_t *)(buf + sz);

#if defined(HAVE_LINUX_MEMPOLICY_H)
					if (context->mmaphuge_numa)
						stress_numa_randomize_pages(context->numa_mask, buf, page_size, sz);
#endif

					if (context->mmaphuge_mlock)
						(void)shim_mlock(buf, sz);

					/* Touch every other 64 pages.. */
					for (val = rndval, ptr = (uint64_t *)buf; ptr < buf_end; ptr += stride, val++) {
						*ptr = val;
					}
					/* ..and sanity check */
					for (val = rndval, ptr = (uint64_t *)buf; ptr < buf_end; ptr += stride, val++) {
						if (UNLIKELY(*ptr != val)) {
							pr_fail("%s: memory %p at offset 0x%zx check error, "
								"got 0x%" PRIx64 ", expecting 0x%" PRIx64 "\n",
								args->name, buf, (uint8_t *)ptr - buf, *ptr, val);
							rc = EXIT_FAILURE;
						}
					}

					stress_bogo_inc(args);
					break;
				}
			}
			stress_get_memlimits(&shmall, &freemem, &totalmem, &freeswap, &totalswap);

			/* Check if we eat into swap */
			if (last_freeswap > freeswap)
				break;
		}

		for (i = 0; LIKELY(stress_continue(args) && (i < context->mmaphuge_mmaps)); i++) {
			if (bufs[i].buf == MAP_FAILED)
				continue;
			/* Try Transparent Huge Pages THP */
#if defined(MADV_HUGEPAGE)
			(void)shim_madvise(bufs[i].buf, bufs[i].sz, MADV_NOHUGEPAGE);
#endif
#if defined(MADV_HUGEPAGE)
			(void)shim_madvise(bufs[i].buf, bufs[i].sz, MADV_HUGEPAGE);
#endif
		}

		for (i = 0; i < context->mmaphuge_mmaps; i++) {
			uint8_t *buf = bufs[i].buf;
			size_t sz;

			if (buf == MAP_FAILED)
				continue;

			sz = bufs[i].sz;
			if (page_size < sz) {
				uint8_t *end_page = buf + (sz - page_size);
				int ret;

				*buf = stress_mwc8();
				*end_page = stress_mwc8();
				/* Unmapping small page may fail on huge pages */
				ret = stress_munmap_retry_enomem((void *)end_page, page_size);
				if (ret == 0)
					ret = stress_munmap_retry_enomem((void *)buf, sz - page_size);
				if (ret != 0)
					(void)stress_munmap_retry_enomem((void *)buf, sz);
			} else {
				*buf = stress_mwc8();
				(void)stress_munmap_retry_enomem((void *)buf, sz);
			}
			bufs[i].buf = (uint8_t *)MAP_FAILED;
		}
	} while (stress_continue(args));

	stress_set_proc_state(args->name, STRESS_STATE_DEINIT);

	return rc;
}

/*
 *  stress_mmaphuge()
 *	stress huge page mmappings and unmappings
 */
static int stress_mmaphuge(stress_args_t *args)
{
	stress_mmaphuge_context_t context;

	int ret;

#if defined(HAVE_LINUX_MEMPOLICY_H)
	context.numa_mask = NULL;
#endif
	context.sz = 16 * MB;
	context.fd = -1;
	context.mmaphuge_mmaps = MAX_MMAP_BUFS;
	if (!stress_get_setting("mmaphuge-mmaps", &context.mmaphuge_mmaps)) {
		if (g_opt_flags & OPT_FLAGS_MAXIMIZE)
			context.mmaphuge_mmaps = MAX_MMAPHUGE_MMAPS;
		if (g_opt_flags & OPT_FLAGS_MINIMIZE)
			context.mmaphuge_mmaps = MIN_MMAPHUGE_MMAPS;
	}
	context.mmaphuge_file = false;
	(void)stress_get_setting("mmaphuge-file", &context.mmaphuge_file);
	context.mmaphuge_numa = false;
	(void)stress_get_setting("mmaphuge-numa", &context.mmaphuge_numa);
	context.mmaphuge_mlock = false;
	(void)stress_get_setting("mmaphuge-mlock", &context.mmaphuge_mlock);

	context.bufs = (stress_mmaphuge_buf_t *)calloc(context.mmaphuge_mmaps, sizeof(*context.bufs));
	if (!context.bufs) {
		pr_inf_skip("%s: cannot allocate buffer array, skipping stressor\n",
			args->name);
		return EXIT_NO_RESOURCE;
	}

	if (context.mmaphuge_file) {
		char filename[PATH_MAX];
		ssize_t rc;

		rc = stress_temp_dir_mk_args(args);
		if (rc < 0) {
			free(context.bufs);
			return stress_exit_status((int)-rc);
		}

		(void)stress_temp_filename_args(args,
			filename, sizeof(filename), stress_mwc32());
		context.fd = open(filename, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
		if (context.fd < 0) {
			rc = stress_exit_status(errno);
			pr_fail("%s: open %s failed, errno=%d (%s)\n",
				args->name, filename, errno, strerror(errno));
			(void)shim_unlink(filename);
			(void)stress_temp_dir_rm_args(args);
			free(context.bufs);

			return (int)rc;
		}
		(void)shim_unlink(filename);
		if (lseek(context.fd, (off_t)(context.sz - args->page_size), SEEK_SET) < 0) {
			pr_fail("%s: lseek failed, errno=%d (%s)\n",
				args->name, errno, strerror(errno));
			(void)close(context.fd);
			(void)stress_temp_dir_rm_args(args);
			free(context.bufs);

			return EXIT_FAILURE;
		}
		/*
		 *  Allocate a 16 MB aligned chunk of data.
		 */
		if (shim_fallocate(context.fd, 0, 0, (off_t)context.sz) < 0) {
			rc = stress_exit_status(errno);
			pr_fail("%s: fallocate of %zu MB failed, errno=%d (%s)\n",
				args->name, (size_t)(context.fd / MB), errno, strerror(errno));
			(void)close(context.fd);
			(void)stress_temp_dir_rm_args(args);
			free(context.bufs);
			return (int)rc;
		}
	}

	if (context.mmaphuge_numa) {
#if defined(HAVE_LINUX_MEMPOLICY_H)
		if (stress_numa_nodes() > 1) {
			context.numa_mask = stress_numa_mask_alloc();
			if (!context.numa_mask) {
				pr_inf("%s: cannot allocate NUMA mask, disabling --mmaphuge-numa\n",
					args->name);
				context.mmaphuge_numa = false;
			}
		} else {
			if (args->instance == 0) {
				pr_inf("%s: only 1 NUMA node available, disabling --mmaphuge-numa\n",
					args->name);
				context.mmaphuge_numa = false;
			}
		}
#else
		if (args->instance == 0)
			pr_inf("%s: --mmaphuge-numa selected but not supported by this system, disabling option\n",
				args->name);
		context.mmaphuge_numa = false;
#endif
	}

	ret = stress_oomable_child(args, (void *)&context, stress_mmaphuge_child, STRESS_OOMABLE_QUIET);

#if defined(HAVE_LINUX_MEMPOLICY_H)
	if (context.numa_mask)
		stress_numa_mask_free(context.numa_mask);
#endif
	free(context.bufs);

	if (context.mmaphuge_file) {
		(void)close(context.fd);
		(void)stress_temp_dir_rm_args(args);
	}

	return ret;
}

const stressor_info_t stress_mmaphuge_info = {
	.stressor = stress_mmaphuge,
	.cls = CLASS_VM | CLASS_OS,
	.opts = opts,
	.verify = VERIFY_ALWAYS,
	.help = help
};

#else

const stressor_info_t stress_mmaphuge_info = {
	.stressor = stress_unimplemented,
	.cls = CLASS_VM | CLASS_OS,
	.opts = opts,
	.verify = VERIFY_ALWAYS,
	.help = help,
	.unimplemented_reason = "built without mmap() MAP_HUGETLB support"
};

#endif

void BM_STRESS_NG_Mmaphuge(benchmark::State& state) {
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;			/* shared memory */
  double g_opt_timeout = 1.0;
  const struct stressor_info *info = &stress_mmaphuge_info;
  stats.start = stress_time_now();

  args.stats = &stats;
  args.name = "mmaphuge";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  // run
  for (auto _ : state) {
    // Reset the counter for each iteration
    args.ci.counter = 0;

    // Call the stress_mmaphuge function
    int rc = stress_mmaphuge(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Mmaphuge stress test failed");
      return;
    }
  }
}
}

namespace cache_ {
/*
 * Copyright (C) 2013-2021 Canonical, Ltd.
 * Copyright (C) 2021-2025 Colin Ian King
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 */


#define CACHE_FLAGS_PREFETCH	(0x0001U)
#define CACHE_FLAGS_CLFLUSH	(0x0002U)
#define CACHE_FLAGS_FENCE	(0x0004U)
#define CACHE_FLAGS_SFENCE	(0x0008U)
#define CACHE_FLAGS_CLFLUSHOPT	(0x0010U)
#define CACHE_FLAGS_CLDEMOTE	(0x0020U)
#define CACHE_FLAGS_CLWB	(0x0040U)
#define CACHE_FLAGS_PREFETCHW	(0x0080U)
#define CACHE_FLAGS_NOAFF	(0x8000U)

#define STRESS_CACHE_MIXED_OPS	(0)
#define STRESS_CACHE_READ	(1)
#define STRESS_CACHE_WRITE	(2)
#define STRESS_CACHE_MAX	(3)

typedef void (*cache_mixed_ops_func_t)(stress_args_t *args,
	uint64_t inc, const uint64_t r,
	uint64_t *pi, uint64_t *pk,
	stress_metrics_t *metrics);
typedef void (*cache_write_page_func_t)(uint8_t *const addr, const uint64_t size);

#define CACHE_FLAGS_MASK	(CACHE_FLAGS_PREFETCH |		\
				 CACHE_FLAGS_CLFLUSH |		\
				 CACHE_FLAGS_FENCE |		\
				 CACHE_FLAGS_SFENCE |		\
				 CACHE_FLAGS_CLFLUSHOPT |	\
				 CACHE_FLAGS_CLDEMOTE |		\
				 CACHE_FLAGS_CLWB | 		\
				 CACHE_FLAGS_PREFETCHW)

typedef struct {
	const uint32_t flag;	/* cache mask flag */
	const char *name;	/* human readable form */
} mask_flag_info_t;

static const mask_flag_info_t mask_flag_info[] = {
	{ CACHE_FLAGS_PREFETCH,		"prefetch" },
	{ CACHE_FLAGS_CLFLUSH,		"flush" },
	{ CACHE_FLAGS_FENCE,		"fence" },
	{ CACHE_FLAGS_SFENCE,		"sfence" },
	{ CACHE_FLAGS_CLFLUSHOPT,	"clflushopt" },
	{ CACHE_FLAGS_CLDEMOTE,		"cldemote" },
	{ CACHE_FLAGS_CLWB,		"clwb" },
	{ CACHE_FLAGS_PREFETCHW,	"prefetchw" },
};

static sigjmp_buf jmp_env;
static volatile int caught_signum;
static volatile uint32_t masked_flags;
static uint64_t disabled_flags;

static const stress_help_t help[] = {
	{ "C N","cache N",	 	"start N CPU cache thrashing workers" },
	{ NULL,	"cache-size N",		"override the default cache size setting to N bytes" },
#if defined(HAVE_ASM_X86_CLDEMOTE)
	{ NULL,	"cache-cldemote",	"cache line demote (x86 only)" },
#endif
#if defined(HAVE_ASM_X86_CLFLUSHOPT)
	{ NULL, "cache-clflushopt",	"optimized cache line flush (x86 only)" },
#endif
	{ NULL, "cache-enable-all",	"enable all cache options (fence,flush,sfence,etc..)" },
	{ NULL,	"cache-fence",		"serialize stores" },
#if defined(HAVE_ASM_X86_CLFLUSH)
	{ NULL,	"cache-flush",		"flush cache after every memory write (x86 only)" },
#endif
	{ NULL,	"cache-level N",	"only exercise specified cache" },
	{ NULL, "cache-no-affinity",	"do not change CPU affinity" },
	{ NULL,	"cache-ops N",	 	"stop after N cache bogo operations" },
	{ NULL,	"cache-prefetch",	"prefetch for memory reads/writes" },
#if defined(HAVE_ASM_X86_PREFETCHW)
	{ NULL,	"cache-prefetchw",	"prefetch for memory write" },
#endif
#if defined(HAVE_BUILTIN_SFENCE)
	{ NULL,	"cache-sfence",		"serialize stores with sfence" },
#endif
	{ NULL,	"cache-ways N",		"only fill specified number of cache ways" },
	{ NULL, "cache-clwb",		"cache line writeback (x86 only)" },
	{ NULL,	NULL,			NULL }
};

static const stress_opt_t opts[] = {
	{ OPT_cache_cldemote,    "cache-cldemote",    TYPE_ID_BOOL, 0, 1, NULL },
	{ OPT_cache_clflushopt,  "cache-cflushopt",   TYPE_ID_BOOL, 0, 1, NULL },
	{ OPT_cache_enable_all,  "cache-enable-all",  TYPE_ID_BOOL, 0, 1, NULL },
	{ OPT_cache_fence,       "cache-fence",       TYPE_ID_BOOL, 0, 1, NULL },
	{ OPT_cache_flush,	 "cache-flush",       TYPE_ID_BOOL, 0, 1, NULL },
	{ OPT_cache_no_affinity, "cache-no-affinity", TYPE_ID_BOOL, 0, 1, NULL },
	{ OPT_cache_prefetch,    "cache-prefetch",    TYPE_ID_BOOL, 0, 1, NULL },
	{ OPT_cache_prefetchw,   "cache-prefetchw",   TYPE_ID_BOOL, 0, 1, NULL },
	{ OPT_cache_sfence,      "cache-sfence",      TYPE_ID_BOOL, 0, 1, NULL },
	{ OPT_cache_clwb,        "cache-clb",         TYPE_ID_BOOL, 0, 1, NULL },
	END_OPT,
};

#if defined(HAVE_BUILTIN_SFENCE)
#define SHIM_SFENCE()		__builtin_ia32_sfence()
#else
#define SHIM_SFENCE()
#endif

#if defined(HAVE_ASM_X86_CLFLUSH)
#define SHIM_CLFLUSH(p)		stress_asm_x86_clflush(p)
#else
#define SHIM_CLFLUSH(p)
#endif

#if defined(HAVE_ASM_X86_CLFLUSHOPT)
#define SHIM_CLFLUSHOPT(p)	stress_asm_x86_clflushopt(p)
#else
#define SHIM_CLFLUSHOPT(p)
#endif

#if defined(HAVE_ASM_X86_CLDEMOTE)
#define SHIM_CLDEMOTE(p)	stress_asm_x86_cldemote(p)
#else
#define SHIM_CLDEMOTE(p)
#endif

#if defined(HAVE_ASM_X86_CLWB)
#define SHIM_CLWB(p)		stress_asm_x86_clwb(p)
#else
#define SHIM_CLWB(p)
#endif

#if defined(HAVE_ASM_X86_PREFETCHW)
#define SHIM_PREFETCHW(p)	stress_asm_x86_prefetchw(p)
#else
#define SHIM_PREFETCHW(p)
#endif

/*
 * The compiler optimises out the unused cache flush and mfence calls
 */
#define CACHE_WRITE_MOD(flags)						\
	for (j = 0; LIKELY(j < buffer_size); j++) {			\
		i += inc;						\
		i = (i >= buffer_size) ? i - buffer_size : i;		\
		k += 33;						\
		k = (k >= buffer_size) ? k - buffer_size : k;		\
									\
		if ((flags) & CACHE_FLAGS_PREFETCH) {			\
			shim_builtin_prefetch(&buffer[i + 1], 1, 3);	\
		}							\
		if ((flags) & CACHE_FLAGS_CLDEMOTE) {			\
			SHIM_CLDEMOTE(&buffer[i]);			\
		}							\
		if ((flags) & CACHE_FLAGS_CLFLUSHOPT) {			\
			SHIM_CLFLUSHOPT(&buffer[i]);			\
		}							\
		buffer[i] += buffer[k] + r;				\
		if ((flags) & CACHE_FLAGS_CLWB) {			\
			SHIM_CLWB(&buffer[i]);				\
		}							\
		if ((flags) & CACHE_FLAGS_CLFLUSH) {			\
			SHIM_CLFLUSH(&buffer[i]);			\
		}							\
		if ((flags) & CACHE_FLAGS_FENCE) {			\
			shim_mfence();					\
		}							\
		if ((flags) & CACHE_FLAGS_SFENCE) {			\
			SHIM_SFENCE();					\
		}							\
		if ((flags) & CACHE_FLAGS_PREFETCHW) {			\
			SHIM_PREFETCHW(&buffer[i]);			\
		}							\
		if (UNLIKELY(!stress_continue_flag()))			\
			break;						\
	}

#define CACHE_WRITE_USE_MOD(x)						\
static void OPTIMIZE3 stress_cache_write_mod_ ## x(			\
	stress_args_t *args,						\
	const uint64_t inc,						\
	const uint64_t r, 						\
	uint64_t *pi,							\
	uint64_t *pk,							\
	stress_metrics_t *metrics)					\
{									\
	uint64_t i = *pi, j, k = *pk;				\
	uint8_t *const buffer = g_shared->mem_cache.buffer;		\
	const uint64_t buffer_size = g_shared->mem_cache.size;		\
	double t;							\
									\
	t = stress_time_now();						\
	CACHE_WRITE_MOD(x);						\
	metrics->duration += stress_time_now() - t;			\
	metrics->count += (double)buffer_size;				\
	stress_bogo_add(args, j >> 10);					\
									\
	*pi = i;							\
	*pk = k;							\
}									\

CACHE_WRITE_USE_MOD(0x00)
CACHE_WRITE_USE_MOD(0x01)
CACHE_WRITE_USE_MOD(0x02)
CACHE_WRITE_USE_MOD(0x03)
CACHE_WRITE_USE_MOD(0x04)
CACHE_WRITE_USE_MOD(0x05)
CACHE_WRITE_USE_MOD(0x06)
CACHE_WRITE_USE_MOD(0x07)
CACHE_WRITE_USE_MOD(0x08)
CACHE_WRITE_USE_MOD(0x09)
CACHE_WRITE_USE_MOD(0x0a)
CACHE_WRITE_USE_MOD(0x0b)
CACHE_WRITE_USE_MOD(0x0c)
CACHE_WRITE_USE_MOD(0x0d)
CACHE_WRITE_USE_MOD(0x0e)
CACHE_WRITE_USE_MOD(0x0f)

CACHE_WRITE_USE_MOD(0x10)
CACHE_WRITE_USE_MOD(0x11)
CACHE_WRITE_USE_MOD(0x12)
CACHE_WRITE_USE_MOD(0x13)
CACHE_WRITE_USE_MOD(0x14)
CACHE_WRITE_USE_MOD(0x15)
CACHE_WRITE_USE_MOD(0x16)
CACHE_WRITE_USE_MOD(0x17)
CACHE_WRITE_USE_MOD(0x18)
CACHE_WRITE_USE_MOD(0x19)
CACHE_WRITE_USE_MOD(0x1a)
CACHE_WRITE_USE_MOD(0x1b)
CACHE_WRITE_USE_MOD(0x1c)
CACHE_WRITE_USE_MOD(0x1d)
CACHE_WRITE_USE_MOD(0x1e)
CACHE_WRITE_USE_MOD(0x1f)

CACHE_WRITE_USE_MOD(0x20)
CACHE_WRITE_USE_MOD(0x21)
CACHE_WRITE_USE_MOD(0x22)
CACHE_WRITE_USE_MOD(0x23)
CACHE_WRITE_USE_MOD(0x24)
CACHE_WRITE_USE_MOD(0x25)
CACHE_WRITE_USE_MOD(0x26)
CACHE_WRITE_USE_MOD(0x27)
CACHE_WRITE_USE_MOD(0x28)
CACHE_WRITE_USE_MOD(0x29)
CACHE_WRITE_USE_MOD(0x2a)
CACHE_WRITE_USE_MOD(0x2b)
CACHE_WRITE_USE_MOD(0x2c)
CACHE_WRITE_USE_MOD(0x2d)
CACHE_WRITE_USE_MOD(0x2e)
CACHE_WRITE_USE_MOD(0x2f)

CACHE_WRITE_USE_MOD(0x30)
CACHE_WRITE_USE_MOD(0x31)
CACHE_WRITE_USE_MOD(0x32)
CACHE_WRITE_USE_MOD(0x33)
CACHE_WRITE_USE_MOD(0x34)
CACHE_WRITE_USE_MOD(0x35)
CACHE_WRITE_USE_MOD(0x36)
CACHE_WRITE_USE_MOD(0x37)
CACHE_WRITE_USE_MOD(0x38)
CACHE_WRITE_USE_MOD(0x39)
CACHE_WRITE_USE_MOD(0x3a)
CACHE_WRITE_USE_MOD(0x3b)
CACHE_WRITE_USE_MOD(0x3c)
CACHE_WRITE_USE_MOD(0x3d)
CACHE_WRITE_USE_MOD(0x3e)
CACHE_WRITE_USE_MOD(0x3f)

CACHE_WRITE_USE_MOD(0x40)
CACHE_WRITE_USE_MOD(0x41)
CACHE_WRITE_USE_MOD(0x42)
CACHE_WRITE_USE_MOD(0x43)
CACHE_WRITE_USE_MOD(0x44)
CACHE_WRITE_USE_MOD(0x45)
CACHE_WRITE_USE_MOD(0x46)
CACHE_WRITE_USE_MOD(0x47)
CACHE_WRITE_USE_MOD(0x48)
CACHE_WRITE_USE_MOD(0x49)
CACHE_WRITE_USE_MOD(0x4a)
CACHE_WRITE_USE_MOD(0x4b)
CACHE_WRITE_USE_MOD(0x4c)
CACHE_WRITE_USE_MOD(0x4d)
CACHE_WRITE_USE_MOD(0x4e)
CACHE_WRITE_USE_MOD(0x4f)

CACHE_WRITE_USE_MOD(0x50)
CACHE_WRITE_USE_MOD(0x51)
CACHE_WRITE_USE_MOD(0x52)
CACHE_WRITE_USE_MOD(0x53)
CACHE_WRITE_USE_MOD(0x54)
CACHE_WRITE_USE_MOD(0x55)
CACHE_WRITE_USE_MOD(0x56)
CACHE_WRITE_USE_MOD(0x57)
CACHE_WRITE_USE_MOD(0x58)
CACHE_WRITE_USE_MOD(0x59)
CACHE_WRITE_USE_MOD(0x5a)
CACHE_WRITE_USE_MOD(0x5b)
CACHE_WRITE_USE_MOD(0x5c)
CACHE_WRITE_USE_MOD(0x5d)
CACHE_WRITE_USE_MOD(0x5e)
CACHE_WRITE_USE_MOD(0x5f)

CACHE_WRITE_USE_MOD(0x60)
CACHE_WRITE_USE_MOD(0x61)
CACHE_WRITE_USE_MOD(0x62)
CACHE_WRITE_USE_MOD(0x63)
CACHE_WRITE_USE_MOD(0x64)
CACHE_WRITE_USE_MOD(0x65)
CACHE_WRITE_USE_MOD(0x66)
CACHE_WRITE_USE_MOD(0x67)
CACHE_WRITE_USE_MOD(0x68)
CACHE_WRITE_USE_MOD(0x69)
CACHE_WRITE_USE_MOD(0x6a)
CACHE_WRITE_USE_MOD(0x6b)
CACHE_WRITE_USE_MOD(0x6c)
CACHE_WRITE_USE_MOD(0x6d)
CACHE_WRITE_USE_MOD(0x6e)
CACHE_WRITE_USE_MOD(0x6f)

CACHE_WRITE_USE_MOD(0x70)
CACHE_WRITE_USE_MOD(0x71)
CACHE_WRITE_USE_MOD(0x72)
CACHE_WRITE_USE_MOD(0x73)
CACHE_WRITE_USE_MOD(0x74)
CACHE_WRITE_USE_MOD(0x75)
CACHE_WRITE_USE_MOD(0x76)
CACHE_WRITE_USE_MOD(0x77)
CACHE_WRITE_USE_MOD(0x78)
CACHE_WRITE_USE_MOD(0x79)
CACHE_WRITE_USE_MOD(0x7a)
CACHE_WRITE_USE_MOD(0x7b)
CACHE_WRITE_USE_MOD(0x7c)
CACHE_WRITE_USE_MOD(0x7d)
CACHE_WRITE_USE_MOD(0x7e)
CACHE_WRITE_USE_MOD(0x7f)

CACHE_WRITE_USE_MOD(0x80)
CACHE_WRITE_USE_MOD(0x81)
CACHE_WRITE_USE_MOD(0x82)
CACHE_WRITE_USE_MOD(0x83)
CACHE_WRITE_USE_MOD(0x84)
CACHE_WRITE_USE_MOD(0x85)
CACHE_WRITE_USE_MOD(0x86)
CACHE_WRITE_USE_MOD(0x87)
CACHE_WRITE_USE_MOD(0x88)
CACHE_WRITE_USE_MOD(0x89)
CACHE_WRITE_USE_MOD(0x8a)
CACHE_WRITE_USE_MOD(0x8b)
CACHE_WRITE_USE_MOD(0x8c)
CACHE_WRITE_USE_MOD(0x8d)
CACHE_WRITE_USE_MOD(0x8e)
CACHE_WRITE_USE_MOD(0x8f)

CACHE_WRITE_USE_MOD(0x90)
CACHE_WRITE_USE_MOD(0x91)
CACHE_WRITE_USE_MOD(0x92)
CACHE_WRITE_USE_MOD(0x93)
CACHE_WRITE_USE_MOD(0x94)
CACHE_WRITE_USE_MOD(0x95)
CACHE_WRITE_USE_MOD(0x96)
CACHE_WRITE_USE_MOD(0x97)
CACHE_WRITE_USE_MOD(0x98)
CACHE_WRITE_USE_MOD(0x99)
CACHE_WRITE_USE_MOD(0x9a)
CACHE_WRITE_USE_MOD(0x9b)
CACHE_WRITE_USE_MOD(0x9c)
CACHE_WRITE_USE_MOD(0x9d)
CACHE_WRITE_USE_MOD(0x9e)
CACHE_WRITE_USE_MOD(0x9f)

CACHE_WRITE_USE_MOD(0xa0)
CACHE_WRITE_USE_MOD(0xa1)
CACHE_WRITE_USE_MOD(0xa2)
CACHE_WRITE_USE_MOD(0xa3)
CACHE_WRITE_USE_MOD(0xa4)
CACHE_WRITE_USE_MOD(0xa5)
CACHE_WRITE_USE_MOD(0xa6)
CACHE_WRITE_USE_MOD(0xa7)
CACHE_WRITE_USE_MOD(0xa8)
CACHE_WRITE_USE_MOD(0xa9)
CACHE_WRITE_USE_MOD(0xaa)
CACHE_WRITE_USE_MOD(0xab)
CACHE_WRITE_USE_MOD(0xac)
CACHE_WRITE_USE_MOD(0xad)
CACHE_WRITE_USE_MOD(0xae)
CACHE_WRITE_USE_MOD(0xaf)

CACHE_WRITE_USE_MOD(0xb0)
CACHE_WRITE_USE_MOD(0xb1)
CACHE_WRITE_USE_MOD(0xb2)
CACHE_WRITE_USE_MOD(0xb3)
CACHE_WRITE_USE_MOD(0xb4)
CACHE_WRITE_USE_MOD(0xb5)
CACHE_WRITE_USE_MOD(0xb6)
CACHE_WRITE_USE_MOD(0xb7)
CACHE_WRITE_USE_MOD(0xb8)
CACHE_WRITE_USE_MOD(0xb9)
CACHE_WRITE_USE_MOD(0xba)
CACHE_WRITE_USE_MOD(0xbb)
CACHE_WRITE_USE_MOD(0xbc)
CACHE_WRITE_USE_MOD(0xbd)
CACHE_WRITE_USE_MOD(0xbe)
CACHE_WRITE_USE_MOD(0xbf)

CACHE_WRITE_USE_MOD(0xc0)
CACHE_WRITE_USE_MOD(0xc1)
CACHE_WRITE_USE_MOD(0xc2)
CACHE_WRITE_USE_MOD(0xc3)
CACHE_WRITE_USE_MOD(0xc4)
CACHE_WRITE_USE_MOD(0xc5)
CACHE_WRITE_USE_MOD(0xc6)
CACHE_WRITE_USE_MOD(0xc7)
CACHE_WRITE_USE_MOD(0xc8)
CACHE_WRITE_USE_MOD(0xc9)
CACHE_WRITE_USE_MOD(0xca)
CACHE_WRITE_USE_MOD(0xcb)
CACHE_WRITE_USE_MOD(0xcc)
CACHE_WRITE_USE_MOD(0xcd)
CACHE_WRITE_USE_MOD(0xce)
CACHE_WRITE_USE_MOD(0xcf)

CACHE_WRITE_USE_MOD(0xd0)
CACHE_WRITE_USE_MOD(0xd1)
CACHE_WRITE_USE_MOD(0xd2)
CACHE_WRITE_USE_MOD(0xd3)
CACHE_WRITE_USE_MOD(0xd4)
CACHE_WRITE_USE_MOD(0xd5)
CACHE_WRITE_USE_MOD(0xd6)
CACHE_WRITE_USE_MOD(0xd7)
CACHE_WRITE_USE_MOD(0xd8)
CACHE_WRITE_USE_MOD(0xd9)
CACHE_WRITE_USE_MOD(0xda)
CACHE_WRITE_USE_MOD(0xdb)
CACHE_WRITE_USE_MOD(0xdc)
CACHE_WRITE_USE_MOD(0xdd)
CACHE_WRITE_USE_MOD(0xde)
CACHE_WRITE_USE_MOD(0xdf)

CACHE_WRITE_USE_MOD(0xe0)
CACHE_WRITE_USE_MOD(0xe1)
CACHE_WRITE_USE_MOD(0xe2)
CACHE_WRITE_USE_MOD(0xe3)
CACHE_WRITE_USE_MOD(0xe4)
CACHE_WRITE_USE_MOD(0xe5)
CACHE_WRITE_USE_MOD(0xe6)
CACHE_WRITE_USE_MOD(0xe7)
CACHE_WRITE_USE_MOD(0xe8)
CACHE_WRITE_USE_MOD(0xe9)
CACHE_WRITE_USE_MOD(0xea)
CACHE_WRITE_USE_MOD(0xeb)
CACHE_WRITE_USE_MOD(0xec)
CACHE_WRITE_USE_MOD(0xed)
CACHE_WRITE_USE_MOD(0xee)
CACHE_WRITE_USE_MOD(0xef)

CACHE_WRITE_USE_MOD(0xf0)
CACHE_WRITE_USE_MOD(0xf1)
CACHE_WRITE_USE_MOD(0xf2)
CACHE_WRITE_USE_MOD(0xf3)
CACHE_WRITE_USE_MOD(0xf4)
CACHE_WRITE_USE_MOD(0xf5)
CACHE_WRITE_USE_MOD(0xf6)
CACHE_WRITE_USE_MOD(0xf7)
CACHE_WRITE_USE_MOD(0xf8)
CACHE_WRITE_USE_MOD(0xf9)
CACHE_WRITE_USE_MOD(0xfa)
CACHE_WRITE_USE_MOD(0xfb)
CACHE_WRITE_USE_MOD(0xfc)
CACHE_WRITE_USE_MOD(0xfd)
CACHE_WRITE_USE_MOD(0xfe)
CACHE_WRITE_USE_MOD(0xff)

static const cache_mixed_ops_func_t cache_mixed_ops_funcs[] = {
	stress_cache_write_mod_0x00,
	stress_cache_write_mod_0x01,
	stress_cache_write_mod_0x02,
	stress_cache_write_mod_0x03,
	stress_cache_write_mod_0x04,
	stress_cache_write_mod_0x05,
	stress_cache_write_mod_0x06,
	stress_cache_write_mod_0x07,
	stress_cache_write_mod_0x08,
	stress_cache_write_mod_0x09,
	stress_cache_write_mod_0x0a,
	stress_cache_write_mod_0x0b,
	stress_cache_write_mod_0x0c,
	stress_cache_write_mod_0x0d,
	stress_cache_write_mod_0x0e,
	stress_cache_write_mod_0x0f,

	stress_cache_write_mod_0x10,
	stress_cache_write_mod_0x11,
	stress_cache_write_mod_0x12,
	stress_cache_write_mod_0x13,
	stress_cache_write_mod_0x14,
	stress_cache_write_mod_0x15,
	stress_cache_write_mod_0x16,
	stress_cache_write_mod_0x17,
	stress_cache_write_mod_0x18,
	stress_cache_write_mod_0x19,
	stress_cache_write_mod_0x1a,
	stress_cache_write_mod_0x1b,
	stress_cache_write_mod_0x1c,
	stress_cache_write_mod_0x1d,
	stress_cache_write_mod_0x1e,
	stress_cache_write_mod_0x1f,

	stress_cache_write_mod_0x20,
	stress_cache_write_mod_0x21,
	stress_cache_write_mod_0x22,
	stress_cache_write_mod_0x23,
	stress_cache_write_mod_0x24,
	stress_cache_write_mod_0x25,
	stress_cache_write_mod_0x26,
	stress_cache_write_mod_0x27,
	stress_cache_write_mod_0x28,
	stress_cache_write_mod_0x29,
	stress_cache_write_mod_0x2a,
	stress_cache_write_mod_0x2b,
	stress_cache_write_mod_0x2c,
	stress_cache_write_mod_0x2d,
	stress_cache_write_mod_0x2e,
	stress_cache_write_mod_0x2f,

	stress_cache_write_mod_0x30,
	stress_cache_write_mod_0x31,
	stress_cache_write_mod_0x32,
	stress_cache_write_mod_0x33,
	stress_cache_write_mod_0x34,
	stress_cache_write_mod_0x35,
	stress_cache_write_mod_0x36,
	stress_cache_write_mod_0x37,
	stress_cache_write_mod_0x38,
	stress_cache_write_mod_0x39,
	stress_cache_write_mod_0x3a,
	stress_cache_write_mod_0x3b,
	stress_cache_write_mod_0x3c,
	stress_cache_write_mod_0x3d,
	stress_cache_write_mod_0x3e,
	stress_cache_write_mod_0x3f,

	stress_cache_write_mod_0x40,
	stress_cache_write_mod_0x41,
	stress_cache_write_mod_0x42,
	stress_cache_write_mod_0x43,
	stress_cache_write_mod_0x44,
	stress_cache_write_mod_0x45,
	stress_cache_write_mod_0x46,
	stress_cache_write_mod_0x47,
	stress_cache_write_mod_0x48,
	stress_cache_write_mod_0x49,
	stress_cache_write_mod_0x4a,
	stress_cache_write_mod_0x4b,
	stress_cache_write_mod_0x4c,
	stress_cache_write_mod_0x4d,
	stress_cache_write_mod_0x4e,
	stress_cache_write_mod_0x4f,

	stress_cache_write_mod_0x50,
	stress_cache_write_mod_0x51,
	stress_cache_write_mod_0x52,
	stress_cache_write_mod_0x53,
	stress_cache_write_mod_0x54,
	stress_cache_write_mod_0x55,
	stress_cache_write_mod_0x56,
	stress_cache_write_mod_0x57,
	stress_cache_write_mod_0x58,
	stress_cache_write_mod_0x59,
	stress_cache_write_mod_0x5a,
	stress_cache_write_mod_0x5b,
	stress_cache_write_mod_0x5c,
	stress_cache_write_mod_0x5d,
	stress_cache_write_mod_0x5e,
	stress_cache_write_mod_0x5f,

	stress_cache_write_mod_0x60,
	stress_cache_write_mod_0x61,
	stress_cache_write_mod_0x62,
	stress_cache_write_mod_0x63,
	stress_cache_write_mod_0x64,
	stress_cache_write_mod_0x65,
	stress_cache_write_mod_0x66,
	stress_cache_write_mod_0x67,
	stress_cache_write_mod_0x68,
	stress_cache_write_mod_0x69,
	stress_cache_write_mod_0x6a,
	stress_cache_write_mod_0x6b,
	stress_cache_write_mod_0x6c,
	stress_cache_write_mod_0x6d,
	stress_cache_write_mod_0x6e,
	stress_cache_write_mod_0x6f,

	stress_cache_write_mod_0x70,
	stress_cache_write_mod_0x71,
	stress_cache_write_mod_0x72,
	stress_cache_write_mod_0x73,
	stress_cache_write_mod_0x74,
	stress_cache_write_mod_0x75,
	stress_cache_write_mod_0x76,
	stress_cache_write_mod_0x77,
	stress_cache_write_mod_0x78,
	stress_cache_write_mod_0x79,
	stress_cache_write_mod_0x7a,
	stress_cache_write_mod_0x7b,
	stress_cache_write_mod_0x7c,
	stress_cache_write_mod_0x7d,
	stress_cache_write_mod_0x7e,
	stress_cache_write_mod_0x7f,

	stress_cache_write_mod_0x80,
	stress_cache_write_mod_0x81,
	stress_cache_write_mod_0x82,
	stress_cache_write_mod_0x83,
	stress_cache_write_mod_0x84,
	stress_cache_write_mod_0x85,
	stress_cache_write_mod_0x86,
	stress_cache_write_mod_0x87,
	stress_cache_write_mod_0x88,
	stress_cache_write_mod_0x89,
	stress_cache_write_mod_0x8a,
	stress_cache_write_mod_0x8b,
	stress_cache_write_mod_0x8c,
	stress_cache_write_mod_0x8d,
	stress_cache_write_mod_0x8e,
	stress_cache_write_mod_0x8f,

	stress_cache_write_mod_0x90,
	stress_cache_write_mod_0x91,
	stress_cache_write_mod_0x92,
	stress_cache_write_mod_0x93,
	stress_cache_write_mod_0x94,
	stress_cache_write_mod_0x95,
	stress_cache_write_mod_0x96,
	stress_cache_write_mod_0x97,
	stress_cache_write_mod_0x98,
	stress_cache_write_mod_0x99,
	stress_cache_write_mod_0x9a,
	stress_cache_write_mod_0x9b,
	stress_cache_write_mod_0x9c,
	stress_cache_write_mod_0x9d,
	stress_cache_write_mod_0x9e,
	stress_cache_write_mod_0x9f,

	stress_cache_write_mod_0xa0,
	stress_cache_write_mod_0xa1,
	stress_cache_write_mod_0xa2,
	stress_cache_write_mod_0xa3,
	stress_cache_write_mod_0xa4,
	stress_cache_write_mod_0xa5,
	stress_cache_write_mod_0xa6,
	stress_cache_write_mod_0xa7,
	stress_cache_write_mod_0xa8,
	stress_cache_write_mod_0xa9,
	stress_cache_write_mod_0xaa,
	stress_cache_write_mod_0xab,
	stress_cache_write_mod_0xac,
	stress_cache_write_mod_0xad,
	stress_cache_write_mod_0xae,
	stress_cache_write_mod_0xaf,

	stress_cache_write_mod_0xb0,
	stress_cache_write_mod_0xb1,
	stress_cache_write_mod_0xb2,
	stress_cache_write_mod_0xb3,
	stress_cache_write_mod_0xb4,
	stress_cache_write_mod_0xb5,
	stress_cache_write_mod_0xb6,
	stress_cache_write_mod_0xb7,
	stress_cache_write_mod_0xb8,
	stress_cache_write_mod_0xb9,
	stress_cache_write_mod_0xba,
	stress_cache_write_mod_0xbb,
	stress_cache_write_mod_0xbc,
	stress_cache_write_mod_0xbd,
	stress_cache_write_mod_0xbe,
	stress_cache_write_mod_0xbf,

	stress_cache_write_mod_0xc0,
	stress_cache_write_mod_0xc1,
	stress_cache_write_mod_0xc2,
	stress_cache_write_mod_0xc3,
	stress_cache_write_mod_0xc4,
	stress_cache_write_mod_0xc5,
	stress_cache_write_mod_0xc6,
	stress_cache_write_mod_0xc7,
	stress_cache_write_mod_0xc8,
	stress_cache_write_mod_0xc9,
	stress_cache_write_mod_0xca,
	stress_cache_write_mod_0xcb,
	stress_cache_write_mod_0xcc,
	stress_cache_write_mod_0xcd,
	stress_cache_write_mod_0xce,
	stress_cache_write_mod_0xcf,

	stress_cache_write_mod_0xd0,
	stress_cache_write_mod_0xd1,
	stress_cache_write_mod_0xd2,
	stress_cache_write_mod_0xd3,
	stress_cache_write_mod_0xd4,
	stress_cache_write_mod_0xd5,
	stress_cache_write_mod_0xd6,
	stress_cache_write_mod_0xd7,
	stress_cache_write_mod_0xd8,
	stress_cache_write_mod_0xd9,
	stress_cache_write_mod_0xda,
	stress_cache_write_mod_0xdb,
	stress_cache_write_mod_0xdc,
	stress_cache_write_mod_0xdd,
	stress_cache_write_mod_0xde,
	stress_cache_write_mod_0xdf,

	stress_cache_write_mod_0xe0,
	stress_cache_write_mod_0xe1,
	stress_cache_write_mod_0xe2,
	stress_cache_write_mod_0xe3,
	stress_cache_write_mod_0xe4,
	stress_cache_write_mod_0xe5,
	stress_cache_write_mod_0xe6,
	stress_cache_write_mod_0xe7,
	stress_cache_write_mod_0xe8,
	stress_cache_write_mod_0xe9,
	stress_cache_write_mod_0xea,
	stress_cache_write_mod_0xeb,
	stress_cache_write_mod_0xec,
	stress_cache_write_mod_0xed,
	stress_cache_write_mod_0xee,
	stress_cache_write_mod_0xef,

	stress_cache_write_mod_0xf0,
	stress_cache_write_mod_0xf1,
	stress_cache_write_mod_0xf2,
	stress_cache_write_mod_0xf3,
	stress_cache_write_mod_0xf4,
	stress_cache_write_mod_0xf5,
	stress_cache_write_mod_0xf6,
	stress_cache_write_mod_0xf7,
	stress_cache_write_mod_0xf8,
	stress_cache_write_mod_0xf9,
	stress_cache_write_mod_0xfa,
	stress_cache_write_mod_0xfb,
	stress_cache_write_mod_0xfc,
	stress_cache_write_mod_0xfd,
	stress_cache_write_mod_0xfe,
	stress_cache_write_mod_0xff,
};

typedef void (*cache_read_func_t)(uint64_t *pi, uint64_t *pk, uint32_t *ptotal);

static void NORETURN MLOCKED_TEXT stress_cache_sighandler(int signum)
{
	(void)signum;

	caught_signum = signum;

	siglongjmp(jmp_env, 1);         /* Ugly, bounce back */
}

static void NORETURN MLOCKED_TEXT stress_cache_sigillhandler(int signum)
{
	uint32_t mask = masked_flags;

	caught_signum = signum;

	/* bit set? then disable it */
	if (mask) {
		size_t i = 0;
		/* Find top bit that is set, work from most modern flag to least */
		while (mask >>= 1)
			i++;
		mask = 1U << i;

		for (i = 0; i < SIZEOF_ARRAY(mask_flag_info); i++) {
			if (mask_flag_info[i].flag & mask) {
				masked_flags &= ~mask;
				disabled_flags |= mask;
				break;
			}
		}
	}

	siglongjmp(jmp_env, 1);         /* Ugly, bounce back */
}

/*
 *  exercise invalid cache flush ops
 */
static void stress_cache_flush(void *addr, void *bad_addr, int size)
{
	(void)shim_cacheflush((char *)addr, size, 0);
	(void)shim_cacheflush((char *)addr, size, ~0);
	(void)shim_cacheflush((char *)addr, 0, SHIM_DCACHE);
	(void)shim_cacheflush((char *)addr, 1, SHIM_DCACHE);
	(void)shim_cacheflush((char *)addr, -1, SHIM_DCACHE);
#if defined(HAVE_BUILTIN___CLEAR_CACHE)
	__builtin___clear_cache((char *)addr, (char *)addr);
#else
	UNEXPECTED
#endif
	(void)shim_cacheflush((char *)bad_addr, size, SHIM_ICACHE);
	(void)shim_cacheflush((char *)bad_addr, size, SHIM_DCACHE);
	(void)shim_cacheflush((char *)bad_addr, size, SHIM_ICACHE | SHIM_DCACHE);
#if defined(HAVE_BUILTIN___CLEAR_CACHE)
	__builtin___clear_cache((char *)addr, (char *)((uint8_t *)addr - 1));
#else
	UNEXPECTED
#endif
}

static void stress_cache_read(
	stress_args_t *args,
	const uint8_t *buffer,
	const uint64_t buffer_size,
	const uint64_t inc,
	uint64_t *i_ptr,
	uint64_t *k_ptr,
	stress_metrics_t *metrics_read)
{
	uint64_t i = *i_ptr;
	uint64_t j;
	uint64_t k = *k_ptr;
	uint32_t total = 0;
	double t;

	t = stress_time_now();
	for (j = 0; j < buffer_size; j++) {
		i += inc;
		i = (i >= buffer_size) ? i - buffer_size : i;
		k += 33;
		k = (k >= buffer_size) ? k - buffer_size : k;
		total += buffer[i] + buffer[k];
		if (UNLIKELY(!stress_continue_flag()))
			break;
	}
	metrics_read->duration += stress_time_now() - t;
	metrics_read->count += (double)(j + j); /* two reads per loop */
	stress_bogo_add(args, j >> 10);

	*i_ptr = i;
	*k_ptr = k;

	stress_uint32_put(total);
}

static void stress_cache_write(
	stress_args_t *args,
	uint8_t *buffer,
	const uint64_t buffer_size,
	const uint64_t inc,
	uint64_t *i_ptr,
	uint64_t *k_ptr,
	stress_metrics_t *metrics_write)
{
	uint64_t i = *i_ptr;
	uint64_t j;
	uint64_t k = *k_ptr;
	uint32_t total = 0;
	double t;

	t = stress_time_now();
	for (j = 0; j < buffer_size; j++) {
		const uint8_t v = j & 0xff;

		i += inc;
		i = (i >= buffer_size) ? i - buffer_size : i;
		k += 33;
		k = (k >= buffer_size) ? k - buffer_size : k;
		buffer[i] = v;
		buffer[k] = v;
		if (UNLIKELY(!stress_continue_flag()))
			break;
	}
	metrics_write->duration += stress_time_now() - t;
	metrics_write->count += (double)(j + j); /* 2 writes per loop */
	stress_bogo_add(args, j >> 10);

	*i_ptr = i;
	*k_ptr = k;

	stress_uint32_put(total);
}

static void stress_cached_str_flags(char *buf, size_t buflen, const uint32_t flags)
{
	size_t i;

	(void)shim_memset(buf, 0, buflen);
	for (i = 0; i < SIZEOF_ARRAY(mask_flag_info); i++) {
		if (flags & mask_flag_info[i].flag) {
			(void)shim_strlcat(buf, " ", buflen);
			(void)shim_strlcat(buf, mask_flag_info[i].name, buflen);
		}
	}
}

static void stress_cache_show_flags(
	stress_args_t *args,
	const uint32_t used_flags,
	const uint32_t ignored_flags)
{
	char buf[256];

	stress_cached_str_flags(buf, sizeof(buf), used_flags);
	if (!*buf)
		(void)shim_strscpy(buf, " none", sizeof(buf));
	pr_inf("%s: cache flags used:%s\n", args->name, buf);
	(void)shim_memset(buf, 0, sizeof(buf));

	stress_cached_str_flags(buf, sizeof(buf), ignored_flags);
	if (*buf)
		pr_inf("%s: unavailable unused cache flags:%s\n", args->name, buf);
}

static void stress_cache_bzero(uint8_t *buffer, const uint64_t buffer_size)
{
#if defined(STRESS_ARCH_RISCV) &&	\
    defined(HAVE_ASM_RISCV_CBO_ZERO) &&	\
    defined(__NR_riscv_hwprobe) && \
    defined(RISCV_HWPROBE_EXT_ZICBOZ)
	cpu_set_t cpus;
	struct riscv_hwprobe pair;

	(void)sched_getaffinity(0, sizeof(cpu_set_t), &cpus);

	pair.key = RISCV_HWPROBE_KEY_IMA_EXT_0;

	if (syscall(__NR_riscv_hwprobe, &pair, 1, sizeof(cpu_set_t), &cpus, 0) == 0) {
		if (pair.value & RISCV_HWPROBE_EXT_ZICBOZ) {
			pair.key = RISCV_HWPROBE_KEY_ZICBOZ_BLOCK_SIZE;

			if (syscall(__NR_riscv_hwprobe, &pair, 1,
				    sizeof(cpu_set_t), &cpus, 0) == 0) {
				int block_size = (int)pair.value;
				uint8_t *ptr;
				const uint8_t *buffer_end = buffer + buffer_size;

				for (ptr = buffer; ptr < buffer_end; ptr += block_size) {
					(void)stress_asm_riscv_cbo_zero((char *)ptr);
				}
			}
		}
	}
#else
	(void)buffer;
	(void)buffer_size;
#endif
}

static void stress_get_cache_flags(const char *opt, uint32_t *cache_flags, uint32_t bitmask)
{
	bool flag = 0;

	(void)stress_get_setting(opt, &flag);
	if (flag)
		*cache_flags |= bitmask;
}

/*
 *  stress_cache()
 *	stress cache by psuedo-random memory read/writes and
 *	if possible change CPU affinity to try to cause
 *	poor cache behaviour
 */
static int stress_cache(stress_args_t *args)
{
#if defined(HAVE_SCHED_GETAFFINITY) &&	\
    defined(HAVE_SCHED_SETAFFINITY) &&	\
    defined(HAVE_SCHED_GETCPU)
	cpu_set_t proc_mask;
	NOCLOBBER uint32_t cpu = 0;
	uint32_t *cpus;
	const uint32_t n_cpus = stress_get_usable_cpus(&cpus, true);
	NOCLOBBER bool pinned = false;
#endif
	NOCLOBBER uint32_t cache_flags = 0;
	NOCLOBBER uint32_t cache_flags_mask = CACHE_FLAGS_MASK;
	NOCLOBBER uint32_t ignored_flags = 0;
	NOCLOBBER uint32_t total = 0;
	int ret = EXIT_SUCCESS;
	uint8_t *const buffer = g_shared->mem_cache.buffer;
	const uint64_t buffer_size = g_shared->mem_cache.size;
	uint64_t i = stress_mwc64modn(buffer_size);
	uint64_t k = i + (buffer_size >> 1);
	NOCLOBBER uint64_t r = 0;
	uint64_t inc = (buffer_size >> 2) + 1;
	void *bad_addr;
	size_t j;
	stress_metrics_t metrics[STRESS_CACHE_MAX];

	static char *const metrics_description[] = {
		"cache ops per second",
		"shared cache reads per second",
		"shared cache writes per second",
	};

	stress_zero_metrics(metrics, SIZEOF_ARRAY(metrics));

	caught_signum = -1;
	disabled_flags = 0;

	(void)cache_flags;
	(void)cache_flags_mask;

	if (sigsetjmp(jmp_env, 1)) {
		const char *signame = stress_get_signal_name(caught_signum);

		pr_inf_skip("%s: signal %s (#%d) caught, skipping stressor\n",
			args->name, signame ? signame : "unknown", caught_signum);
		ret = EXIT_NO_RESOURCE;
		goto tidy_cpus;
	}

	if (stress_sighandler(args->name, SIGSEGV, stress_cache_sighandler, NULL) < 0) {
		ret = EXIT_NO_RESOURCE;
		goto tidy_cpus;
	}
#if !defined(STRESS_ARCH_X86)
	if (stress_sighandler(args->name, SIGBUS, stress_cache_sighandler, NULL) < 0) {
		ret = EXIT_NO_RESOURCE;
		goto tidy_cpus;
	}
#endif
	if (stress_sighandler(args->name, SIGILL, stress_cache_sigillhandler, NULL) < 0) {
		ret = EXIT_NO_RESOURCE;
		goto tidy_cpus;
	}

	(void)stress_get_cache_flags("cache-cldemote", &cache_flags, CACHE_FLAGS_CLDEMOTE);
	(void)stress_get_cache_flags("cache-cflushopt", &cache_flags, CACHE_FLAGS_CLFLUSHOPT);
	(void)stress_get_cache_flags("cache-enable-all", &cache_flags, CACHE_FLAGS_MASK);
	(void)stress_get_cache_flags("cache-fence", &cache_flags, CACHE_FLAGS_FENCE);
	(void)stress_get_cache_flags("cache-flush", &cache_flags, CACHE_FLAGS_CLFLUSH);
	(void)stress_get_cache_flags("cache-no-affinity", &cache_flags, CACHE_FLAGS_NOAFF);
	(void)stress_get_cache_flags("cache-prefetch", &cache_flags, CACHE_FLAGS_PREFETCH);
	(void)stress_get_cache_flags("cache-sfence", &cache_flags, CACHE_FLAGS_SFENCE);
	(void)stress_get_cache_flags("cache-clb", &cache_flags, CACHE_FLAGS_CLWB);
	(void)stress_get_cache_flags("cache-prefetchw", &cache_flags, CACHE_FLAGS_PREFETCHW);

	if (args->instance == 0)
		pr_dbg("%s: using cache buffer size of %" PRIu64 "K\n",
			args->name, buffer_size / 1024);

#if defined(HAVE_SCHED_GETAFFINITY) && 	\
    defined(HAVE_SCHED_SETAFFINITY) &&	\
    defined(HAVE_SCHED_GETCPU)
	if (sched_getaffinity(0, sizeof(proc_mask), &proc_mask) < 0)
		pinned = true;
	else
		if (!CPU_COUNT(&proc_mask))
			pinned = true;

	if (pinned) {
		pr_inf("%s: can't get sched affinity, pinning to "
			"CPU %d (instance %" PRIu32 ")\n",
			args->name, sched_getcpu(), pinned);
	}
#else
	UNEXPECTED
#endif

#if !defined(HAVE_BUILTIN_SFENCE)
	if (cache_flags & CACHE_FLAGS_SFENCE)
		ignored_flags |= CACHE_FLAGS_SFENCE;
	cache_flags &= ~CACHE_FLAGS_SFENCE;
	cache_flags_mask &= ~CACHE_FLAGS_SFENCE;
#endif

#if !defined(HAVE_ASM_X86_CLDEMOTE)
	if (cache_flags & CACHE_FLAGS_CLDEMOTE)
		ignored_flags |= CACHE_FLAGS_CLDEMOTE;
	cache_flags &= ~CACHE_FLAGS_CLDEMOTE;
	cache_flags_mask &= ~CACHE_FLAGS_CLDEMOTE;
#endif

#if defined(HAVE_ASM_X86_CLDEMOTE)
	if (!stress_cpu_x86_has_cldemote() && (cache_flags & CACHE_FLAGS_CLDEMOTE)) {
		cache_flags &= ~CACHE_FLAGS_CLDEMOTE;
		cache_flags_mask &= ~CACHE_FLAGS_CLDEMOTE;
		ignored_flags |= CACHE_FLAGS_CLDEMOTE;
	}
#endif

#if !defined(HAVE_ASM_X86_CLFLUSH)
	if (cache_flags & CACHE_FLAGS_CLFLUSH)
		ignored_flags |= CACHE_FLAGS_CLFLUSH;
	cache_flags &= ~CACHE_FLAGS_CLFLUSH;
	cache_flags_mask &= ~CACHE_FLAGS_CLFLUSH;
#endif

#if defined(HAVE_ASM_X86_CLFLUSH)
	if (!stress_cpu_x86_has_clfsh() && (cache_flags & CACHE_FLAGS_CLFLUSH)) {
		cache_flags &= ~CACHE_FLAGS_CLFLUSH;
		cache_flags_mask &= ~CACHE_FLAGS_CLFLUSH;
		ignored_flags |= CACHE_FLAGS_CLFLUSH;
	}
#endif

#if !defined(HAVE_ASM_X86_CLFLUSHOPT)
	if (cache_flags & CACHE_FLAGS_CLFLUSHOPT)
		ignored_flags |= CACHE_FLAGS_CLFLUSHOPT;
	cache_flags &= ~CACHE_FLAGS_CLFLUSHOPT;
	cache_flags_mask &= ~CACHE_FLAGS_CLFLUSHOPT;
#endif

#if defined(HAVE_ASM_X86_CLFLUSHOPT)
	if (!stress_cpu_x86_has_clflushopt() && (cache_flags & CACHE_FLAGS_CLFLUSHOPT)) {
		cache_flags &= ~CACHE_FLAGS_CLFLUSHOPT;
		cache_flags_mask &= ~CACHE_FLAGS_CLFLUSHOPT;
		ignored_flags |= CACHE_FLAGS_CLFLUSHOPT;
	}
#endif

#if !defined(HAVE_ASM_X86_CLWB)
	if (cache_flags & CACHE_FLAGS_CLWB)
		ignored_flags |= CACHE_FLAGS_CLWB;
	cache_flags &= ~CACHE_FLAGS_CLWB;
	cache_flags_mask &= ~CACHE_FLAGS_CLWB;
#endif

#if defined(HAVE_ASM_X86_CLWB)
	if (!stress_cpu_x86_has_clwb() && (cache_flags & CACHE_FLAGS_CLWB)) {
		cache_flags &= ~CACHE_FLAGS_CLWB;
		cache_flags_mask &= ~CACHE_FLAGS_CLWB;
		ignored_flags |= CACHE_FLAGS_CLWB;
	}
#endif
	/*
	 *  map a page then unmap it, then we have an address
	 *  that is known to be not available. If the mapping
	 *  fails we have MAP_FAILED which too is an invalid
	 *  bad address.
	 */
	bad_addr = mmap(NULL, args->page_size, PROT_READ,
		MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	if (bad_addr != MAP_FAILED)
		(void)munmap(bad_addr, args->page_size);

	masked_flags = cache_flags & CACHE_FLAGS_MASK;
	if (args->instance == 0) {
		stress_cache_show_flags(args, masked_flags, ignored_flags);
		if (masked_flags == 0)
			pr_inf("%s: use --cache-enable-all to enable all cache flags for heavier cache stressing\n", args->name);
	}
	(void)shim_memset(buffer, 0, buffer_size);

	do {
		int jmpret;
		uint32_t flags;

		jmpret = sigsetjmp(jmp_env, 1);
		/*
		 *  We return here if we segfault, so
		 *  check if we need to terminate
		 */
		if (jmpret) {
			if (LIKELY(stress_continue(args)))
				goto next;
			break;
		}
		switch (r) {
		case STRESS_CACHE_MIXED_OPS:
			flags = masked_flags ? masked_flags : ((stress_mwc32() & CACHE_FLAGS_MASK) & masked_flags);
			cache_mixed_ops_funcs[flags](args, inc, r, &i, &k, &metrics[STRESS_CACHE_MIXED_OPS]);
			break;
		case STRESS_CACHE_READ:
			stress_cache_read(args, buffer, buffer_size, inc, &i, &k, &metrics[STRESS_CACHE_READ]);
			break;
		case STRESS_CACHE_WRITE:
			stress_cache_write(args, buffer, buffer_size, inc, &i, &k, &metrics[STRESS_CACHE_WRITE]);
			break;
		}
		r++;
		if (r >= STRESS_CACHE_MAX)
			r = 0;
#if defined(HAVE_SCHED_GETAFFINITY) &&	\
    defined(HAVE_SCHED_SETAFFINITY) &&	\
    defined(HAVE_SCHED_GETCPU)
		if ((cache_flags & CACHE_FLAGS_NOAFF) && !pinned) {
			const int current = sched_getcpu();

			if (current < 0) {
				pr_fail("%s: getcpu failed, errno=%d (%s)\n",
					args->name, errno, strerror(errno));
				ret = EXIT_FAILURE;
				goto tidy_cpus;
			}
			cpu = (uint32_t)current;
		} else {
			static uint32_t cpu_idx = 0;

			if (cpus) {
				cpu = cpus[cpu_idx];
				cpu_idx++;
				cpu_idx = (cpu_idx >= n_cpus) ? 0 : cpu_idx;
			} else {
				const int current = sched_getcpu();

				if (current < 0) {
					pr_fail("%s: getcpu failed, errno=%d (%s)\n",
						args->name, errno, strerror(errno));
					ret = EXIT_FAILURE;
					goto tidy_cpus;
				}
				cpu = (uint32_t)current;
			}
		}

		if (!(cache_flags & CACHE_FLAGS_NOAFF) || !pinned) {
			cpu_set_t mask;

			CPU_ZERO(&mask);
			CPU_SET(cpu, &mask);
			(void)sched_setaffinity(0, sizeof(mask), &mask);

			if ((cache_flags & CACHE_FLAGS_NOAFF)) {
				/* Don't continually set the affinity */
				pinned = true;
			}

		}
#else
		UNEXPECTED
#endif
		(void)shim_cacheflush((char *)stress_cache, 8192, SHIM_ICACHE);
		(void)shim_cacheflush((char *)buffer, (int)buffer_size, SHIM_DCACHE);
		stress_cache_bzero(buffer, buffer_size);
#if defined(HAVE_BUILTIN___CLEAR_CACHE)
		__builtin___clear_cache((char *)stress_cache,
					(char *)((char *)stress_cache + 64));
#endif
		/*
		 * Periodically exercise invalid cache ops
		 */
		if ((r & 0x1f) == 0) {
			jmpret = sigsetjmp(jmp_env, 1);
			/*
			 *  We return here if we segfault, so
			 *  first check if we need to terminate
			 */
			if (UNLIKELY(!stress_continue(args)))
				break;

			if (!jmpret)
				stress_cache_flush(buffer, bad_addr, (int)args->page_size);
		}
next:
		/* Move forward a bit */
		i += inc;
		i = (i >= buffer_size) ? i - buffer_size : i;

	} while (stress_continue(args));

	/*
	 *  Hit an illegal instruction, report the disabled flags
	 */
	if ((args->instance == 0) && (disabled_flags)) {
		char buf[1024], *ptr = buf;
		size_t buf_len = sizeof(buf);

		(void)shim_memset(buf, 0, sizeof(buf));
		for (j = 0; j < SIZEOF_ARRAY(mask_flag_info); j++) {
			if (mask_flag_info[j].flag & disabled_flags) {
				const size_t len = strlen(mask_flag_info[j].name);

				(void)shim_strscpy(ptr, " ", buf_len);
				buf_len--;
				ptr++;

				(void)shim_strscpy(ptr, mask_flag_info[j].name, buf_len);
				buf_len -= len;
				ptr += len;
			}
		}
		*ptr = '\0';
		pr_inf("%s: disabled%s due to illegal instruction signal\n", args->name, buf);
	}

	stress_uint32_put(total);

	stress_set_proc_state(args->name, STRESS_STATE_DEINIT);

	for (j = 0; j < SIZEOF_ARRAY(metrics); j++) {
		const double rate = metrics[j].duration > 0.0 ?
			metrics[j].count / metrics[j].duration : 0.0;

		stress_metrics_set(args, j, metrics_description[j],
			rate, STRESS_METRIC_HARMONIC_MEAN);
	}
tidy_cpus:
#if defined(HAVE_SCHED_GETAFFINITY) &&	\
    defined(HAVE_SCHED_SETAFFINITY) &&	\
    defined(HAVE_SCHED_GETCPU)
	stress_free_usable_cpus(&cpus);
#endif

	return ret;
}

const stressor_info_t stress_cache_info = {
	.stressor = stress_cache,
	.cls = CLASS_CPU_CACHE,
	.opts = opts,
	.help = help
};

void BM_STRESS_NG_Cache(benchmark::State& state) {
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;			/* shared memory */
  double g_opt_timeout = 1.0;
  const struct stressor_info *info = &stress_cache_info;
  stats.start = stress_time_now();

  args.stats = &stats;
  args.name = "cache";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  // run
  for (auto _ : state) {
    // Reset the counter for each iteration
    args.ci.counter = 0;

    // Call the stress_cache function
    int rc = stress_cache(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Cache stress test failed");
      return;
    }
  }
}
}

namespace stream_ {
/*
 * Copyright (C) 2016-2021 Canonical, Ltd.
 * Copyright (C) 2022-2025 Colin Ian King.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * This stressor is loosely based on the STREAM Sustainable
 * Memory Bandwidth In High Performance Computers tool.
 *   https://www.cs.virginia.edu/stream/
 *   https://www.cs.virginia.edu/stream/FTP/Code/stream.c
 *
 * This is loosely based on a variant of the STREAM benchmark code,
 * so DO NOT submit results based on this as it is intended to
 * stress memory and compute and NOT intended for STREAM accurate
 * tuned or non-tuned benchmarking whatsoever.  I believe this
 * conforms to section 3a, 3b of the original License.
 *
 */


#define MIN_STREAM_L3_SIZE	(4 * KB)
#define MAX_STREAM_L3_SIZE	(MAX_MEM_LIMIT)
#define DEFAULT_STREAM_L3_SIZE	(4 * MB)

#if defined(HAVE_NT_STORE_DOUBLE)
#define NT_STORE(dst, src)		stress_nt_store_double(&dst, src)
#endif

#define STORE(dst, src)			dst = src

typedef struct {
	const char *name;
	const int advice;
} stress_stream_madvise_info_t;

static const stress_help_t help[] = {
	{ NULL,	"stream N",		"start N workers exercising memory bandwidth" },
	{ NULL,	"stream-index N",	"specify number of indices into the data (0..3)" },
	{ NULL,	"stream-l3-size N",	"specify the L3 cache size of the CPU" },
	{ NULL,	"stream-madvise M",	"specify mmap'd stream buffer madvise advice" },
	{ NULL,	"stream-mlock",		"attempt to mlock pages into memory" },
	{ NULL,	"stream-ops N",		"stop after N bogo stream operations" },
	{ NULL,	NULL,                   NULL }
};

static const stress_stream_madvise_info_t stream_madvise_info[] = {
#if !defined(HAVE_MADVISE)
	/* No MADVISE, default to normal, ignored */
	{ "normal",	0 },
#else
#if defined(MADV_HUGEPAGE)
	{ "hugepage",	MADV_HUGEPAGE },
#endif
#if defined(MADV_NOHUGEPAGE)
	{ "nohugepage",	MADV_NOHUGEPAGE },
#endif
#if defined(MADV_COLLAPSE)
	{ "collapse",	MADV_COLLAPSE },
#endif
#if defined(MADV_NORMAL)
	{ "normal",	MADV_NORMAL },
#endif
#endif
};

/*
 *  stress_stream_checksum_to_hexstr()
 *	turn a double into a hexadecimal string making zero assumptions about
 *	the size of a double since this maybe arch specific.
 */
static void stress_stream_checksum_to_hexstr(char *str, const size_t len, const double checksum)
{
	const unsigned char *ptr = (const unsigned char *)&checksum;
	size_t i, j;

	for (i = 0, j = 0; (i < sizeof(checksum)) && (j < len); i++, j += 2) {
		(void)snprintf(str + j, 3, "%2.2x", ptr[i]);
	}
	str[j] = '\0';
}

static inline void OPTIMIZE3 TARGET_CLONES stress_stream_copy_index0(
	double *const RESTRICT c,
	const double *const RESTRICT a,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;
	double volatile *RESTRICT cv = c;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i += 4) {
		STORE(cv[i + 0], a[i + 0]);
		STORE(cv[i + 1], a[i + 1]);
		STORE(cv[i + 2], a[i + 2]);
		STORE(cv[i + 3], a[i + 3]);
	}

	*rd_bytes += (double)n * (double)(sizeof(*a));
	*wr_bytes += (double)n * (double)(sizeof(*c));
	*fp_ops += 0.0;
}

#if defined(HAVE_NT_STORE_DOUBLE)
static inline void OPTIMIZE3 TARGET_CLONES stress_stream_copy_index0_nt(
	double *const RESTRICT c,
	const double *const RESTRICT a,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i += 4) {
		NT_STORE(c[i + 0], a[i + 0]);
		NT_STORE(c[i + 1], a[i + 1]);
		NT_STORE(c[i + 2], a[i + 2]);
		NT_STORE(c[i + 3], a[i + 3]);
	}

	// *rd_bytes += (double)n * (double)(sizeof(*a));
	// *wr_bytes += (double)n * (double)(sizeof(*c));
	*fp_ops += 0.0;
}
#endif

static inline void OPTIMIZE3 stress_stream_copy_index1(
	double *const RESTRICT c,
	const double *const RESTRICT a,
	const size_t *const RESTRICT idx1,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;
	double volatile *RESTRICT cv = c;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i++) {
		const size_t idx = idx1[i];

		STORE(cv[idx], a[idx]);
	}

	*rd_bytes += (double)n * (double)(sizeof(*a) + sizeof(*idx1));
	*wr_bytes += (double)n * (double)(sizeof(*c));
	*fp_ops += 0.0;
}

static inline void OPTIMIZE3 stress_stream_copy_index2(
	double *const RESTRICT c,
	const double *const RESTRICT a,
	const size_t *const RESTRICT idx1,
	const size_t *const RESTRICT idx2,
	const uint64_t n,
	double *rd_bytes,
	double *wr_bytes,
	double *fp_ops)
{
	uint64_t i;
	double volatile *RESTRICT cv = c;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i++)
		STORE(cv[idx1[i]], a[idx2[i]]);

	*rd_bytes += (double)n * (double)(sizeof(*a) + sizeof(*idx1) + sizeof(*idx2));
	*wr_bytes += (double)n * (double)(sizeof(*c));
	*fp_ops += 0.0;
}

static inline void OPTIMIZE3 stress_stream_copy_index3(
	double *const RESTRICT c,
	const double *const RESTRICT a,
	const size_t *const RESTRICT idx1,
	const size_t *const RESTRICT idx2,
	const size_t *const RESTRICT idx3,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;
	double volatile *RESTRICT cv = c;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i++)
		STORE(cv[idx3[idx1[i]]], a[idx2[i]]);

	*rd_bytes += (double)n * (double)(sizeof(*a) + sizeof(*idx1) + sizeof(*idx2) + sizeof(*idx3));
	*wr_bytes += (double)n * (double)(sizeof(*c));
	*fp_ops += 0.0;
}

static inline void OPTIMIZE3 TARGET_CLONES stress_stream_scale_index0(
	double *const RESTRICT b,
	const double *const RESTRICT c,
	const double q,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;
	double volatile *RESTRICT bv = b;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i += 4) {
		STORE(bv[i + 0], q * c[i + 0]);
		STORE(bv[i + 1], q * c[i + 1]);
		STORE(bv[i + 2], q * c[i + 2]);
		STORE(bv[i + 3], q * c[i + 3]);
	}

	*rd_bytes += (double)n * (double)(sizeof(*c));
	*wr_bytes += (double)n * (double)(sizeof(*b));
	*fp_ops += (double)n;
}

#if defined(HAVE_NT_STORE_DOUBLE)
static inline void OPTIMIZE3 TARGET_CLONES stress_stream_scale_index0_nt(
	double *const RESTRICT b,
	const double *const RESTRICT c,
	const double q,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i += 4) {
		NT_STORE(b[i + 0], q * c[i + 0]);
		NT_STORE(b[i + 1], q * c[i + 1]);
		NT_STORE(b[i + 2], q * c[i + 2]);
		NT_STORE(b[i + 3], q * c[i + 3]);
	}

	*rd_bytes += (double)n * (double)(sizeof(*c));
	*wr_bytes += (double)n * (double)(sizeof(*b));
	*fp_ops += (double)n;
}
#endif

static inline void OPTIMIZE3 TARGET_CLONES stress_stream_scale_index1(
	double *const RESTRICT b,
	const double *const RESTRICT c,
	const double q,
	const size_t *const RESTRICT idx1,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;
	double volatile *RESTRICT bv = b;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i++) {
		const size_t idx = idx1[i];

		STORE(bv[idx], q * c[idx]);
	}

	*rd_bytes += (double)n * (double)(sizeof(*c) + sizeof(*idx1));
	*wr_bytes += (double)n * (double)(sizeof(*b));
	*fp_ops += (double)n;
}

static inline void OPTIMIZE3 stress_stream_scale_index2(
	double *const RESTRICT b,
	const double *const RESTRICT c,
	const double q,
	const size_t *const RESTRICT idx1,
	const size_t *const RESTRICT idx2,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;
	double volatile *RESTRICT bv = b;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i++)
		STORE(bv[idx1[i]], q * c[idx2[i]]);

	*rd_bytes += (double)n * (double)(sizeof(*c) + sizeof(*idx1) + sizeof(*idx2));
	*wr_bytes += (double)n * (double)(sizeof(*b));
	*fp_ops += (double)n;
}

static inline void OPTIMIZE3 stress_stream_scale_index3(
	double *const RESTRICT b,
	const double *const RESTRICT c,
	const double q,
	const size_t *const RESTRICT idx1,
	const size_t *const RESTRICT idx2,
	const size_t *const RESTRICT idx3,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;
	double volatile *RESTRICT bv = b;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i++)
		STORE(bv[idx3[idx1[i]]], q * c[idx2[i]]);

	*rd_bytes += (double)n * (double)(sizeof(*c) + sizeof(*idx1) + sizeof(*idx2) + sizeof(*idx3));
	*wr_bytes += (double)n * (double)(sizeof(*b));
	*fp_ops += (double)n;
}

static inline void OPTIMIZE3 TARGET_CLONES stress_stream_add_index0(
	const double *const RESTRICT a,
	const double *const RESTRICT b,
	double *const RESTRICT c,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;
	double volatile *RESTRICT cv = c;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i += 4) {
		STORE(cv[i + 0], a[i + 0] + b[i + 0]);
		STORE(cv[i + 1], a[i + 1] + b[i + 1]);
		STORE(cv[i + 2], a[i + 2] + b[i + 2]);
		STORE(cv[i + 3], a[i + 3] + b[i + 3]);
	}

	*rd_bytes += (double)n * (double)(sizeof(*a) + sizeof(*b));
	*wr_bytes += (double)n * (double)(sizeof(*c));
	*fp_ops += (double)n;
}

#if defined(HAVE_NT_STORE_DOUBLE)
static inline void OPTIMIZE3 TARGET_CLONES stress_stream_add_index0_nt(
	const double *const RESTRICT a,
	const double *const RESTRICT b,
	double *const RESTRICT c,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i += 4) {
		NT_STORE(c[i + 0], a[i + 0] + b[i + 0]);
		NT_STORE(c[i + 1], a[i + 1] + b[i + 1]);
		NT_STORE(c[i + 2], a[i + 2] + b[i + 2]);
		NT_STORE(c[i + 3], a[i + 3] + b[i + 3]);
	}

	*rd_bytes += (double)n * (double)(sizeof(*a) + sizeof(*b));
	*wr_bytes += (double)n * (double)(sizeof(*c));
	*fp_ops += (double)n;
}
#endif

static inline void OPTIMIZE3 stress_stream_add_index1(
	const double *const RESTRICT a,
	const double *const RESTRICT b,
	double *const RESTRICT c,
	const size_t *const RESTRICT idx1,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;
	double volatile *RESTRICT cv = c;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i++) {
		const size_t idx = idx1[i];

		STORE(cv[idx], a[idx] + b[idx]);
	}

	*rd_bytes += (double)n * (double)(sizeof(*a) + sizeof(*b) + sizeof(*idx1));
	*wr_bytes += (double)n * (double)(sizeof(*c));
	*fp_ops += (double)n;
}

static inline void OPTIMIZE3 stress_stream_add_index2(
	const double *const RESTRICT a,
	const double *const RESTRICT b,
	double *const RESTRICT c,
	const size_t *const RESTRICT idx1,
	const size_t *const RESTRICT idx2,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;
	double volatile *RESTRICT cv = c;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i++) {
		const size_t idx = idx1[i];

		STORE(cv[idx], a[idx2[i]] + b[idx]);
	}

	*rd_bytes += (double)n * (double)(sizeof(*a) + sizeof(*b) + sizeof(*idx1) + sizeof(*idx2));
	*wr_bytes += (double)n * (double)(sizeof(*c));
	*fp_ops += (double)n;
}

static inline void OPTIMIZE3 stress_stream_add_index3(
	const double *const RESTRICT a,
	const double *const RESTRICT b,
	double *const RESTRICT c,
	const size_t *const RESTRICT idx1,
	const size_t *const RESTRICT idx2,
	const size_t *const RESTRICT idx3,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;
	double volatile *RESTRICT cv = c;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i++)
		STORE(cv[idx1[i]], a[idx2[i]] + b[idx3[i]]);

	*rd_bytes += (double)n * (double)(sizeof(*a) + sizeof(*b) + sizeof(*idx1) + sizeof(*idx2) + sizeof(*idx3));
	*wr_bytes += (double)n * (double)(sizeof(*c));
	*fp_ops += (double)n;
}

static inline void OPTIMIZE3 TARGET_CLONES stress_stream_triad_index0(
	double *const RESTRICT a,
	const double *const RESTRICT b,
	const double *const RESTRICT c,
	const double q,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;
	double volatile *RESTRICT av = a;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i += 4) {
		STORE(av[i + 0], b[i + 0] + (c[i + 0] * q));
		STORE(av[i + 1], b[i + 1] + (c[i + 1] * q));
		STORE(av[i + 2], b[i + 2] + (c[i + 2] * q));
		STORE(av[i + 3], b[i + 3] + (c[i + 3] * q));
	}

	*rd_bytes += (double)n * (double)(sizeof(*b) + sizeof(*c));
	*wr_bytes += (double)n * (double)(sizeof(*a));
	*fp_ops += (double)n * 2.0;
}

#if defined(HAVE_NT_STORE_DOUBLE)
static inline void OPTIMIZE3 TARGET_CLONES stress_stream_triad_index0_nt(
	double *const RESTRICT a,
	const double *const RESTRICT b,
	const double *const RESTRICT c,
	const double q,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i += 4) {
		NT_STORE(a[i + 0], b[i + 0] + (c[i + 0] * q));
		NT_STORE(a[i + 1], b[i + 1] + (c[i + 1] * q));
		NT_STORE(a[i + 2], b[i + 2] + (c[i + 2] * q));
		NT_STORE(a[i + 3], b[i + 3] + (c[i + 3] * q));
	}

	*rd_bytes += (double)n * (double)(sizeof(*b) + sizeof(*c));
	*wr_bytes += (double)n * (double)(sizeof(*a));
	*fp_ops += (double)n * 2.0;
}
#endif

static inline void OPTIMIZE3 stress_stream_triad_index1(
	double *const RESTRICT a,
	const double *const RESTRICT b,
	const double *const RESTRICT c,
	const double q,
	const size_t *const RESTRICT idx1,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;
	double volatile *RESTRICT av = a;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i++) {
		size_t idx = idx1[i];

		STORE(av[idx], b[idx] + (c[idx] * q));
	}
	*rd_bytes += (double)n * (double)(sizeof(*b) + sizeof(*c) + sizeof(*idx1));
	*wr_bytes += (double)n * (double)(sizeof(*a));
	*fp_ops += (double)n * 2.0;
}

static inline void OPTIMIZE3 stress_stream_triad_index2(
	double *const RESTRICT a,
	const double *const RESTRICT b,
	const double *const RESTRICT c,
	const double q,
	const size_t *const RESTRICT idx1,
	const size_t *const RESTRICT idx2,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;
	double volatile *RESTRICT av = a;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i++) {
		const size_t idx = idx1[i];

		STORE(av[idx], b[idx2[i]] + (c[idx] * q));
	}

	*rd_bytes += (double)n * (double)(sizeof(*b) + sizeof(*c) + sizeof(*idx1) + sizeof(*idx2));
	*wr_bytes += (double)n * (double)(sizeof(*a));
	*fp_ops += (double)n * 2.0;
}

static inline void OPTIMIZE3 stress_stream_triad_index3(
	double *const RESTRICT a,
	const double *const RESTRICT b,
	const double *const RESTRICT c,
	const double q,
	const size_t *const RESTRICT idx1,
	const size_t *const RESTRICT idx2,
	const size_t *const RESTRICT idx3,
	const uint64_t n,
	double *const RESTRICT rd_bytes,
	double *const RESTRICT wr_bytes,
	double *const RESTRICT fp_ops)
{
	uint64_t i;
	double volatile *RESTRICT av = a;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i++)
		STORE(av[idx1[i]], b[idx2[i]] + (c[idx3[i]] * q));

	*rd_bytes += (double)n * (double)(sizeof(*b) + sizeof(*c) + sizeof(*idx1) + sizeof(*idx2) + sizeof(*idx3));
	*wr_bytes += (double)n * (double)(sizeof(*a));
	*fp_ops += (double)n * 2.0;
}

static inline TARGET_CLONES OPTIMIZE3 void stress_stream_init_data(
	double *const RESTRICT a,
	double *const RESTRICT b,
	double *const RESTRICT c,
	const uint64_t n)
{
	const double divisor = 1.0 / (double)(4294967296ULL);
	const double delta = (double)stress_mwc32() * divisor;

	const uint32_t r = stress_mwc32();
	double v = (double)r * divisor;
	double *ptr, *ptr_end;

PRAGMA_UNROLL_N(4)
	for (ptr = a, ptr_end = a + n; ptr < ptr_end; ptr += 4) {
		STORE(ptr[0], v);
		STORE(ptr[1], v);
		STORE(ptr[2], v);
		STORE(ptr[3], v);
		v += delta;
	}

PRAGMA_UNROLL_N(4)
	for (ptr = b, ptr_end = b + n; ptr < ptr_end; ptr += 4) {
		STORE(ptr[0], v);
		STORE(ptr[1], v);
		STORE(ptr[2], v);
		STORE(ptr[3], v);
		v += delta;
	}

PRAGMA_UNROLL_N(4)
	for (ptr = c, ptr_end = c + n; ptr < ptr_end; ptr += 4) {
		STORE(ptr[0], v);
		STORE(ptr[1], v);
		STORE(ptr[2], v);
		STORE(ptr[3], v);
		v += delta;
	}
}

static double TARGET_CLONES OPTIMIZE3 stress_stream_checksum_data(
	const double *const RESTRICT a,
	const double *const RESTRICT b,
	const double *const RESTRICT c,
	const uint64_t n)
{
	double checksum = 0.0;
	uint64_t i;

PRAGMA_UNROLL_N(8)
	for (i = 0; i < n; i++) {
		checksum += a[i] + b[i] + c[i];
	}
	return checksum;
}

static inline void *stress_stream_mmap(
	stress_args_t *args,
	const uint64_t sz,
	const bool stream_mlock)
{
	void *ptr;

	ptr = stress_mmap_populate(NULL, (size_t)sz, PROT_READ | PROT_WRITE,
#if defined(HAVE_MADVISE)
		MAP_PRIVATE |
#else
		MAP_SHARED |
#endif
		MAP_ANONYMOUS, -1, 0);
	/* Coverity Scan believes NULL can be returned, doh */
	if (!ptr || (ptr == MAP_FAILED)) {
		pr_err("%s: cannot allocate %" PRIu64 " bytes\n",
			args->name, sz);
		ptr = MAP_FAILED;
	} else {
		stress_set_vma_anon_name(ptr, sz, "stream-buffer");
		if (stream_mlock)
			(void)shim_mlock(ptr, (size_t)sz);
#if defined(HAVE_MADVISE)
		size_t stream_madvise;
		int advice = MADV_NORMAL;

		if (stress_get_setting("stream-madvise", &stream_madvise))
			advice = stream_::stream_madvise_info[stream_madvise].advice;

		VOID_RET(int, madvise(ptr, (size_t)sz, advice));
#else
		UNEXPECTED
#endif
	}
	return ptr;
}

static inline uint64_t get_stream_L3_size(stress_args_t *args)
{
	uint64_t cache_size = 2 * MB;
#if defined(__linux__)
	stress_cpu_cache_cpus_t *cpu_caches;
	stress_cpu_cache_t *cache = NULL;
	uint16_t max_cache_level;
	const int numa_nodes = stress_numa_nodes();

	cpu_caches = stress_cpu_cache_get_all_details();
	if (!cpu_caches) {
		if (!args->instance)
			pr_inf("%s: using built-in defaults as unable to "
				"determine cache details\n", args->name);
		goto report_size;
	}
	max_cache_level = stress_cpu_cache_get_max_level(cpu_caches);
	if ((max_cache_level > 0) && (max_cache_level < 3) && (!args->instance))
		pr_inf("%s: no L3 cache, using L%" PRIu16 " size instead\n",
			args->name, max_cache_level);

	cache = stress_cpu_cache_get(cpu_caches, max_cache_level);
	if (!cache) {
		if (!args->instance)
			pr_inf("%s: using built-in defaults as no suitable "
				"cache found\n", args->name);
		stress_free_cpu_caches(cpu_caches);
		goto report_size;
	}
	if (!cache->size) {
		if (!args->instance)
			pr_inf("%s: using built-in defaults as unable to "
				"determine cache size\n", args->name);
		stress_free_cpu_caches(cpu_caches);
		goto report_size;
	}
	cache_size = cache->size;

	stress_free_cpu_caches(cpu_caches);
#else
	if (!args->instance)
		pr_inf("%s: using built-in defaults as unable to "
			"determine cache details\n", args->name);
#endif

#if defined(__linux__)
report_size:
	cache_size *= numa_nodes;
	if ((args->instance == 0) && (numa_nodes > 1))
		pr_inf("%s: scaling L3 cache size by number of numa nodes %d to %" PRIu64 "K\n",
			args->name, numa_nodes, cache_size / 1024);
#endif
	return cache_size;
}

static void stress_stream_init_index(
	size_t *RESTRICT idx,
	const uint64_t n)
{
	uint64_t i;

	for (i = 0; i < n; i++)
		idx[i] = i;

	for (i = 0; i < n; i++) {
		const uint64_t j = stress_mwc64modn(n);
		const uint64_t tmp = idx[i];

		idx[i] = idx[j];
		idx[j] = tmp;
	}
}

/*
 *  stress_stream()
 *	stress cache/memory/CPU with stream stressors
 */
static int stress_stream(stress_args_t *args)
{
	int rc = EXIT_FAILURE;
	double *a = (double *)MAP_FAILED, *b = (double *)MAP_FAILED, *c = (double *)MAP_FAILED;
	size_t *idx1 = (size_t *)MAP_FAILED, *idx2 = (size_t *)MAP_FAILED, *idx3 = (size_t *)MAP_FAILED;
	const double q = 3.0;
	double old_checksum = -1.0;
	double fp_ops = 0.0, t1, t2, dt;
	uint32_t w, z, stream_index = 0;
	uint64_t L3, sz, n, sz_idx;
	uint64_t stream_L3_size = DEFAULT_STREAM_L3_SIZE;
	uint32_t init_counter, init_counter_max;
	bool guess = false;
	bool stream_mlock = false;
#if defined(HAVE_NT_STORE_DOUBLE)
	const bool has_sse2 = stress_cpu_x86_has_sse2();
#endif
	double rd_bytes = 0.0, wr_bytes = 0.0;
	const bool verify = !!(g_opt_flags & OPT_FLAGS_VERIFY);

	stress_catch_sigill();

	(void)stress_get_setting("stream-mlock", &stream_mlock);

	if (stress_get_setting("stream-l3-size", &stream_L3_size))
		L3 = stream_L3_size;
	else
		L3 = get_stream_L3_size(args);

	(void)stress_get_setting("stream-index", &stream_index);

	/* Have to take a hunch and badly guess size */
	if (!L3) {
		guess = true;
		L3 = (uint64_t)stress_get_processors_configured() * DEFAULT_STREAM_L3_SIZE;
	}

	if (args->instance == 0) {
		pr_inf("%s: stressor loosely based on a variant of the "
			"STREAM benchmark code\n", args->name);
		pr_inf("%s: do NOT submit any of these results "
			"to the STREAM benchmark results\n", args->name);
		if (guess) {
			pr_inf("%s: cannot determine CPU L3 cache size, "
				"defaulting to %" PRIu64 "K\n",
				args->name, L3 / 1024);
		} else {
			pr_inf("%s: Using cache size of %" PRIu64 "K\n",
				args->name, L3 / 1024);
		}
	}

	/* ..and shared amongst all the STREAM stressor instances */
	L3 /= args->instances;
	if (L3 < args->page_size)
		L3 = args->page_size;

	/*
	 *  Each array must be at least 4 x the
	 *  size of the L3 cache
	 */
	sz = (L3 * 4);
	n = sz / sizeof(*a);
	/*
	 *  n must be a multiple of the max unroll size (8)
	 */
	n = (n + 7) & ~(uint64_t)7;
	sz = n * sizeof(*a);
	sz_idx = n * sizeof(size_t);

	a = (double *)stress_stream_mmap(args, sz, stream_mlock);
	if (a == MAP_FAILED)
		goto err_unmap;
	b = (double *)stress_stream_mmap(args, sz, stream_mlock);
	if (b == MAP_FAILED)
		goto err_unmap;
	c = (double *)stress_stream_mmap(args, sz, stream_mlock);
	if (c == MAP_FAILED)
		goto err_unmap;

	switch (stream_index) {
	case 3:
		idx3 = (size_t *)stress_stream_mmap(args, sz_idx, stream_mlock);
		if (idx3 == MAP_FAILED)
			goto err_unmap;
		stress_stream_init_index(idx3, n);
		goto case_stream_index_2;
	case 2:
case_stream_index_2:
		idx2 = (size_t *)stress_stream_mmap(args, sz_idx, stream_mlock);
		if (idx2 == MAP_FAILED)
			goto err_unmap;
		stress_stream_init_index(idx2, n);
		goto case_stream_index_1;
	case 1:
case_stream_index_1:
		idx1 = (size_t *)stress_stream_mmap(args, sz_idx, stream_mlock);
		if (idx1 == MAP_FAILED)
			goto err_unmap;
		stress_stream_init_index(idx1, n);
		break;
	case 0:
	default:
		break;
	}

	stress_mwc_get_seed(&w, &z);

	init_counter = 0;
	init_counter_max = verify ? 1 : 64;

	rc = EXIT_SUCCESS;
	dt = 0.0;
	do {
		if (init_counter == 0) {
			stress_mwc_set_seed(w, z);
			stress_stream_init_data(a, b, c, n);
		}
		init_counter++;
		if (init_counter >= init_counter_max)
			init_counter = 0;

		switch (stream_index) {
		case 3:
			t1 = stress_time_now();
			stress_stream_copy_index3(c, a, idx1, idx2, idx3, n, &rd_bytes, &wr_bytes, &fp_ops);
			stress_stream_scale_index3(b, c, q, idx1, idx2, idx3, n, &rd_bytes, &wr_bytes, &fp_ops);
			stress_stream_add_index3(c, b, a, idx1, idx2, idx3, n, &rd_bytes, &wr_bytes, &fp_ops);
			stress_stream_triad_index3(a, b, c, q, idx1, idx2, idx3, n, &rd_bytes, &wr_bytes, &fp_ops);
			t2 = stress_time_now();
			break;
		case 2:
			t1 = stress_time_now();
			stress_stream_copy_index2(c, a, idx1, idx2, n, &rd_bytes, &wr_bytes, &fp_ops);
			stress_stream_scale_index2(b, c, q, idx1, idx2, n, &rd_bytes, &wr_bytes, &fp_ops);
			stress_stream_add_index2(c, b, a, idx1, idx2, n, &rd_bytes, &wr_bytes, &fp_ops);
			stress_stream_triad_index2(a, b, c, q, idx1, idx2, n, &rd_bytes, &wr_bytes, &fp_ops);
			t2 = stress_time_now();
			break;
		case 1:
			t1 = stress_time_now();
			stress_stream_copy_index1(c, a, idx1, n, &rd_bytes, &wr_bytes, &fp_ops);
			stress_stream_scale_index1(b, c, q, idx1, n, &rd_bytes, &wr_bytes, &fp_ops);
			stress_stream_add_index1(c, b, a, idx1, n, &rd_bytes, &wr_bytes, &fp_ops);
			stress_stream_triad_index1(a, b, c, q, idx1, n, &rd_bytes, &wr_bytes, &fp_ops);
			t2 = stress_time_now();
			break;
		case 0:
		default:
#if defined(HAVE_NT_STORE_DOUBLE)
			if (has_sse2) {
				t1 = stress_time_now();
				stress_stream_copy_index0_nt(c, a, n, &rd_bytes, &wr_bytes, &fp_ops);
				stress_stream_scale_index0_nt(b, c, q, n, &rd_bytes, &wr_bytes, &fp_ops);
				stress_stream_add_index0_nt(c, b, a, n,  &rd_bytes, &wr_bytes, &fp_ops);
				stress_stream_triad_index0_nt(a, b, c, q, n, &rd_bytes, &wr_bytes, &fp_ops);
				t2 = stress_time_now();
				break;
			}
#endif
			t1 = stress_time_now();
			stress_stream_copy_index0(c, a, n, &rd_bytes, &wr_bytes, &fp_ops);
			stress_stream_scale_index0(b, c, q, n, &rd_bytes, &wr_bytes, &fp_ops);
			stress_stream_add_index0(c, b, a, n, &rd_bytes, &wr_bytes, &fp_ops);
			stress_stream_triad_index0(a, b, c, q, n, &rd_bytes, &wr_bytes, &fp_ops);
			t2 = stress_time_now();
			break;
		}
		dt += (t2 - t1);

		if (verify) {
			double new_checksum;

			new_checksum = stress_stream_checksum_data(a, b, c, n);
			if ((old_checksum > 0.0) && (fabs(new_checksum - old_checksum) > 0.001)) {
				char new_str[32], old_str[32];

				stress_stream_checksum_to_hexstr(new_str, sizeof(new_str), new_checksum);
				stress_stream_checksum_to_hexstr(old_str, sizeof(old_str), old_checksum);

				if (strcmp(old_str, new_str)) {
					pr_fail("%s: checksum failure, got 0x%s, expecting 0x%s\n",
						args->name, new_str, old_str);
					rc = EXIT_FAILURE;
					break;
				}
			} else {
				old_checksum = new_checksum;
			}
		}
		stress_bogo_inc(args);
	} while (stress_continue(args));

	if (dt >= 4.5) {
		const double mb_rd_rate = (rd_bytes / (double)MB) / dt;
		const double mb_wr_rate = (wr_bytes / (double)MB) / dt;
		const double fp_rate = (fp_ops / 1000000.0) / dt;

		pr_inf("%s: memory rate: %.2f MB read/sec, %.2f MB write/sec, %.2f double precision Mflop/sec"
			" (instance %" PRIu32 ")\n",
			args->name, mb_rd_rate, mb_wr_rate, fp_rate, args->instance);
		stress_metrics_set(args, 0, "MB per sec memory read rate",
			mb_rd_rate, STRESS_METRIC_HARMONIC_MEAN);
		stress_metrics_set(args, 1, "MB per sec memory write rate",
			mb_wr_rate, STRESS_METRIC_HARMONIC_MEAN);
		stress_metrics_set(args, 2, "Mflop per sec (double precision) compute rate",
			fp_rate, STRESS_METRIC_HARMONIC_MEAN);
	} else {
		if (args->instance == 0)
			pr_inf("%s: run duration too short to reliably determine memory rate\n", args->name);
	}

err_unmap:
	stress_set_proc_state(args->name, STRESS_STATE_DEINIT);
	if (idx3 != MAP_FAILED)
		(void)munmap((void *)idx3, sz_idx);
	if (idx2 != MAP_FAILED)
		(void)munmap((void *)idx2, sz_idx);
	if (idx1 != MAP_FAILED)
		(void)munmap((void *)idx1, sz_idx);
	if (c != MAP_FAILED)
		(void)munmap((void *)c, sz);
	if (b != MAP_FAILED)
		(void)munmap((void *)b, sz);
	if (a != MAP_FAILED)
		(void)munmap((void *)a, sz);
	return rc;
}

static const char *stress_stream_madvise(const size_t i)
{
	return (i < SIZEOF_ARRAY(stream_madvise_info)) ? stream_madvise_info[i].name : NULL;
}

static const stress_opt_t opts[] = {
	{ OPT_stream_index,   "stream-index",   TYPE_ID_UINT32, 0, 3, NULL },
	{ OPT_stream_l3_size, "stream-l3-size", TYPE_ID_UINT64_BYTES_VM, MIN_STREAM_L3_SIZE, MAX_STREAM_L3_SIZE, NULL },
	{ OPT_stream_madvise, "stream-madvise", TYPE_ID_SIZE_T_METHOD, 0, 0, (void*)stress_stream_madvise },
	{ OPT_stream_mlock,   "stream-mlock",   TYPE_ID_BOOL, 0, 1, NULL },
	END_OPT,
};

const stressor_info_t stress_stream_info = {
	.stressor = stress_stream,
	.cls = CLASS_CPU | CLASS_FP | CLASS_CPU_CACHE | CLASS_MEMORY,
	.opts = opts,
	.verify = VERIFY_OPTIONAL,
	.help = help
};

void BM_STRESS_NG_Stream(benchmark::State& state) {
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;			/* shared memory */
  double g_opt_timeout = 1.0;
  const struct stressor_info *info = &stress_stream_info;
  stats.start = stress_time_now();

  args.stats = &stats;
  args.name = "stream";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  // run
  for (auto _ : state) {
    // Reset the counter for each iteration
    args.ci.counter = 0;

    // Call the stress_stream function
    int rc = stress_stream(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Stream stress test failed");
      return;
    }
  }
}
}

namespace hdd_ {

extern "C" {
    extern const stressor_info_t stress_hdd_info;
}

void BM_STRESS_NG_Hdd(benchmark::State& state) {
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 1.0;
  const struct stressor_info *info = &stress_hdd_info;
  stats.start = stress_time_now();

  args.stats = &stats;
  args.name = "hdd";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Hdd stress test failed");
      return;
    }
  }
}
} // namespace hdd_

namespace hdd_1MB_ {

extern "C" {
    extern const stressor_info_t stress_hdd_info;
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Hdd_1MB(benchmark::State& state) {
  uint64_t hdd_bytes = 1048576ULL;
  stress_set_setting_global("hdd-bytes", TYPE_ID_UINT64_BYTES_FS, &hdd_bytes);
  // HDD_OPT_WR_SEQ=0x1: sequential write only (no read phase)
  int hdd_flags = 0x00000001;
  stress_set_setting_global("hdd-flags", TYPE_ID_INT, &hdd_flags);
  // O_DSYNC=0x1000: each write syscall blocks until data reaches disk
  int hdd_oflags = 0x1000;
  stress_set_setting_global("hdd-oflags", TYPE_ID_INT, &hdd_oflags);
  // opts_set=true: disable aggressive mode cycling
  bool opts_set = true;
  stress_set_setting_global("hdd-opts-set", TYPE_ID_BOOL, &opts_set);

  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 60.0;
  const struct stressor_info *info = &stress_hdd_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "hdd";
  args.max_ops = 16;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Hdd 1MB stress test failed");
      return;
    }
  }
}
} // namespace hdd_1MB_

namespace hdd_4MB_ {

extern "C" {
    extern const stressor_info_t stress_hdd_info;
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Hdd_4MB(benchmark::State& state) {
  uint64_t hdd_bytes = 4194304ULL;
  stress_set_setting_global("hdd-bytes", TYPE_ID_UINT64_BYTES_FS, &hdd_bytes);
  // HDD_OPT_WR_SEQ=0x1: sequential write only (no read phase)
  int hdd_flags = 0x00000001;
  stress_set_setting_global("hdd-flags", TYPE_ID_INT, &hdd_flags);
  // O_DSYNC=0x1000: each write syscall blocks until data reaches disk
  int hdd_oflags = 0x1000;
  stress_set_setting_global("hdd-oflags", TYPE_ID_INT, &hdd_oflags);
  // opts_set=true: disable aggressive mode cycling
  bool opts_set = true;
  stress_set_setting_global("hdd-opts-set", TYPE_ID_BOOL, &opts_set);

  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 60.0;
  const struct stressor_info *info = &stress_hdd_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "hdd";
  args.max_ops = 64;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Hdd 4MB stress test failed");
      return;
    }
  }
}
} // namespace hdd_4MB_

namespace hdd_8MB_ {

extern "C" {
    extern const stressor_info_t stress_hdd_info;
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Hdd_8MB(benchmark::State& state) {
  uint64_t hdd_bytes = 8388608ULL;
  stress_set_setting_global("hdd-bytes", TYPE_ID_UINT64_BYTES_FS, &hdd_bytes);
  // HDD_OPT_WR_SEQ=0x1: sequential write only (no read phase)
  int hdd_flags = 0x00000001;
  stress_set_setting_global("hdd-flags", TYPE_ID_INT, &hdd_flags);
  // O_DSYNC=0x1000: each write syscall blocks until data reaches disk
  int hdd_oflags = 0x1000;
  stress_set_setting_global("hdd-oflags", TYPE_ID_INT, &hdd_oflags);
  // opts_set=true: disable aggressive mode cycling
  bool opts_set = true;
  stress_set_setting_global("hdd-opts-set", TYPE_ID_BOOL, &opts_set);

  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 60.0;
  const struct stressor_info *info = &stress_hdd_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "hdd";
  args.max_ops = 86;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Hdd 8MB stress test failed");
      return;
    }
  }
}
} // namespace hdd_8MB_

namespace hdd_16MB_ {

extern "C" {
    extern const stressor_info_t stress_hdd_info;
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_Hdd_16MB(benchmark::State& state) {
  uint64_t hdd_bytes = 16777216ULL;
  stress_set_setting_global("hdd-bytes", TYPE_ID_UINT64_BYTES_FS, &hdd_bytes);
  // HDD_OPT_WR_SEQ=0x1: sequential write only (no read phase)
  int hdd_flags = 0x00000001;
  stress_set_setting_global("hdd-flags", TYPE_ID_INT, &hdd_flags);
  // O_DSYNC=0x1000: each write syscall blocks until data reaches disk
  int hdd_oflags = 0x1000;
  stress_set_setting_global("hdd-oflags", TYPE_ID_INT, &hdd_oflags);
  // opts_set=true: disable aggressive mode cycling
  bool opts_set = true;
  stress_set_setting_global("hdd-opts-set", TYPE_ID_BOOL, &opts_set);

  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 60.0;
  const struct stressor_info *info = &stress_hdd_info;
  stats.start = stress_time_now();
  args.stats = &stats;
  args.name = "hdd";
  args.max_ops = 96;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Hdd 16MB stress test failed");
      return;
    }
  }
}
} // namespace hdd_16MB_

namespace iomix_ {

extern "C" {
    extern const stressor_info_t stress_iomix_info;
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
    int stress_lock_mem_map(void);
    void stress_lock_mem_unmap(void);
}

void BM_STRESS_NG_Iomix(benchmark::State& state) {
  // iomix forks child processes that share a bogo-op counter through a
  // stress_lock_create() lock.  stress_lock_mem_map() must be called first
  // to initialise the shared-memory lock arena (not done by the benchmark
  // framework by default).
  if (stress_lock_mem_map() < 0) {
    state.SkipWithError("Iomix: stress_lock_mem_map failed");
    return;
  }

  // Reduce from 1GB default to 4MB so the initial fallocate+O_SYNC setup is fast.
  // max_ops=0 (unlimited) lets timeout control the iteration length.
  // The stressor already opens with O_SYNC so writes go directly to disk.
  off_t iomix_bytes = 4 * 1024 * 1024;
  stress_set_setting_global("iomix-bytes", TYPE_ID_OFF_T, &iomix_bytes);

  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 1.5;
  const struct stressor_info *info = &stress_iomix_info;
  stats.start = stress_time_now();

  args.stats = &stats;
  args.name = "iomix";
  args.max_ops = 0;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      stress_lock_mem_unmap();
      state.SkipWithError("Iomix stress test failed");
      return;
    }
    args.time_end = stress_time_now() + (double)g_opt_timeout;
  }
  stress_lock_mem_unmap();
}
} // namespace iomix_

namespace splice_ {

extern "C" {
    extern const stressor_info_t stress_splice_info;
}

void BM_STRESS_NG_Splice(benchmark::State& state) {
  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 1.0;
  const struct stressor_info *info = &stress_splice_info;
  stats.start = stress_time_now();

  args.stats = &stats;
  args.name = "splice";
  args.max_ops = 1;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("Splice stress test failed");
      return;
    }
  }
}
} // namespace splice_

namespace sync_file_ {

extern "C" {
    extern const stressor_info_t stress_sync_file_info;
    int stress_set_setting_global(const char *name, stress_type_id_t type_id, const void *value);
}

void BM_STRESS_NG_SyncFile(benchmark::State& state) {
  // Use 4MB file and max_ops=2 (2 complete write+sync passes).
  // Timeout-only (max_ops=0) caused hangs: the write loop dirtied pages
  // faster than the HDD could flush, and the final sync_file_range()
  // had to drain a large backlog while blocked in kernel (D state).
  // At ~5 MB/s HDD speed: 2 passes × ~0.8s ≈ 1.6s per iteration.
  off_t sf_bytes = 4 * 1024 * 1024;
  stress_set_setting_global("sync_file-bytes", TYPE_ID_OFF_T, &sf_bytes);

  stress_args_t args;
  stress_stats_t stats;
  stress_shared_t *g_shared;
  double g_opt_timeout = 10.0;
  const struct stressor_info *info = &stress_sync_file_info;
  stats.start = stress_time_now();

  args.stats = &stats;
  args.name = "sync-file";
  args.max_ops = 2;
  args.instance = 1;
  args.instances = 1;
  args.page_size = stress_get_page_size();
  args.time_end = stress_time_now() + (double)g_opt_timeout;
  args.mapped = &g_shared->mapped;
  args.metrics = &stats.metrics;
  args.info = info;
  args.pid = stress_mwc16();
  for (auto _ : state) {
    args.ci.counter = 0;
    int rc = info->stressor(&args);
    if (rc != EXIT_SUCCESS) {
      state.SkipWithError("SyncFile stress test failed");
      return;
    }
  }
}
} // namespace sync_file_

void RegisterBenchmarks() {
  auto benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Readahead", fleetbench::stress_ng::readahead_::BM_STRESS_NG_Readahead);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Readahead_1MB", fleetbench::stress_ng::readahead_1MB_::BM_STRESS_NG_Readahead_1MB);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Readahead_4MB", fleetbench::stress_ng::readahead_4MB_::BM_STRESS_NG_Readahead_4MB);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Readahead_8MB", fleetbench::stress_ng::readahead_8MB_::BM_STRESS_NG_Readahead_8MB);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Readahead_16MB", fleetbench::stress_ng::readahead_16MB_::BM_STRESS_NG_Readahead_16MB);
  benchmark->Iterations(1);


  // benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_TLB_Shootdown", fleetbench::stress_ng::tlb_shootdown_::BM_STRESS_NG_TLB_Shootdown);
  // benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Radixsort", fleetbench::stress_ng::radixsort_::BM_STRESS_NG_Radixsort);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Fallocate", fleetbench::stress_ng::fallocate_::BM_STRESS_NG_Fallocate);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Fallocate_1MB", fleetbench::stress_ng::fallocate_1MB_::BM_STRESS_NG_Fallocate_1MB);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Fallocate_32MB", fleetbench::stress_ng::fallocate_32MB_::BM_STRESS_NG_Fallocate_32MB);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Fallocate_128MB", fleetbench::stress_ng::fallocate_128MB_::BM_STRESS_NG_Fallocate_128MB);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Fallocate_512MB", fleetbench::stress_ng::fallocate_512MB_::BM_STRESS_NG_Fallocate_512MB);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Fallocate_2GB", fleetbench::stress_ng::fallocate_2GB_::BM_STRESS_NG_Fallocate_2GB);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Fallocate_2MB", fleetbench::stress_ng::fallocate_2MB_::BM_STRESS_NG_Fallocate_2MB);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Fallocate_4MB", fleetbench::stress_ng::fallocate_4MB_::BM_STRESS_NG_Fallocate_4MB);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Fallocate_16MB", fleetbench::stress_ng::fallocate_16MB_::BM_STRESS_NG_Fallocate_16MB);
  benchmark->Iterations(1);


  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Sendfile", fleetbench::stress_ng::sendfile_::BM_STRESS_NG_Sendfile);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Sendfile_1MB", fleetbench::stress_ng::sendfile_1MB_::BM_STRESS_NG_Sendfile_1MB);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Sendfile_32MB", fleetbench::stress_ng::sendfile_32MB_::BM_STRESS_NG_Sendfile_32MB);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Sendfile_256MB", fleetbench::stress_ng::sendfile_256MB_::BM_STRESS_NG_Sendfile_256MB);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Sendfile_1GB", fleetbench::stress_ng::sendfile_1GB_::BM_STRESS_NG_Sendfile_1GB);
  benchmark->Iterations(1);


  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Mmaphuge", fleetbench::stress_ng::mmaphuge_::BM_STRESS_NG_Mmaphuge);
  benchmark->Iterations(1);

  // benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Cache", fleetbench::stress_ng::cache_::BM_STRESS_NG_Cache);
  // benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Stream", fleetbench::stress_ng::stream_::BM_STRESS_NG_Stream);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Hdd", fleetbench::stress_ng::hdd_::BM_STRESS_NG_Hdd);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Hdd_1MB", fleetbench::stress_ng::hdd_1MB_::BM_STRESS_NG_Hdd_1MB);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Hdd_4MB", fleetbench::stress_ng::hdd_4MB_::BM_STRESS_NG_Hdd_4MB);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Hdd_8MB", fleetbench::stress_ng::hdd_8MB_::BM_STRESS_NG_Hdd_8MB);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Hdd_16MB", fleetbench::stress_ng::hdd_16MB_::BM_STRESS_NG_Hdd_16MB);
  benchmark->Iterations(1);






  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Iomix", fleetbench::stress_ng::iomix_::BM_STRESS_NG_Iomix);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_Splice", fleetbench::stress_ng::splice_::BM_STRESS_NG_Splice);
  benchmark->Iterations(1);

  benchmark = benchmark::RegisterBenchmark("BM_STRESS_NG_SyncFile", fleetbench::stress_ng::sync_file_::BM_STRESS_NG_SyncFile);
  benchmark->Iterations(1);

}

}  // namespace stress_ng
}  // namespace fleetbench

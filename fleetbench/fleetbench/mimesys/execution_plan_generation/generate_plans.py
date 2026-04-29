import random
import matplotlib.pyplot as plt
import h5py
import numpy as np

action_profile = [
    ["BM_COMPRESSION_Snappy_COMPRESS_Fleet", 53900576, 1657.46176],
    ["BM_COMPRESSION_Snappy_DECOMPRESS_Fleet", 80294446, 942.06976],
    ["BM_COMPRESSION_ZSTD_COMPRESS_Fleet/compression_level:-1/window_log:15", 170377236, 8215.10144],
    ["BM_COMPRESSION_ZSTD_DECOMPRESS_Fleet/compression_level:0/window_log:0", 312057555, 5105.201024],
    # ["BM_COMPRESSION_Flate_COMPRESS_Fleet/compression_level:6/window_log:15", 1045365509, 96671.824],
    # ["BM_COMPRESSION_Flate_DECOMPRESS_Fleet/compression_level:6/window_log:15", 1382011054, 13886.52288],
    # ["BM_COMPRESSION_Brotli_COMPRESS_Fleet/compression_level:2/window_log:18", 348662669, 22567.75936],
    # ["BM_COMPRESSION_Brotli_DECOMPRESS_Fleet/compression_level:2/window_log:18", 265078050, 6995.1232],
    ["BM_HASHING_Extendcrc32cinternal_Fleet_cold", 448282303, 219.571074],
    # ["BM_HASHING_Computecrc32c_Fleet_cold", 100459044, 16.31022],
    ["BM_HASHING_Combine_contiguous_Fleet_cold", 732656243, 16.31016],
    ["BM_LIBC_Memcpy_Fleet_L1", 72306296, 352.678],
    # ["BM_LIBC_Memcpy_Fleet_Cold", 534375403, 45907.97],
    ["BM_LIBC_Memmove_Fleet_L1", 73532659, 378.539],
    # ["BM_LIBC_Memmove_Fleet_Cold", 1021251821, 91638.661],
    ["BM_LIBC_Memcmp_Fleet_L1", 72217505, 586.846],
    # ["BM_LIBC_Memcmp_Fleet_Cold", 1102271691, 44251.133],
    ["BM_LIBC_Bcmp_Fleet_L1", 69947348, 446.573],
    # ["BM_LIBC_Bcmp_Fleet_Cold", 1064260863, 36756.43],
    ["BM_LIBC_Memset_Fleet_L1", 71437245, 617.701],
    # ["BM_LIBC_Memset_Fleet_Cold", 1146657723, 184098.735],
    ["BM_PROTO_Arena", 96566194, 17723.627],
    ["BM_SIMD_SerialDistanceComputation/num_blocks:256/enable_avx512:false/flush_cache:false", 25856611, 613.954],
    ["BM_SIMD_SerialTopn/num_blocks:256/enable_avx512:false/flush_cache:false", 269063738, 524.563],
    # ["BM_SIMD_SerialTopnFloat/num_blocks:256/enable_avx512:false/flush_cache:false", 281884124, 604.629],
    ["BM_CORD_Fleet", 17071780, 86.226],
    # ["BM_SWISSMAP_InsertHit_Cold<::absl::flat_hash_set, 64>/set_size:64/density:0", 20913973, 1.4763],
    # ["BM_SWISSMAP_InsertHit_Hot<::absl::flat_hash_set, 64>/set_size:64/density:0", 88235926, 5306.1184],
    ["BM_STRESS_NG_Readahead", 61080339],
    ["BM_STRESS_NG_Radixsort", 159113580],
    ["BM_STRESS_NG_Fallocate", 17552815],
    ["BM_STRESS_NG_Sendfile", 17100289],
    ["BM_STRESS_NG_Mmaphuge", 17075991],
    ["BM_STRESS_NG_Stream", 378222191],
]

def get_num_benchmark_iters_from_execution_plan(execution_plan, num_instruction_budget, action_profile):
    """
    Given an execution plan (list of ratios) and an instruction budget,
    compute the number of iterations for each benchmark to reach the target instruction count.
    Returns (num_iters_list, no_op_ratio)
    """
    num_iters = []
    total_ratio = sum(execution_plan)
    no_op_ratio = max(0.0, 1 - total_ratio)
    for i, ratio in enumerate(execution_plan):
        if total_ratio > 1:
            ratio /= total_ratio
        target_insts = ratio * num_instruction_budget
        # The second item in each action_profile entry is the instruction count per iteration
        insts_per_iter = action_profile[i][1]
        # Compute the number of iterations (closest integer)
        lower = int(target_insts // insts_per_iter)
        upper = lower + 1
        lower_diff = abs(target_insts - (insts_per_iter * lower))
        upper_diff = abs(target_insts - (insts_per_iter * upper))
        num_iters_for_benchmark = lower if lower_diff <= upper_diff else upper
        num_iters.append(num_iters_for_benchmark)
    return num_iters, no_op_ratio

def estimated_duration_from_execution_plan(execution_plan, action_profile):
    num_iters, no_op_ratio = get_num_benchmark_iters_from_execution_plan(execution_plan, 3e9, action_profile)
    estimated_instruction_count = 0.0
    for i, num_iter in enumerate(num_iters):
        if num_iter > 0:
            # The third item in each action_profile entry is the time per iteration
            estimated_instruction_count += num_iter * action_profile[i][1]

def write_to_csv(action_weights, file_path):
    with open(file_path, 'w') as f:
        for row in action_weights:
            f.write(','.join(map(str, row)) + '\n')

def write_to_hdf5(action_weights, file_path):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('execution_plan', data=action_weights)

def select_distribution(include_one_hot=False):
    target_avg = random.uniform(0.2, 1.0)
    target_var = random.uniform(0, 0.3)
    target_dist = random.choice(
        ['normal', 'uniform', 'exponential', 'bimodal', 'sawtooth', 'burst', 'high_utilization'] + (['one_hot'] if include_one_hot else [])
    )

    def clamp(x):
        return max(0.2, min(1, x))

    if target_dist == 'normal':
        mean = target_avg
        stddev = target_var ** 0.5
        return lambda: clamp(random.gauss(mean, stddev))
    elif target_dist == 'uniform':
        low = max(0, target_avg - target_var)
        high = min(1, target_avg + target_var)
        return lambda: random.uniform(low, high)
    elif target_dist == 'exponential':
        rate = 1 / target_avg
        return lambda: clamp(random.expovariate(rate))

    # 'bimodal': Simulate systems with two distinct utilization levels (e.g., idle and busy)
    elif target_dist == 'bimodal':
        low = random.uniform(0.2, 0.5)
        high = random.uniform(0.7, 1.0)
        prob_high = random.uniform(0.2, 0.8)
        return lambda: high if random.random() < prob_high else low

    # 'sawtooth': Simulate periodic ramp-up and drop-off in utilization
    elif target_dist == 'sawtooth':
        period = random.randint(5, 20)
        counter = {'i': 0}
        def sawtooth():
            val = 0.2 + (counter['i'] % period) / period * (1.0 - 0.2)
            counter['i'] += 1
            return val
        return sawtooth

    # 'burst': Simulate mostly low utilization with occasional bursts
    elif target_dist == 'burst':
        burst_prob = 0.1
        low_val = random.uniform(0.2, 0.4)
        burst_val = random.uniform(0.8, 1.0)
        return lambda: burst_val if random.random() < burst_prob else low_val

    # 'high_utilization': Simulate consistently high utilization scenarios
    elif target_dist == 'high_utilization':
        high_val = random.uniform(0.7, 1.0)
        return lambda: high_val

    elif target_dist == 'one_hot':
        return lambda: 1.0 if random.random() < 0.05 else 0.0

def sample_simplex_direct(n, d):
    # n: number of samples, d: dimension of the vector
    uniforms = np.random.rand(n, d + 1)
    uniforms_sorted = np.sort(uniforms, axis=1)
    # Add 0 at beginning, 1 at end, then take differences
    uniforms_ext = np.hstack([np.zeros((n, 1)), uniforms_sorted, np.ones((n, 1))])
    diffs = uniforms_ext[:, 1:] - uniforms_ext[:, :-1]
    return diffs[:, :-1]  # First d differences are the sample

def generate_random_action_weights(num_actions, target_utilization):
    weights = sample_simplex_direct(1, num_actions)[0]

    print("Sum of weights:", sum(weights))

    return [w * target_utilization for w in weights]

# def generate_random_action_weights_mutation(num_actions, current_pool):

def generate_next_pool(archive, prev_archive_len=0):
    new_pool = []
    for i in range(len(archive)):
        for j in range(max(prev_archive_len, i + 1), len(archive)):
            v1 = archive[i]
            v2 = archive[j]
            mixed = [(a + b) / 2.0 for a, b in zip(v1, v2)]
            new_pool.append(mixed)
    return new_pool

if __name__ == "__main__":
    num_data_points = 1000
    num_actions = 21 # 34
    target_duration = 10000
    avg_utilizations = []
    all_utilizations = []
    action_weight_sums = []
    target_utilizations = [0.2 * i for i in range(1, 5 + 1)]
    current_pool = []
    archive = []
    prev_archive_len = len(archive)

    window_len = 5
    max_concurrency = 20

    random_int_val = random.randint(0, 1000)

    num_data_points_per_num_actions = num_data_points // (num_actions - 1)
    for file_idx in range(num_data_points):
        action_weights_list = []
        # action_weights = generate_random_action_weights(num_actions, 1)

        if file_idx < num_actions:
            action_weights = [1.0 if i == file_idx else 0.0 for i in range(num_actions)]
        else:
            num_actions_in_weights = (file_idx % (num_actions - 1)) + 2
            selected_indices = random.sample(range(num_actions), num_actions_in_weights)
            action_weights_partial = generate_random_action_weights(num_actions_in_weights, 1.0)
            action_weights = [0.0] * num_actions
            for idx, weight in zip(selected_indices, action_weights_partial):
                action_weights[idx] = weight

        random_idx = (file_idx + random_int_val) % 20
        new_target_utilizations = [e - (0.01 * random_idx) for e in target_utilizations]

        for target_util_idx, target_utilization in enumerate(new_target_utilizations):
            updated_action_weights = [w * target_utilization for w in action_weights]
            action_weight_sums.append(sum(updated_action_weights))

            concurrency = [1]
            # concurrency += random.sample(range(2, max_concurrency + 1), 9)
            for concurrency_idx, concurrency_level in enumerate(concurrency):
                action_weights_list = [updated_action_weights] * concurrency_level
                current_file_idx = file_idx * len(target_utilizations) * len(concurrency) + target_util_idx * len(concurrency) + concurrency_idx
                write_to_hdf5([action_weights_list], f"../execution_plans/plan_{current_file_idx:06d}.h5")
                # write_to_csv(action_weights_list, f"../execution_plans/plan_{current_file_idx:06d}.csv")

    plt.figure()
    plt.hist(action_weight_sums, bins=20, edgecolor='black')
    plt.xlabel('Sum of Action Weights')
    plt.ylabel('Frequency')
    plt.title('Histogram of Action Weight Sums')
    plt.grid(True)
    plt.savefig('action_weight_sums_histogram.png')

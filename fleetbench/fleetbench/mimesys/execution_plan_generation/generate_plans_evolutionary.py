import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsRegressor

def write_to_csv(action_weights, file_path):
    with open(file_path, 'w') as f:
        for row in action_weights:
            f.write(','.join(map(str, row)) + '\n')

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

# Step 1: Define the black-box metric function f(r)
def dummy_metric_fn(r):
    """
    A fake metric function that outputs a 3D metric vector.
    Replace this with your actual profiling function.
    """
    mean = np.mean(r)
    var = np.var(r)
    entropy = -np.sum(r * np.log(r + 1e-8))  # avoid log(0)
    return np.array([mean, var, entropy])


# Step 3: Diversity-aware full archive exploration
def full_archive_diverse_sampling(metric_fn, dim_ratio, total_rounds, candidates_per_round):
    archive_r = []
    archive_m = []

    for round_num in range(total_rounds):
        candidates_r = sample_simplex_direct(candidates_per_round, dim_ratio)
        candidates_m = [metric_fn(r) for r in candidates_r]

        archive_r.extend(candidates_r)
        archive_m.extend(candidates_m)

    return np.array(archive_r), np.array(archive_m)

if __name__ == "__main__":
    # ---- Configurable settings ----
    num_functions = 23                # Dimensionality of ratio vectors
    metric_dim = 4                   # Dimensionality of metric space
    num_initial_samples = 10
    num_candidates = 100
    samples_per_round = 5
    num_rounds = 3

    # Run the search
    num_samples = 50
    num_rounds = 100

    plan_path = "../execution_plans"
    archive_path = "/home/dhkim/results"
    np.random.seed(42)

    for _ in range(num_rounds):
        num_files = len([f for f in os.listdir(archive_path) if os.path.isfile(os.path.join(archive_path, f))])
        print(f"Number of files in {archive_path}: {num_files}")

        starting_idx = num_files

        # Load all files in plan_path as existing_ratios
        existing_ratios = []
        for fname in os.listdir(plan_path):
            fpath = os.path.join(plan_path, fname)
            if os.path.isfile(fpath):
                with open(fpath, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        values = list(map(float, line.split(',')))
                        if len(values) == num_functions:
                            existing_ratios.append(values)

        plan_metric_pairs = []
        # For each plan file, find the corresponding metrics file
        for fname in os.listdir(plan_path):
            if fname.startswith("plan_") and fname.endswith(".csv"):
                plan_idx = fname[len("plan_"):-len(".csv")]
                metric_fname = f"stats-plan_{plan_idx}.txt"
                metric_fpath = os.path.join(archive_path, metric_fname)
                plan_fpath = os.path.join(plan_path, fname)
                if os.path.isfile(metric_fpath):
                    plan_metric_pairs.append((plan_fpath, metric_fpath))

        existing_ratios = np.array(existing_ratios)

        if starting_idx == 0:
            for file_idx in range(num_samples):
                if file_idx < num_functions:
                    action_weights = [1.0 if i == file_idx else 0.0 for i in range(num_functions)]
                else:
                    action_weights = sample_simplex_direct(1, num_functions)[0]
        else:


    # ---- Simulated "profile" function ----
    def profile_function_ratios(ratios):
        # Nonlinear transform + noise (replace with real profiler)
        return np.sin(2 * np.pi * ratios[:, :metric_dim]) + 0.1 * np.random.randn(len(ratios), metric_dim)

    # ---- Initial samples ----
    existing_ratios = np.random.dirichlet(alpha=[1]*num_functions, size=num_initial_samples)
    existing_metrics = profile_function_ratios(existing_ratios)

    # ---- Greedy Farthest-Point Sampling ----
    for round_i in range(num_rounds):
        # Fit KNN regressor
        knn_regressor = KNeighborsRegressor(n_neighbors=5, weights='distance')
        knn_regressor.fit(existing_ratios, existing_metrics)

        # Generate candidate ratios (by perturbing existing samples)
        candidates = []
        while len(candidates) < num_candidates:
            base = existing_ratios[np.random.choice(len(existing_ratios))]
            perturb = np.random.normal(scale=0.1, size=base.shape)
            candidate = base + perturb
            candidate = np.clip(candidate, 0, None)
            if candidate.sum() == 0:
                candidate = np.ones_like(candidate) / len(candidate)
            else:
                candidate /= candidate.sum()
            candidates.append(candidate)
        candidates = np.array(candidates)

        # Predict metrics for candidates
        predicted_metrics = knn_regressor.predict(candidates)

        # Compute each candidate's minimum Euclidean distance to all existing metrics
        dist_matrix = np.linalg.norm(predicted_metrics[:, None, :] - existing_metrics[None, :, :], axis=2)
        min_dists = dist_matrix.min(axis=1)

        # Select candidates that are farthest from all existing metrics
        top_indices = np.argsort(min_dists)[-samples_per_round:]
        selected_candidates = candidates[top_indices]

        # Profile true metrics for selected candidates
        true_metrics = profile_function_ratios(selected_candidates)

        # Update the stored ratios and metrics
        existing_ratios = np.vstack([existing_ratios, selected_candidates])
        existing_metrics = np.vstack([existing_metrics, true_metrics])

        print(f"Round {round_i+1}, Added {samples_per_round} samples, "
            f"Mean min-distance: {min_dists[top_indices].mean():.3f}")

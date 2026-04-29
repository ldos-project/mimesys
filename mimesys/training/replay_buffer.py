import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self.log_probs = []
        self.final_states = []
        self.device = device

    def add_to_buffer(self, state, action, reward, log_probs, final_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_probs)
        self.final_states.append(final_state)

    def set_advantages(self, advantages):
        self.advantages = advantages

    def stack_dictionaries_list(self, dicts_list):
        """
        Stacks tensor values across a list of dictionaries by recursively stacking
        values for each key across all dictionaries in the list.
        """
        # Initialize the result dictionary
        stacked_dict = {}

        # Get the keys from the first dictionary (assuming all dicts have the same structure)
        keys = dicts_list[0].keys()

        for key in keys:
            values = [d[key] for d in dicts_list]
            if isinstance(values[0], dict):
                # Recursively handle nested dictionaries
                stacked_dict[key] = self.stack_dictionaries_list(values)

            elif isinstance(values[0], torch.Tensor):
                # Stack tensors along a new dimension
                stacked_dict[key] = torch.stack(values).to(self.device)
            elif isinstance(values[0], list) and isinstance(values[0][0], torch.Tensor):
                # Recursively handle nested dictionaries
                stacked_dict[key] = [torch.stack(o) for o in zip(*values)]
            else:
                # If values are not tensors, you can customize this part as needed
                stacked_dict[key] = values

        return stacked_dict

    def helper_build_states(self, states):
        # assume states are of the same form
        xts = []
        ts = []
        diffusion_context_cond_list = []
        prev_trace_list = []
        for state in states:
            xt, t, diffusion_context_cond, prev_trace = state
            xts.append(xt)
            ts.append(t)
            diffusion_context_cond_list.append(diffusion_context_cond)
            prev_trace_list.append(prev_trace)

        xts = torch.stack(xts).to(self.device)
        ts = torch.stack(ts).to(self.device)
        diffusion_context_cond_tensor = torch.stack(diffusion_context_cond_list).to(self.device)
        prev_trace_tensor = torch.stack(prev_trace_list).to(self.device)
        return (xts, ts, diffusion_context_cond_tensor, prev_trace_tensor)

    def collate(self, sample_batch):
        states_tensor = self.helper_build_states(sample_batch["states"])
        actions_tensor = torch.stack(sample_batch["actions"]).to(self.device)
        log_probs_tensor = torch.stack(sample_batch["log_probs"]).to(self.device)
        advantages_tensor = torch.stack(sample_batch["advantages"]).to(self.device)
        final_states_tensor = torch.stack(sample_batch["final_states"]).to(self.device)
        return {
            "states": states_tensor,
            "actions": actions_tensor,
            "log_probs": log_probs_tensor,
            "advantages": advantages_tensor,
            "final_states": final_states_tensor
        }

    def sample(self):
        assert len(self.states) == len(self.actions) == len(self.rewards) == len(self.log_probs) == len(self.advantages)
        NT = len(self.states)
        batch_indices = np.arange(NT)
        np.random.shuffle(batch_indices)
        batch_indices = batch_indices[: (NT // self.batch_size) * self.batch_size] # truncate
        batch_indices = batch_indices.reshape(-1, self.batch_size)

        samples = []
        for batch_idx in batch_indices:
            states = [self.states[i] for i in batch_idx]
            actions = [self.actions[i] for i in batch_idx]
            rewards = [self.rewards[i] for i in batch_idx]
            log_probs = [self.log_probs[i] for i in batch_idx]
            advantages = [self.advantages[i] for i in batch_idx]
            final_states = [self.final_states[i] for i in batch_idx]

            samples.append({
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "log_probs": log_probs,
                "advantages": advantages,
                "final_states": final_states
            })
        return samples

    def __len__(self):
        return len(self.states) // self.batch_size

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self.log_probs = []
        self.final_states = []

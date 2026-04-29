from __future__ import annotations

from dataclasses import dataclass

import h5py

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FleetBenchAction:
    weights: list[list[list[float]]]

    @classmethod
    def from_action_file(cls, action_file: str):
        with h5py.File(action_file, 'r') as f:
            action_weights = f['execution_plan'][:]
            actions = action_weights.tolist()
        return cls(weights=actions)

    def to_2d_list(self, num_max_threads: int = 20, transpose: bool = False) -> list[list[float]]:
        num_actions = len(self.weights[0][0])
        zero_weights = [0.0] * (num_actions)

        padded_weights_3d = []
        for w in self.weights:
            padded_weights = w.copy()
            while len(padded_weights) < num_max_threads:
                padded_weights.append(zero_weights)
            if transpose:
                padded_weights = list(map(list, zip(*padded_weights)))
            padded_weights_3d.append(padded_weights)

        return padded_weights_3d

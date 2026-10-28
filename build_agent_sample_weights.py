"""
Stockade -- precompute per-scene-frame sampling weights for the agent-5/4
data-scarcity follow-up experiment.

agent5 has real (non-placeholder) data in only 21.2% of training scenes,
agent4 in 48.8% -- confirmed in diagnose_agent5.py. Rather than surgically
reweighting the loss function internals (risk of silently breaking
corner_loss's masking/flattening logic, which would be hard to verify
correctly in the time available), this oversamples scene-frames where
those agents have real data via a WeightedRandomSampler, so the model
sees their sparse real signal more often per epoch.

Usage (as a library):
    from build_agent_sample_weights import compute_weights
    weights = compute_weights(dataset_root, num_samples)
"""

import numpy as np

AGENT4_WEIGHT_IF_REAL = 1.5
AGENT5_WEIGHT_IF_REAL = 3.0


def is_real_sample(filepath):
    d = np.load(filepath, allow_pickle=True)
    return not (d.shape == () and d == 0)


def compute_weights(train_root):
    """Returns a numpy array of per-index sampling weights, same length/order
    as the V2XSimDet dataset's indexing (agent1's sorted seq_files order)."""
    import os

    agent4_dir = f"{train_root}/agent4"
    agent5_dir = f"{train_root}/agent5"

    # Match V2XSimDet's own index ordering: sort scene dirs by (scene, frame).
    def sorted_dirs(root):
        names = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        names.sort(key=lambda x: tuple(int(p) for p in x.split("_")))
        return names

    dirs4 = sorted_dirs(agent4_dir)
    dirs5 = sorted_dirs(agent5_dir)
    assert len(dirs4) == len(dirs5), "agent4/agent5 index count mismatch"

    weights = np.ones(len(dirs4), dtype=np.float64)
    n_agent4_real = 0
    n_agent5_real = 0
    for i, (d4, d5) in enumerate(zip(dirs4, dirs5)):
        f4 = os.path.join(agent4_dir, d4, "0.npy")
        f5 = os.path.join(agent5_dir, d5, "0.npy")
        agent4_real = is_real_sample(f4)
        agent5_real = is_real_sample(f5)
        if agent4_real:
            n_agent4_real += 1
            weights[i] = max(weights[i], AGENT4_WEIGHT_IF_REAL)
        if agent5_real:
            n_agent5_real += 1
            weights[i] = max(weights[i], AGENT5_WEIGHT_IF_REAL)

    print(f"agent4 real: {n_agent4_real}/{len(dirs4)} ({100*n_agent4_real/len(dirs4):.1f}%)")
    print(f"agent5 real: {n_agent5_real}/{len(dirs5)} ({100*n_agent5_real/len(dirs5):.1f}%)")
    print(f"weight distribution: min={weights.min()}, max={weights.max()}, mean={weights.mean():.3f}")
    return weights


if __name__ == "__main__":
    w = compute_weights("data/V2X-Sim-det-001/V2X-Sim-det/train")
    np.save("agent_sample_weights.npy", w)
    print("Saved to agent_sample_weights.npy")

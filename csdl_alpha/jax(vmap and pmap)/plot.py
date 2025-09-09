import numpy as np
import matplotlib.pyplot as plt

def plot(batch_sizes, times_vmap_s, times_pmap_s):
    bs = np.asarray(batch_sizes, dtype=int)
    tv = np.asarray(times_vmap_s, dtype=float)
    tp = np.asarray(times_pmap_s, dtype=float)

    plt.figure(figsize=(7,4))
    plt.plot(bs, tv, marker="o", label="vmap")
    plt.plot(bs, tp, marker="o", label="pmap")
    plt.grid(True)
    plt.xlabel("Batch size")
    plt.ylabel("Time (s)")
    plt.title("Vmap vs Pmap (TPU)")
    plt.legend()
    plt.tight_layout()
    plt.show()

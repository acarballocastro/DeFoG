import os
import math
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
#  PLOTTING UTILITIES
# ============================================================

class AttentionPlotter:
    """Handles consistent plotting for attention maps, rollout, and flow."""
    
    def __init__(self, cmap="viridis", dpi=150, max_cols=4, show_axis=False):
        self.cmap = cmap
        self.dpi = dpi
        self.max_cols = max_cols
        self.show_axis = show_axis

    def save_heatmap(self, data, filename, title="", vmin=None, vmax=None):
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            data, cmap=self.cmap, cbar=True, vmin=vmin, vmax=vmax
        )
        plt.title(title)
        if not self.show_axis:
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename, dpi=self.dpi)
        plt.close()

    def grid_plot(self, heads, out_path, title, share_clim=True):
        """Plot multiple heads on a grid."""
        H, N, _ = heads.shape
        ncols = min(self.max_cols, H)
        nrows = math.ceil(H / ncols)
        vmin, vmax = (heads.min(), heads.max()) if share_clim else (None, None)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        axes = axes.flatten()

        for h in range(len(axes)):
            ax = axes[h]
            if h < H:
                sns.heatmap(heads[h], cmap=self.cmap, vmin=vmin, vmax=vmax, ax=ax, cbar=False)
                ax.set_title(f"H{h+1}")
                if not self.show_axis:
                    ax.axis("off")
            else:
                ax.axis("off")

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=self.dpi)
        plt.close()


# ============================================================
#  CORE ATTENTION ANALYZER
# ============================================================

class AttentionAnalyzer:
    """Main interface for analyzing and visualizing attention in graph models."""

    def __init__(self, out_dir="./attn_analysis", residual=True, threshold=1e-4, device="cpu"):
        self.out_dir = out_dir
        self.residual = residual
        self.threshold = threshold
        self.plotter = AttentionPlotter()
        self.device = device

    # ------------------------------------------------------------
    # 1. ATTENTION MAPS
    # ------------------------------------------------------------
    def plot_maps(self, attn, mode="grid"):
        """Visualize raw attention maps (B, L, H, N, N)."""
        B, L, H, N, N2 = attn.shape
        assert N == N2, "attention must be square"

        for b in range(B):
            batch_dir = os.path.join(self.out_dir, f"batch_{b}", "maps")
            os.makedirs(batch_dir, exist_ok=True)
            for l in range(L):
                heads = attn[b, l].detach().cpu().numpy()
                title = f"Batch {b}, Layer {l+1}"
                out_path = os.path.join(batch_dir, f"layer{l+1}_grid.png")
                self.plotter.grid_plot(heads, out_path, title)

    # ------------------------------------------------------------
    # 2. ATTENTION ROLLOUT
    # ------------------------------------------------------------
    def compute_rollout(self, attn):
        """Compute attention rollout per batch."""
        B, L, H, N, N2 = attn.shape
        attn = attn.mean(dim=2)
        I = torch.eye(N, device=attn.device)
        rollout = torch.zeros((B, L, N, N), device=attn.device)

        for b in range(B):
            result = torch.eye(N, device=attn.device)
            for l in range(L):
                A = attn[b, l]
                if self.residual:
                    A = (A + I) / 2
                A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
                result = A @ result
                rollout[b, l] = result
        return rollout

    def plot_rollout(self, attn, max_nodes_to_plot=6):
        rollout = self.compute_rollout(attn)
        B, L, N, _ = rollout.shape

        for b in range(B):
            batch_dir = os.path.join(self.out_dir, f"batch_{b}", "rollout")
            os.makedirs(batch_dir, exist_ok=True)
            node_indices = range(min(N, max_nodes_to_plot))
            for node in node_indices:
                self.plotter.save_heatmap(
                    rollout[b, :, node, :].cpu().numpy(),
                    os.path.join(batch_dir, f"node{node}_rollout.png"),
                    title=f"Batch {b}: Node {node} rollout"
                )

            self.plotter.save_heatmap(
                rollout[b, -1].cpu().numpy(),
                os.path.join(batch_dir, "rollout_final.png"),
                title=f"Batch {b}: Final rollout"
            )

    # ------------------------------------------------------------
    # 3. ATTENTION FLOW
    # ------------------------------------------------------------
    def compute_flow(self, attn, per_layer_plot=False, max_nodes_to_plot=6):
        """Compute graph-based flow using max-flow."""
        B, L, H, N, N2 = attn.shape
        attn = attn.mean(dim=2)
        I = np.eye(N)
        flows = torch.zeros((B, N, N))
        for b in range(B):
            G = nx.DiGraph()
            for l in range(L):
                for n in range(N):
                    G.add_node((l, n))
            for l in range(1, L):
                A = attn[b, l].detach().cpu().numpy()
                if self.residual:
                    A = 0.5 * (A + I)
                A = A / (A.sum(axis=-1, keepdims=True) + 1e-8)
                A[A < self.threshold] = 0
                for i in range(N):
                    for j in np.where(A[i] > 0)[0]:
                        G.add_edge((l, i), (l-1, j), capacity=float(A[i, j]))

            for src in range(N):
                for tgt in range(N):
                    try:
                        val = nx.maximum_flow_value(G, (L-1, src), (0, tgt), capacity='capacity')
                    except nx.NetworkXError:
                        val = 0.0
                    flows[b, tgt, src] = val
            flows[b] /= (flows[b].sum(dim=-1, keepdim=True) + 1e-8)
        return flows

    def plot_flow(self, attn):
        flows = self.compute_flow(attn)
        B = flows.shape[0]
        for b in range(B):
            batch_dir = os.path.join(self.out_dir, f"batch_{b}", "flow")
            os.makedirs(batch_dir, exist_ok=True)
            self.plotter.save_heatmap(
                flows[b].cpu().numpy(),
                os.path.join(batch_dir, "flow_final.png"),
                title=f"Batch {b}: Final flow"
            )


# ============================================================
#  TEMPORAL DYNAMICS WRAPPER
# ============================================================

class AttentionDynamics:
    """
    Stores and compares attention analysis across timesteps
    (useful for graph generative models with diffusion or autoregressive time steps).
    """
    def __init__(self, analyzer: AttentionAnalyzer):
        self.analyzer = analyzer
        self.timesteps = {}

    def add_timestep(self, t: int, attn: torch.Tensor):
        """Add attention tensor for a specific timestep."""
        self.timesteps[t] = attn

    def analyze_all(self, methods=("maps", "rollout", "flow")):
        """Run chosen analyses for all stored timesteps."""
        for t, attn in self.timesteps.items():
            print(f"Analyzing timestep {t}...")
            t_dir = os.path.join(self.analyzer.out_dir, f"timestep_{t}")
            os.makedirs(t_dir, exist_ok=True)
            analyzer = AttentionAnalyzer(out_dir=t_dir)
            if "maps" in methods:
                analyzer.plot_maps(attn)
            if "rollout" in methods:
                analyzer.plot_rollout(attn)
            if "flow" in methods:
                analyzer.plot_flow(attn)

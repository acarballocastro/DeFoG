# attn_analysis.py
import os
import math
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

sns.set(style="whitegrid")


# -------------------------
# Utilities
# -------------------------
def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_rows_np(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = mat.sum(axis=-1, keepdims=True)
    s[s == 0] = 1.0
    return mat / (s + eps)


def _ensure_batch_shape(attn: torch.Tensor) -> torch.Tensor:
    """
    Ensure attn is torch.Tensor and shape (B, L, H, N, N).
    """
    if not torch.is_tensor(attn):
        attn = torch.tensor(attn)
    if attn.ndim != 5:
        raise ValueError("attn must be 5D tensor with shape (B, L, H, N, N)")
    return attn


# -------------------------
# Plotter
# -------------------------
class AttentionPlotter:
    def __init__(self, cmap: str = "viridis", dpi: int = 150, max_cols: int = 4, show_axis: bool = False):
        self.cmap = cmap
        self.dpi = dpi
        self.max_cols = max_cols
        self.show_axis = show_axis

    def save_heatmap(self, mat: np.ndarray, path: str, title: Optional[str] = None, vmin: float = None, vmax: float = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure(figsize=(6, 4))
        sns.heatmap(mat, cmap=self.cmap, vmin=vmin, vmax=vmax)
        if title:
            plt.title(title)
        if not self.show_axis:
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

    def grid_heads(self, heads: np.ndarray, path: str, title: Optional[str] = None, share_clim: bool = True):
        """
        heads: (H, N, N)
        """
        H, N, _ = heads.shape
        ncols = min(self.max_cols, H)
        nrows = math.ceil(H / ncols)
        vmin = float(heads.min()) if share_clim else None
        vmax = float(heads.max()) if share_clim else None

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        axes_flat = axes.flatten()
        for idx in range(nrows * ncols):
            ax = axes_flat[idx]
            if idx < H:
                sns.heatmap(heads[idx], cmap=self.cmap, vmin=vmin, vmax=vmax, ax=ax, cbar=False)
                ax.set_title(f"H{idx+1}")
                if not self.show_axis:
                    ax.axis("off")
            else:
                ax.axis("off")
        if title:
            fig.suptitle(title)
        # single colorbar
        mpl_im = axes_flat[0].collections[0]
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(mpl_im, cax=cbar_ax)
        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

    def bar_tokenrank_multilayer(self, tokenranks: List[np.ndarray], path: str, batch_idx: int = 0):
        """
        tokenranks: list length L, each is (N,)
        """
        L = len(tokenranks)
        N = tokenranks[0].shape[0]
        fig, axes = plt.subplots(L, 1, figsize=(10, 2.2 * L), sharex=True)
        if L == 1:
            axes = [axes]
        ymax = max(tr.max() for tr in tokenranks) * 1.05
        for l, ax in enumerate(axes):
            ax.bar(np.arange(N), tokenranks[l], color="royalblue")
            ax.set_ylabel(f"L{l+1}", fontsize=9)
            ax.set_ylim(0, ymax)
            ax.grid(alpha=0.15)
            if not self.show_axis:
                ax.set_xticks([])
        axes[-1].set_xlabel("Node index")
        fig.suptitle(f"TokenRank per layer (batch {batch_idx})", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

    def heatmap_tokenrank(self, tokenranks: List[np.ndarray], path: str, batch_idx: int = 0):
        # stacked heatmap L x N
        mat = np.stack(tokenranks, axis=0)
        plt.figure(figsize=(10, 0.6 * mat.shape[0]))
        sns.heatmap(mat, cmap=self.cmap, cbar=True)
        plt.xlabel("Node index")
        plt.ylabel("Layer")
        plt.title(f"TokenRank heatmap (batch {batch_idx})")
        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close()


# -------------------------
# AttentionAnalyzer class
# -------------------------
class AttentionAnalyzer:
    """
    Main analyzer that contains:
     - plot_maps
     - compute/plot_rollout
     - compute/plot_flow (exact via networkx)
     - build_attention_graphs (AAgg)
     - DTMC analysis: multi-bounce, TokenRank, lambda2
     - fast_attention_flow approx
    """

    def __init__(self, out_dir: str = "./attn_analysis", residual: bool = True, threshold: float = 1e-6, device: str = "cpu"):
        self.out_dir = out_dir
        self.residual = residual
        self.threshold = threshold
        self.device = device
        self.plotter = AttentionPlotter()
        os.makedirs(self.out_dir, exist_ok=True)

    # -------------------------
    # Raw maps plotting (B, L, H, N, N)
    # -------------------------
    def plot_attention_maps(self, attn: torch.Tensor, mode: str = "grid", share_clim: bool = True,
                            node_mask: Optional[np.ndarray] = None):
        attn = _ensure_batch_shape(attn).to(self.device)
        B, L, H, N, N2 = attn.shape
        assert N == N2
        valid_mask = None
        if node_mask is not None:
            node_mask = np.asarray(node_mask).astype(bool)
            if node_mask.shape[0] != N:
                raise ValueError("node_mask length != N")
            valid_mask = np.outer(node_mask, node_mask).astype(float)

        for b in range(B):
            batch_dir = os.path.join(self.out_dir, f"timestep_batch_batch{b}/maps")
            os.makedirs(batch_dir, exist_ok=True)
            for l in range(L):
                heads = _to_numpy(attn[b, l])  # (H,N,N)
                if valid_mask is not None:
                    heads = heads * valid_mask
                title = f"Batch {b}, Layer {l+1}"
                path_grid = os.path.join(batch_dir, f"layer{l+1}_all_heads_grid.png")
                self.plotter.grid_heads(heads, path_grid, title=title, share_clim=share_clim)

                # mean
                mean_mat = heads.mean(axis=0)
                p = os.path.join(batch_dir, f"layer{l+1}_heads_mean.png")
                self.plotter.save_heatmap(mean_mat, p, title=f"{title} - heads mean")

    # -------------------------
    # Rollout (B, L, H, N, N) -> returns (B, L, N, N)
    # -------------------------
    def compute_rollout(self, attn: torch.Tensor) -> torch.Tensor:
        attn = _ensure_batch_shape(attn).to(self.device)
        B, L, H, N, _ = attn.shape
        attn_mean = attn.mean(dim=2)  # (B,L,N,N)
        I = torch.eye(N, device=attn.device)
        rollout = torch.zeros((B, L, N, N), device=attn.device)
        for b in range(B):
            result = torch.eye(N, device=attn.device)
            for l in range(L):
                A = attn_mean[b, l]
                if self.residual:
                    A = 0.5 * (A + I)
                A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
                result = A @ result
                rollout[b, l] = result
        return rollout

    def plot_rollout(self, attn: torch.Tensor, max_nodes_to_plot: int = 6):
        rollout = self.compute_rollout(attn)
        B, L, N, _ = rollout.shape
        for b in range(B):
            batch_dir = os.path.join(self.out_dir, f"timestep_batch_batch{b}/rollout")
            os.makedirs(batch_dir, exist_ok=True)
            node_indices = range(min(N, max_nodes_to_plot))
            for node in node_indices:
                mat = rollout[b, :, node, :].detach().cpu().numpy()  # (L, N)
                p = os.path.join(batch_dir, f"node{node}_rollout.png")
                self.plotter.save_heatmap(mat, p, title=f"Batch {b}: Node {node} rollout")
            pfinal = os.path.join(batch_dir, "rollout_final.png")
            self.plotter.save_heatmap(rollout[b, -1].detach().cpu().numpy(), pfinal, title=f"Batch {b}: Final rollout")

    # -------------------------
    # Exact attention flow (max-flow) using networkx
    # -------------------------
    def compute_flow(self, attn: torch.Tensor, per_layer_plot: bool = False,
                     max_nodes_to_plot: int = 6) -> torch.Tensor:
        attn = _ensure_batch_shape(attn).to(self.device)
        B, L, H, N, _ = attn.shape
        attn_mean = attn.mean(dim=2)  # (B,L,N,N)
        flows = torch.zeros((B, N, N))
        I = np.eye(N)
        for b in range(B):
            # build DAG with nodes (layer, node)
            G = nx.DiGraph()
            for l in range(L):
                for n in range(N):
                    G.add_node((l, n))
            # add edges
            for l in range(1, L):
                A = _to_numpy(attn_mean[b, l])
                if self.residual:
                    A = 0.5 * (A + I)
                A = _normalize_rows_np(A)
                A[A < self.threshold] = 0
                for i in range(N):
                    nz = np.where(A[i] > 0)[0]
                    for j in nz:
                        G.add_edge((l, i), (l - 1, j), capacity=float(A[i, j]))
            # optional per-layer flows
            if per_layer_plot:
                node_indices = range(min(N, max_nodes_to_plot))
                layer_flows = np.zeros((L, N, N))
                for l in range(L):
                    for src in node_indices:
                        for tgt in range(N):
                            try:
                                val = nx.maximum_flow_value(G, (l, src), (0, tgt), capacity="capacity")
                            except nx.NetworkXError:
                                val = 0.0
                            layer_flows[l, tgt, src] = val
                # save per-layer flow heatmaps for plotted nodes
                batch_dir = os.path.join(self.out_dir, f"timestep_batch_batch{b}/flow")
                os.makedirs(batch_dir, exist_ok=True)
                for node in node_indices:
                    p = os.path.join(batch_dir, f"node{node}_flow_per_layer.png")
                    self.plotter.save_heatmap(layer_flows[:, :, node], p, title=f"Batch {b} node {node} flow per layer")

            # final flows: from (L-1,src) to (0,tgt)
            for src in range(N):
                for tgt in range(N):
                    try:
                        val = nx.maximum_flow_value(G, (L - 1, src), (0, tgt), capacity="capacity")
                    except nx.NetworkXError:
                        val = 0.0
                    flows[b, tgt, src] = float(val)
            # normalize per source to sum to 1 (makes comparable)
            denom = flows[b].sum(dim=-1, keepdim=True)
            denom[denom == 0] = 1.0
            flows[b] = flows[b] / denom
        return flows

    def plot_flow(self, attn: torch.Tensor, per_layer_plot: bool = False, max_nodes_to_plot: int = 6):
        flows = self.compute_flow(attn, per_layer_plot=per_layer_plot, max_nodes_to_plot=max_nodes_to_plot)
        B = flows.shape[0]
        for b in range(B):
            batch_dir = os.path.join(self.out_dir, f"timestep_batch_batch{b}/flow")
            os.makedirs(batch_dir, exist_ok=True)
            p = os.path.join(batch_dir, "flow_final.png")
            self.plotter.save_heatmap(flows[b].detach().cpu().numpy(), p, title=f"Batch {b} final flow")

    # -------------------------
    # Fast approximate flow (min-max propagation) - very fast
    # -------------------------
    def compute_flow_approx(self, attn: torch.Tensor, per_layer_plot: bool = False, max_nodes_to_plot: int = 6) -> torch.Tensor:
        """
        Fast approximation of attention flow using min-over-edges (bottleneck) + max-over-paths trick.
        Complexity O(B * L * N^2) and returns flows shape (B, N, N).
        """
        attn = _ensure_batch_shape(attn).to(self.device)
        B, L, H, N, _ = attn.shape
        attn_mean = attn.mean(dim=2)
        I = torch.eye(N, device=attn.device)
        flows = torch.zeros((B, N, N), device=attn.device)

        for b in range(B):
            # result matrix: initialize as identity (node -> itself)
            result = I.clone()  # (N, N) where row i is distribution from i
            per_layer_results = []
            for l in range(L):
                A = attn_mean[b, l]
                if self.residual:
                    A = 0.5 * (A + I)
                A = A / (A.sum(dim=-1, keepdim=True) + 1e-12)
                # approximate bottleneck propagation:
                # new_result[j,k] = max_i min( A[j,i], result[i,k] )
                # using broadcasting: A shape (N,N), result (N,N)
                # compute min along intermediate i, then max over i
                # implement with numpy for speed on CPU if needed
                A_np = _to_numpy(A)
                R_np = _to_numpy(result)
                # min along edges: shape (N, N, N) -> heavy but manageable for N~100
                # optimized: compute pairwise min via broadcasting then max
                tmp = np.minimum(A_np[:, :, None], R_np[None, :, :])  # (N, N, N) -> memory O(N^3)
                new_res = tmp.max(axis=1)  # (N, N)
                result = torch.tensor(new_res, device=attn.device, dtype=attn.dtype)
                per_layer_results.append(result.cpu().numpy())
            flows[b] = result  # final approx flow
            if per_layer_plot:
                # save per-layer images for a few nodes
                batch_dir = os.path.join(self.out_dir, f"timestep_batch_batch{b}/flow_approx")
                os.makedirs(batch_dir, exist_ok=True)
                node_indices = range(min(N, max_nodes_to_plot))
                per_layer_results = np.stack(per_layer_results, axis=0)  # (L,N,N)
                for node in node_indices:
                    p = os.path.join(batch_dir, f"node{node}_flow_approx_per_layer.png")
                    self.plotter.save_heatmap(per_layer_results[:, :, node], p, title=f"Batch {b} node {node} approx flow per layer")
                pfinal = os.path.join(batch_dir, "flow_approx_final.png")
                self.plotter.save_heatmap(flows[b].cpu().numpy(), pfinal, title=f"Batch {b} approx flow final")

        return flows

    # -------------------------
    # Attention Graphs (head agg + layer aggregation)
    # -------------------------
    def build_attention_graphs(self, attn: torch.Tensor, agg_heads: str = "mean", agg_layers: str = "multiply",
                               top_k: Optional[int] = None, threshold: Optional[float] = None,
                               node_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[nx.DiGraph]]:
        """
        Return AAgg_per_batch (B,N,N) plus list of nx graphs.
        agg_heads: 'mean'|'max'|'median'
        agg_layers: 'multiply'|'rollout'|'mean'|'sum'
        """
        attn = _ensure_batch_shape(attn).to(self.device)
        B, L, H, N, _ = attn.shape
        arr = _to_numpy(attn)  # (B,L,H,N,N)
        # aggregate heads
        if agg_heads == "mean":
            A_layer = arr.mean(axis=2)  # (B,L,N,N)
        elif agg_heads == "max":
            A_layer = arr.max(axis=2)
        elif agg_heads == "median":
            A_layer = np.median(arr, axis=2)
        else:
            raise ValueError("unknown agg_heads")

        # apply node mask
        if node_mask is not None:
            nm = np.asarray(node_mask).astype(bool)
            if nm.shape[0] != N:
                raise ValueError("node_mask length mismatch")
            mask2d = np.outer(nm, nm).astype(float)
            A_layer = A_layer * mask2d[None, None, :, :]

        # residual + normalize + prune per layer
        I = np.eye(N, dtype=float)
        A_proc = np.zeros_like(A_layer)
        for b in range(B):
            for l in range(L):
                W = A_layer[b, l].astype(float)
                W = _normalize_rows_np(W)
                if self.residual:
                    A = W + I
                    A = _normalize_rows_np(A)
                else:
                    A = W
                if threshold is not None:
                    A[A < threshold] = 0.0
                if top_k is not None and top_k < N:
                    mask = np.zeros_like(A, dtype=bool)
                    for i in range(N):
                        k = min(top_k, N)
                        idx = np.argpartition(-A[i], k-1)[:k]
                        mask[i, idx] = True
                    A = A * mask.astype(float)
                    A = _normalize_rows_np(A)
                A_proc[b, l] = A

        AAgg_per_batch = np.zeros((B, N, N), dtype=float)
        graphs = []
        if agg_layers in ("multiply", "rollout"):
            for b in range(B):
                result = np.eye(N, dtype=float)
                for l in range(L):
                    result = A_proc[b, l] @ result
                AAgg_per_batch[b] = result
        elif agg_layers == "mean":
            AAgg_per_batch = A_proc.mean(axis=1)
        elif agg_layers == "sum":
            AAgg_per_batch = A_proc.sum(axis=1)
        else:
            raise ValueError("unknown agg_layers")

        # build nx graphs
        for b in range(B):
            G = nx.DiGraph()
            for i in range(N):
                if node_mask is None or node_mask[i]:
                    G.add_node(i)
            M = AAgg_per_batch[b]
            for i in range(N):
                for j in range(N):
                    w = float(M[i, j])
                    if w > 0:
                        G.add_edge(j, i, weight=w)  # j -> i (source j to target i)
            graphs.append(G)
        return AAgg_per_batch, graphs

    def plot_attention_graphs(self, AAgg_per_batch: np.ndarray):
        B, N, _ = AAgg_per_batch.shape
        for b in range(B):
            p = os.path.join(self.out_dir, f"timestep_batch_batch{b}/attngraph_heatmap.png")
            self.plotter.save_heatmap(AAgg_per_batch[b], p, title=f"Batch {b} Attention Graph")

    # -------------------------
    # DTMC-based analysis: Multi-bounce, TokenRank, lambda2 weighting
    # -------------------------
    def dtmc_analysis(self, attn: torch.Tensor, alpha: float = 0.9, tol: float = 1e-8, max_iter: int = 500,
                      compute_tokenrank: bool = True, multi_bounce_steps: int = 5, plot: bool = True,
                      max_nodes_to_plot: int = 6) -> Dict[str, Any]:
        """
        attn: (B, L, H, N, N)
        Returns dict with keys:
          - tokenrank_per_batch: List[List[np.ndarray]] (B lists of length L)
          - lambda2_per_batch: List[List[float]]
          - multi_bounce_per_batch: List[List[List[np.ndarray]]] (B x L x steps x N)
        """
        attn = _ensure_batch_shape(attn).to(self.device)
        B, L, H, N, _ = attn.shape
        attn_mean = attn.mean(dim=2)  # (B,L,N,N)
        results = {
            "tokenrank": [[] for _ in range(B)],
            "lambda2": [[] for _ in range(B)],
            "multi_bounce": [[] for _ in range(B)],
        }

        for b in range(B):
            tokenranks_b = []
            lambda2_b = []
            multi_bounce_b = []
            for l in range(L):
                A = attn_mean[b, l].detach().cpu().numpy()
                A = _normalize_rows_np(A)
                # teleportation for irreducibility
                P = alpha * A + (1.0 - alpha) / N * np.ones_like(A)

                # eigenvalues for lambda2: compute eigenvalues of P^T since stationary is left eigenvector
                try:
                    eigvals = LA.eigvals(torch.tensor(P.T))
                    eigs = np.asarray(eigvals.cpu().numpy()).real
                    eigs_sorted = np.sort(eigs)[::-1]
                    lambda2 = float(eigs_sorted[1]) if eigs_sorted.size > 1 else 0.0
                except Exception:
                    # fallback to numpy
                    eigs = np.linalg.eigvals(P.T)
                    eigs_sorted = np.sort(eigs)[::-1]
                    lambda2 = float(np.real(eigs_sorted[1])) if eigs_sorted.size > 1 else 0.0

                # TokenRank via power method
                tokenrank = None
                if compute_tokenrank:
                    v = np.ones(N, dtype=float) / N
                    for it in range(max_iter):
                        v_next = v @ P
                        if np.linalg.norm(v_next - v) < tol:
                            v = v_next
                            break
                        v = v_next
                    tokenrank = v / (v.sum() + 1e-12)

                # multi-bounce: start from identity (each node one-hot) and iterate
                mb_steps = []
                V = np.eye(N, dtype=float)  # (N,N) rows are one-hot from node i
                for k in range(multi_bounce_steps):
                    V = V @ P  # (N,N)
                    mb_steps.append(V.copy())  # store after this bounce; V[i,:] distribution from i
                tokenranks_b.append(None if tokenrank is None else tokenrank.copy())
                lambda2_b.append(lambda2)
                multi_bounce_b.append(mb_steps)

            results["tokenrank"][b] = tokenranks_b
            results["lambda2"][b] = lambda2_b
            results["multi_bounce"][b] = multi_bounce_b

            # plotting per batch
            if plot:
                # multilayer TokenRank bar grid and heatmap
                if compute_tokenrank:
                    pbar = os.path.join(self.out_dir, f"timestep_batch_batch{b}/tokenrank_multilayer.png")
                    self.plotter.bar_tokenrank_multilayer([tr for tr in tokenranks_b], pbar, batch_idx=b)
                    pheat = os.path.join(self.out_dir, f"timestep_batch_batch{b}/tokenrank_heatmap.png")
                    self.plotter.heatmap_tokenrank([tr for tr in tokenranks_b], pheat, batch_idx=b)
                # per-node multi-bounce: for a few nodes save heatmaps
                batch_dir = os.path.join(self.out_dir, f"timestep_batch_batch{b}/dtmc")
                os.makedirs(batch_dir, exist_ok=True)
                node_indices = range(min(N, max_nodes_to_plot))
                for l in range(L):
                    for node in node_indices:
                        mat = np.stack([step[node, :] for step in multi_bounce_b[l]], axis=0)  # (steps, N)
                        p = os.path.join(batch_dir, f"layer{l}_node{node}_multi_bounce.png")
                        self.plotter.save_heatmap(mat, p, title=f"Batch {b} L{l} node{node} multi-bounce")
        return results


# -------------------------
# Temporal wrapper
# -------------------------
class AttentionDynamics:
    """
    Manage attention across timesteps. Stores attention tensors per timestep and runs analyses.
    """

    def __init__(self, analyzer: AttentionAnalyzer):
        self.analyzer = analyzer
        self.timesteps: Dict[int, torch.Tensor] = {}

    def add_timestep(self, t: int, attn: torch.Tensor):
        attn = _ensure_batch_shape(attn)
        self.timesteps[t] = attn

    def analyze_all(self, methods: Tuple[str, ...] = ("maps", "rollout", "flow", "attngraph", "dtmc"),
                    flow_per_layer: bool = False, max_nodes_to_plot: int = 6):
        for t, attn in sorted(self.timesteps.items()):
            print(f"[Dynamics] analyzing timestep {t}")
            tdir = os.path.join(self.analyzer.out_dir, f"timestep_{t}")
            os.makedirs(tdir, exist_ok=True)
            # create a new analyzer scoped to this timestep dir
            analyzer = AttentionAnalyzer(out_dir=tdir, residual=self.analyzer.residual,
                                         threshold=self.analyzer.threshold, device=self.analyzer.device)
            if "maps" in methods:
                analyzer.plot_attention_maps(attn)
            if "rollout" in methods:
                analyzer.plot_rollout(attn)
            if "flow" in methods:
                analyzer.plot_flow(attn, per_layer_plot=flow_per_layer, max_nodes_to_plot=max_nodes_to_plot)
            if "attngraph" in methods:
                AAgg, graphs = analyzer.build_attention_graphs(attn)
                analyzer.plot_attention_graphs(AAgg)
            if "dtmc" in methods:
                analyzer.dtmc_analysis(attn)


# -------------------------
# Usage example
# -------------------------
if __name__ == "__main__":
    # Example usage: replace with your actual file loads
    # Suppose you have attn saved as (B, L, H, N, N) in a .pt file
    # attn = torch.load("attn.pt")
    B, L, H, N = 1, 8, 8, 63
    # random synthetic example
    attn = torch.rand((B, L, H, N, N))
    # softmax rows to make them stochastic (per-head)
    attn = torch.softmax(attn, dim=-1)

    analyzer = AttentionAnalyzer(out_dir="./attn_analysis_all", residual=True, threshold=1e-6)
    dynamics = AttentionDynamics(analyzer)
    dynamics.add_timestep(0, attn)
    dynamics.analyze_all(methods=("maps", "rollout", "attngraph", "dtmc"), flow_per_layer=False)

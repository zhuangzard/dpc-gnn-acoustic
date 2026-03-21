"""
Antisymmetric Message Passing layer for Route B wave GNN.

Guarantees F_ij = -F_ji by construction via EXACT antisymmetry:
  raw_ij = MLP([h_i, h_j, e_ij])
  raw_ji = MLP([h_j, h_i, e_ji])
  msg_ij = 0.5 * (raw_ij - raw_ji)

Physical meaning in acoustics:
  - Pressure gradient reciprocity: nabla_p(i->j) = -nabla_p(j->i)
  - Enforces momentum conservation between node pairs

Edge features layout: [dx, dy, dist, c_avg_norm, rho_avg_norm] (5-dim).
  - dx, dy: signed spatial offsets (antisymmetric under edge reversal)
  - dist: unsigned Euclidean distance (symmetric)
  - c_avg_norm: normalized average speed of sound (symmetric)
  - rho_avg_norm: normalized average density (symmetric)
"""

import torch
import torch.nn as nn


class AntisymmetricMPLayer(nn.Module):
    """
    Message passing with exact F_ij = -F_ji antisymmetry.

    Computes messages in both directions, then takes the antisymmetric
    combination: msg_ij = 0.5 * (raw_ij - raw_ji).

    Uses Pre-Norm (LayerNorm before message computation) for deep GNN stability,
    scatter_add for aggregation, and residual connections.

    Edge features: [dx, dy, dist, c_avg_norm, rho_avg_norm] (5-dim).

    Args:
        node_dim: Dimension of node features.
        edge_dim: Dimension of edge features (must be 5).
        msg_dim: Dimension of message MLP output.
    """

    def __init__(self, node_dim: int = 64, edge_dim: int = 5, msg_dim: int = 64):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.msg_dim = msg_dim

        # Pre-Norm: applied to h BEFORE computing messages (P2-2)
        self.norm = nn.LayerNorm(node_dim)

        # Message MLP: [h_i(node_dim), h_j(node_dim), e_ij(edge_dim)] -> msg_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, msg_dim),
            nn.GELU(),
            nn.Linear(msg_dim, msg_dim),
        )

        # Node update MLP: [h(node_dim), agg(msg_dim)] -> node_dim
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + msg_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (N, node_dim) node features.
            edge_index: (2, E) directed edges [src, dst].
            edge_attr: (E, edge_dim) edge features.
                       Layout: [dx, dy, dist, c_avg_norm, rho_avg_norm].
                       dx/dy are antisymmetric; dist/c_avg/rho_avg are symmetric.

        Returns:
            h_new: (N, node_dim) updated node features.
        """
        assert edge_attr.size(-1) == self.edge_dim, (
            f"Expected edge_attr dim {self.edge_dim}, got {edge_attr.size(-1)}. "
            f"Layout: [dx, dy, dist, c_avg_norm, rho_avg_norm]"
        )

        src, dst = edge_index  # (E,), (E,)
        N = h.size(0)

        # Pre-Norm (P2-2): normalize h before message computation
        h_normed = self.norm(h)

        h_src = h_normed[src]  # (E, node_dim)
        h_dst = h_normed[dst]  # (E, node_dim)

        # Reverse edge features: e_ji = [-dx, -dy, dist, c_avg, rho_avg]
        edge_attr_rev = edge_attr.clone()
        edge_attr_rev[:, :2] = -edge_attr_rev[:, :2]  # Flip dx, dy

        # P2-5: Batch ij/ji MLP calls - concatenate and run once
        feat_ij = torch.cat([h_src, h_dst, edge_attr], dim=-1)      # (E, 2*node_dim+edge_dim)
        feat_ji = torch.cat([h_dst, h_src, edge_attr_rev], dim=-1)  # (E, 2*node_dim+edge_dim)
        feat_both = torch.cat([feat_ij, feat_ji], dim=0)            # (2E, ...)
        raw_both = self.msg_mlp(feat_both)                          # (2E, msg_dim)
        raw_ij, raw_ji = raw_both.chunk(2, dim=0)                   # each (E, msg_dim)

        # Exact antisymmetric message
        msg = 0.5 * (raw_ij - raw_ji)  # (E, msg_dim), guaranteed F_ij = -F_ji

        # Aggregate messages at destination nodes
        agg = torch.zeros(N, self.msg_dim, device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)

        # Node update with residual connection (on un-normed h)
        h_update = self.update_mlp(torch.cat([h_normed, agg], dim=-1))  # (N, node_dim)
        h_new = h + h_update  # residual on original h (Pre-Norm pattern)

        return h_new


if __name__ == '__main__':
    print("Testing AntisymmetricMPLayer...")
    layer = AntisymmetricMPLayer(node_dim=64, edge_dim=5, msg_dim=64)
    n_params = sum(p.numel() for p in layer.parameters())
    print(f"  Parameters: {n_params:,}")

    # Small test graph: 4 nodes, 8 edges (fully connected pairs)
    h = torch.randn(4, 64)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0],
                                [1, 0, 2, 1, 3, 2, 0, 3]])
    edge_attr = torch.randn(8, 5)
    # Make spatial components antisymmetric for reverse edges
    edge_attr[1, :2] = -edge_attr[0, :2]
    edge_attr[3, :2] = -edge_attr[2, :2]
    edge_attr[5, :2] = -edge_attr[4, :2]
    edge_attr[7, :2] = -edge_attr[6, :2]

    h_out = layer(h, edge_index, edge_attr)
    print(f"  Input shape: {h.shape} -> Output shape: {h_out.shape}")
    print(f"  Output stats: mean={h_out.mean():.4f}, std={h_out.std():.4f}")
    print("  OK")

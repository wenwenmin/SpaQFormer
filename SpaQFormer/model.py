import inspect

import torch
from torch import nn
import torch.nn.functional as F

from .open_set import energy_score
from .utils import neighbor_mean


class ExpressionEncoder(nn.Module):
    """
    Project raw gene expression into the shared query-former dimension.
    """

    def \
            __init__(self, input_dim, hidden_dim):
        super(ExpressionEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class SpatialGraphEncoder(nn.Module):
    """
    GraphSAGE encoder for ST spatial graph evidence.

    PyTorch Geometric is imported lazily so non-graph utilities can be tested
    in environments where torch_geometric is not installed.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, act=F.relu):
        super(SpatialGraphEncoder, self).__init__()
        try:
            from torch_geometric.nn import SAGEConv
        except ImportError as exc:
            raise ImportError(
                "SpaQFormerModel requires torch_geometric for SpatialGraphEncoder."
            ) from exc

        dims = [input_dim] + list(hidden_dims)
        if dims[-1] != output_dim:
            dims.append(output_dim)

        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(SAGEConv(in_dim, out_dim))

        self.act = act

    def forward(self, x, edge_index):
        for conv in self.layers[:-1]:
            x = self.act(conv(x, edge_index))

        x = self.layers[-1](x, edge_index)
        return self.act(x)


class CellTypeQueryFormer(nn.Module):
    """
    Cross-attend cell-type queries to ST spot evidence tokens.
    """

    def __init__(
        self,
        hidden_dim,
        num_classes,
        num_queries=4,
        num_heads=4,
        dropout=0.1,
        max_pairs_per_chunk=8192,
    ):
        super(CellTypeQueryFormer, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.max_pairs_per_chunk = max_pairs_per_chunk

        self.type_queries = nn.Parameter(
            torch.randn(num_classes, num_queries, hidden_dim) * 0.02
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.match_head = nn.Linear(hidden_dim, 1)

    def initialize_queries_from_prototypes(self, prototypes, noise_std=0.0):
        """
        Initialize each cell-type query from its scRNA reference prototype.
        """
        if prototypes.dim() != 2:
            raise ValueError("prototypes must have shape [C, d].")
        if prototypes.size(0) != self.num_classes:
            raise ValueError("prototype class count must match num_classes.")
        if prototypes.size(1) != self.hidden_dim:
            raise ValueError("prototype dimension must match hidden_dim.")
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative.")

        with torch.no_grad():
            query_init = prototypes[:, None, :].expand(
                self.num_classes,
                self.num_queries,
                self.hidden_dim,
            ).clone()
            if noise_std > 0:
                query_init = query_init + torch.randn_like(query_init) * noise_std
            self.type_queries.copy_(query_init)

    def forward(self, spot_tokens, prototypes, return_attention=False):
        """
        Parameters
        ----------
        spot_tokens:
            Tensor with shape [N, 3, d].
        prototypes:
            Tensor with shape [C, d]. Prototypes initialize queries before
            forward and are used here only to validate class metadata.
        return_attention:
            Whether to return per-head attention weights when supported.
        """
        if spot_tokens.dim() != 3:
            raise ValueError("spot_tokens must have shape [N, T, d].")
        if prototypes.dim() != 2:
            raise ValueError("prototypes must have shape [C, d].")

        n_spots = spot_tokens.size(0)
        n_classes = prototypes.size(0)
        token_count = spot_tokens.size(1)

        if n_classes != self.num_classes:
            raise ValueError("prototype class count must match num_classes.")
        if spot_tokens.size(2) != self.hidden_dim:
            raise ValueError("spot token dimension must match hidden_dim.")
        if prototypes.size(1) != self.hidden_dim:
            raise ValueError("prototype dimension must match hidden_dim.")

        spot_tokens = spot_tokens[:, None, :, :].expand(
            n_spots,
            n_classes,
            spot_tokens.size(1),
            self.hidden_dim,
        )
        evidence_tokens = spot_tokens

        queries = self.type_queries[None, :, :, :].expand(
            n_spots,
            n_classes,
            self.num_queries,
            self.hidden_dim,
        )

        evidence_flat = evidence_tokens.reshape(
            n_spots * n_classes,
            token_count,
            self.hidden_dim,
        )
        queries_flat = queries.reshape(
            n_spots * n_classes,
            self.num_queries,
            self.hidden_dim,
        )

        scores, attn_weights = self._run_attention(
            queries_flat,
            evidence_flat,
            n_spots,
            n_classes,
            token_count,
            return_attention,
        )
        return scores, attn_weights

    def _run_attention(
        self,
        queries_flat,
        evidence_flat,
        n_spots,
        n_classes,
        token_count,
        return_attention,
    ):
        pair_count = queries_flat.size(0)
        chunk_size = self.max_pairs_per_chunk

        if chunk_size is None or pair_count <= chunk_size:
            scores, attn_weights = self._run_attention_chunk(
                queries_flat,
                evidence_flat,
                return_attention,
            )
            scores = scores.view(n_spots, n_classes)
            if return_attention:
                attn_weights = attn_weights.view(
                    n_spots,
                    n_classes,
                    -1,
                    self.num_queries,
                    token_count,
                )
            return scores, attn_weights

        score_chunks = []
        attn_chunks = []
        for start in range(0, pair_count, chunk_size):
            end = min(start + chunk_size, pair_count)
            chunk_scores, chunk_attn = self._run_attention_chunk(
                queries_flat[start:end],
                evidence_flat[start:end],
                return_attention,
            )
            score_chunks.append(chunk_scores)
            if return_attention:
                attn_chunks.append(chunk_attn)

        scores = torch.cat(score_chunks, dim=0).view(n_spots, n_classes)
        attn_weights = None
        if return_attention:
            attn_weights = torch.cat(attn_chunks, dim=0).view(
                n_spots,
                n_classes,
                -1,
                self.num_queries,
                token_count,
            )

        return scores, attn_weights

    def _run_attention_chunk(self, queries_flat, evidence_flat, return_attention):
        query_norm = self.norm_q(queries_flat)
        evidence_norm = self.norm_v(evidence_flat)

        attn_kwargs = {
            "query": query_norm,
            "key": evidence_norm,
            "value": evidence_norm,
            "need_weights": return_attention,
        }
        signature = inspect.signature(self.cross_attn.forward)
        supports_per_head = "average_attn_weights" in signature.parameters
        if return_attention and supports_per_head:
            attn_kwargs["average_attn_weights"] = False

        attn_out, attn_weights = self.cross_attn(**attn_kwargs)
        query_out = self.norm_attn(queries_flat + attn_out)
        query_out = self.norm_out(query_out + self.ffn(query_out))

        pooled = query_out.mean(dim=1)
        scores = self.match_head(pooled)

        if not return_attention:
            return scores, None

        if not supports_per_head:
            attn_weights = attn_weights.unsqueeze(1)

        return scores, attn_weights


class SpaQFormerModel(nn.Module):
    """
    First SpaQFormer model skeleton.

    The forward path returns known-type matching scores and energy scores.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        num_queries=4,
        num_heads=4,
        gnn_dims=(128,),
        dropout=0.1,
        max_pairs_per_chunk=8192,
    ):
        super(SpaQFormerModel, self).__init__()
        self.expr_encoder = ExpressionEncoder(input_dim, hidden_dim)
        self.graph_encoder = SpatialGraphEncoder(
            input_dim=input_dim,
            hidden_dims=gnn_dims,
            output_dim=hidden_dim,
        )
        self.query_former = CellTypeQueryFormer(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_queries=num_queries,
            num_heads=num_heads,
            dropout=dropout,
            max_pairs_per_chunk=max_pairs_per_chunk,
        )

    def forward(self, x, edge_index, prototypes, return_attention=False):
        expr_token = self.expr_encoder(x)
        spatial_token = self.graph_encoder(x, edge_index)
        neighbor_token = neighbor_mean(spatial_token, edge_index)
        spot_tokens = torch.stack(
            [expr_token, spatial_token, neighbor_token],
            dim=1,
        )

        scores, attn = self.query_former(
            spot_tokens=spot_tokens,
            prototypes=prototypes,
            return_attention=return_attention,
        )
        energy = energy_score(scores)

        if return_attention:
            return scores, energy, spatial_token, attn

        return scores, energy, spatial_token

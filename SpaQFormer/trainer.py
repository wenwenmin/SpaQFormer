import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


from .losses import (
    energy_known_loss,
    known_matching_loss,
    query_diversity_loss,
    spatial_consistency_loss,
)
from .model import SpaQFormerModel
from .open_set import estimate_energy_threshold, predict_open_set
from .preprocess import load_data
from .pretrain import FocalLoss, PretrainModel, SingleCellDataset
from .prototype import compute_reference_prototypes


class SpaQFormer(object):
    """
    Independent SpaQFormer training wrapper.

    This class intentionally lives outside the original SpatialMap package so
    the original GraphSAGE + Linear pipeline remains available as a baseline.
    """

    def __init__(
        self,
        sc_file=None,
        srt_file=None,
        device=torch.device("cpu"),
        lr1=5e-4,
        lr2=5e-4,
        epochs1=80,
        epochs2=200,
        seed=2026,
        Kn=12,
        gamma=2.0,
        hidden_dims=(512, 256),
        hidden_dim=None,
        batch_size1=2048,
        weight_dict=None,
        alpha=15.0,
        beta=0.2,
        temperature=0.3,
        slice_col=None,
        num_queries=4,
        num_heads=4,
        gnn_dims=None,
        dropout=0.1,
        qformer_chunk_size=8192,
        query_init_noise_std=0.0,
        energy_loss_weight=0.0,
        spatial_loss_weight=0.0,
        query_diversity_weight=0.01,
        show_progress=True,
        progress_refresh_epochs=5,
        sc_x=None,
        sc_y=None,
        srt_x=None,
        srt_edges=None,
        cell_type_dict=None,
        inverse_dict=None,
        gene_names=None,
        sc_obs_names=None,
        srt_obs_names=None,
        srt_coords=None,
    ):
        self.sc_file = sc_file
        self.srt_file = srt_file
        self.device = device

        self.lr1 = lr1
        self.lr2 = lr2
        self.epochs1 = epochs1
        self.epochs2 = epochs2
        self.seed = seed
        self.Kn = Kn
        self.gamma = gamma
        self.hidden_dims = list(hidden_dims)
        self.hidden_dim = hidden_dim if hidden_dim is not None else self.hidden_dims[-1]
        self.batch_size1 = batch_size1
        self.weight_dict = weight_dict
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.slice_col = slice_col
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.gnn_dims = tuple(gnn_dims) if gnn_dims is not None else (self.hidden_dim,)
        self.dropout = dropout
        self.qformer_chunk_size = qformer_chunk_size
        self.query_init_noise_std = query_init_noise_std
        self.energy_loss_weight = energy_loss_weight
        self.spatial_loss_weight = spatial_loss_weight
        self.query_diversity_weight = query_diversity_weight
        self.show_progress = show_progress
        self.progress_refresh_epochs = progress_refresh_epochs

        if sc_x is None or sc_y is None or srt_x is None or srt_edges is None:
            if sc_file is None or srt_file is None:
                raise ValueError("Provide either data arrays or sc_file/srt_file.")

            (
                sc_x,
                sc_y,
                srt_x,
                srt_edges,
                cell_type_dict,
                inverse_dict,
                gene_names,
                sc_obs_names,
                srt_obs_names,
                srt_coords,
            ) = load_data(sc_file, srt_file, Kn=Kn, slice_col=slice_col)

        self.sc_x = np.asarray(sc_x, dtype=np.float32)
        self.sc_y = np.asarray(sc_y, dtype=np.int64)
        self.srt_x = np.asarray(srt_x, dtype=np.float32)
        self.srt_edges = np.asarray(srt_edges, dtype=np.int64)

        if cell_type_dict is None:
            labels = sorted(np.unique(self.sc_y).tolist())
            cell_type_dict = {str(label): int(label) for label in labels}
        if inverse_dict is None:
            inverse_dict = {idx: name for name, idx in cell_type_dict.items()}

        self.cell_type_dict = cell_type_dict
        self.inverse_dict = inverse_dict
        self.gene_names = gene_names
        self.sc_obs_names = sc_obs_names
        self.srt_obs_names = srt_obs_names
        self.srt_coords = srt_coords

        self.num_classes = len(self.cell_type_dict)
        self.input_dim = self.sc_x.shape[1]

        self.weight = None
        self.model1 = None
        self.model2 = None
        self.prototype_proj = None

        self.reference_prototypes = None

        self.pseudo_labels = None
        self.pseudo_labels_soft = None
        self.pseudo_mask = None
        self.pseudo_confidence = None
        self.pseudo_entropy = None
        self.pseudo_selected_indices = None

        self.annotation = None
        self.open_set_annotation = None
        self.annotation_logits = None
        self.annotation_probs = None
        self.annotation_ids = None
        self.srt_qformer_embeddings = None
        self.unknown_scores = None
        self.unknown_mask = None
        self.energy_threshold = None

    @classmethod
    def from_arrays(cls, **kwargs):
        """
        Construct a trainer from preloaded arrays.
        """
        return cls(**kwargs)

    def pretrain(self):
        """
        Pretrain expression encoder and classifier on reference scRNA data.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if self.weight_dict:
            self.weight = torch.tensor(
                [
                    self.weight_dict.get(self.inverse_dict[i], 1.0)
                    for i in range(len(self.inverse_dict))
                ],
                dtype=torch.float32,
                device=self.device,
            )

        criterion = FocalLoss(gamma=self.gamma, weight=self.weight)
        self.model1 = PretrainModel(
            self.input_dim,
            self.hidden_dims,
            self.num_classes,
        ).to(self.device)

        optimizer = Adam(self.model1.parameters(), lr=self.lr1)
        dataset = SingleCellDataset(self.sc_x, self.sc_y)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size1,
            shuffle=True,
        )

        epoch_bar = self._epoch_progress(self.epochs1, "SpaQFormer pretrain")
        last_progress_epoch = 0
        for epoch in range(self.epochs1):
            self.model1.train()
            epoch_loss = 0.0
            batch_count = 0
            for x, y in dataloader:
                x = x.to(torch.float32).to(self.device)
                y = y.to(torch.long).to(self.device)

                _, class_logits, recon_x = self.model1(x)
                class_loss = criterion(class_logits, y)
                recon_loss = F.mse_loss(x, recon_x) * self.alpha
                loss = class_loss + recon_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1

            if batch_count > 0:
                last_progress_epoch = self._refresh_epoch_progress(
                    epoch_bar,
                    epoch,
                    last_progress_epoch,
                    self.epochs1,
                    loss=epoch_loss / batch_count,
                )
        self._close_progress(epoch_bar)

        return self.model1

    def pseudo_label(self):
        """
        Generate entropy-filtered soft pseudo-labels for ST spots.
        """
        if self.model1 is None:
            raise RuntimeError("Please run pretrain() before pseudo_label().")

        self.model1.eval()
        x = torch.tensor(self.srt_x, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, class_logits, _ = self.model1(x)
            probs = torch.softmax(class_logits / self.temperature, dim=1)
            probs = probs.cpu().numpy()

        pred_labels = np.argmax(probs, axis=1)
        entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
        entropy = entropy / np.log(self.num_classes)

        pseudo_label_indices = []
        for label in np.unique(pred_labels):
            label_indices = np.where(pred_labels == label)[0]
            label_entropy = entropy[label_indices]
            sorted_indices = np.argsort(label_entropy)
            cutoff_index = max(int(len(sorted_indices) * self.beta), 1)
            selected_label_indices = label_indices[sorted_indices[:cutoff_index]]
            pseudo_label_indices.extend(selected_label_indices)

        pseudo_label_indices = np.array(pseudo_label_indices, dtype=np.int64)

        self.pseudo_labels = np.full(len(self.srt_x), -1, dtype=np.int64)
        self.pseudo_labels[pseudo_label_indices] = pred_labels[pseudo_label_indices]

        self.pseudo_labels_soft = np.zeros(
            (len(self.srt_x), self.num_classes),
            dtype=np.float32,
        )
        self.pseudo_labels_soft[pseudo_label_indices] = probs[pseudo_label_indices]

        self.pseudo_mask = self.pseudo_labels > -1
        self.pseudo_confidence = probs.max(axis=1)
        self.pseudo_entropy = entropy
        self.pseudo_selected_indices = pseudo_label_indices

        return self.pseudo_labels, self.pseudo_mask

    def build_prototypes(self):
        """
        Build scRNA reference prototypes from the pretrained encoder.
        """
        if self.model1 is None:
            raise RuntimeError("Please run pretrain() before build_prototypes().")

        x = torch.tensor(self.sc_x, dtype=torch.float32, device=self.device)
        y = torch.tensor(self.sc_y, dtype=torch.long, device=self.device)

        self.model1.eval()
        with torch.no_grad():
            embeddings = self.model1.encoder(x)

        prototypes = compute_reference_prototypes(
            embeddings,
            y,
            self.num_classes,
            normalize=False,
        )

        if prototypes.size(1) != self.hidden_dim:
            raise ValueError(
                "prototype dimension must match hidden_dim. "
                f"Got prototype dimension {prototypes.size(1)} and "
                f"hidden_dim {self.hidden_dim}."
            )

        prototypes = F.normalize(prototypes, dim=1)
        self.reference_prototypes = prototypes.detach()
        return self.reference_prototypes

    def annotate(
        self,
        use_cluster=False,
        num_parts=10,
        cluster_batch_size=1,
        open_set=False,
    ):
        """
        Train SpaQFormer on the ST graph using pseudo-labeled known spots.

        When use_cluster is True, training and inference use PyG
        ClusterData/ClusterLoader and recover global spot order via data.index.
        """
        if self.pseudo_labels is None or self.pseudo_mask is None:
            raise RuntimeError("Please run pseudo_label() before annotate().")
        if self.reference_prototypes is None:
            self.build_prototypes()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        prototypes = self.reference_prototypes.to(self.device)
        self.model2 = self._init_annotation_model(prototypes)

        if use_cluster:
            scores, energy, embeddings = self._annotate_cluster(
                prototypes,
                num_parts=num_parts,
                cluster_batch_size=cluster_batch_size,
            )
        else:
            scores, energy, embeddings = self._annotate_full_graph(prototypes)

        return self._store_annotation_outputs(
            scores,
            energy,
            embeddings,
            open_set=open_set,
        )

    def _init_annotation_model(self, prototypes=None):
        model = SpaQFormerModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_queries=self.num_queries,
            num_heads=self.num_heads,
            gnn_dims=self.gnn_dims,
            dropout=self.dropout,
            max_pairs_per_chunk=self.qformer_chunk_size,
        ).to(self.device)
        if prototypes is not None:
            model.query_former.initialize_queries_from_prototypes(
                prototypes.to(self.device),
                noise_std=self.query_init_noise_std,
            )
        return model

    def _annotate_full_graph(self, prototypes):
        x = torch.tensor(self.srt_x, dtype=torch.float32, device=self.device)
        edge_index = self._edge_index_tensor().to(self.device)
        mask = torch.tensor(self.pseudo_mask, dtype=torch.bool, device=self.device)
        soft_targets = torch.tensor(
            self.pseudo_labels_soft[self.pseudo_mask],
            dtype=torch.float32,
            device=self.device,
        )
        hard_targets = torch.tensor(
            self.pseudo_labels[self.pseudo_mask],
            dtype=torch.long,
            device=self.device,
        )
        optimizer = Adam(self.model2.parameters(), lr=self.lr2)

        epoch_bar = self._epoch_progress(self.epochs2, "SpaQFormer annotate")
        last_progress_epoch = 0
        for epoch in range(self.epochs2):
            self.model2.train()
            scores, energy, _ = self.model2(x, edge_index, prototypes)

            loss = known_matching_loss(
                scores[mask],
                soft_targets,
                hard_targets,
            )
            if self.energy_loss_weight > 0:
                loss = loss + self.energy_loss_weight * energy_known_loss(
                    energy[mask],
                    margin=energy[mask].detach().mean(),
                )
            if self.query_diversity_weight > 0:
                loss = loss + self.query_diversity_weight * query_diversity_loss(
                    self.model2.query_former.type_queries
                )

            if self.spatial_loss_weight > 0 and edge_index.size(1) > 0:
                loss = loss + self.spatial_loss_weight * spatial_consistency_loss(
                    scores,
                    edge_index,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_progress_epoch = self._refresh_epoch_progress(
                epoch_bar,
                epoch,
                last_progress_epoch,
                self.epochs2,
                loss=loss.item(),
            )
        self._close_progress(epoch_bar)

        self.model2.eval()
        with torch.no_grad():
            scores, energy, embeddings = self.model2(x, edge_index, prototypes)

        return scores, energy, embeddings

    def _annotate_cluster(self, prototypes, num_parts, cluster_batch_size):
        from torch_geometric.data import Data
        from torch_geometric.loader import ClusterData, ClusterLoader

        target_data = Data(
            x=torch.tensor(self.srt_x, dtype=torch.float32),
            edge_index=self._edge_index_tensor(),
            y=torch.tensor(self.pseudo_labels, dtype=torch.long),
        )
        target_data.pseudo_mask = torch.tensor(self.pseudo_mask, dtype=torch.bool)
        target_data.index = torch.arange(len(self.srt_x), dtype=torch.long)

        target_dataset = ClusterData(
            target_data,
            num_parts=num_parts,
            recursive=False,
        )
        target_loader = ClusterLoader(
            target_dataset,
            batch_size=cluster_batch_size,
            shuffle=True,
            num_workers=0,
        )

        optimizer = Adam(self.model2.parameters(), lr=self.lr2)

        epoch_bar = self._epoch_progress(
            self.epochs2,
            "SpaQFormer cluster annotate",
        )
        last_progress_epoch = 0
        for epoch in range(self.epochs2):
            self.model2.train()
            epoch_loss = 0.0
            valid_batches = 0
            for data in target_loader:
                data = data.to(self.device)
                local_mask = data.pseudo_mask
                if local_mask.sum().item() == 0:
                    continue

                scores, energy, _ = self.model2(data.x, data.edge_index, prototypes)
                global_ids = data.index[local_mask].cpu().numpy()
                soft_targets = torch.tensor(
                    self.pseudo_labels_soft[global_ids],
                    dtype=torch.float32,
                    device=self.device,
                )
                hard_targets = torch.tensor(
                    self.pseudo_labels[global_ids],
                    dtype=torch.long,
                    device=self.device,
                )

                loss = known_matching_loss(
                    scores[local_mask],
                    soft_targets,
                    hard_targets,
                )
                if self.energy_loss_weight > 0:
                    loss = loss + self.energy_loss_weight * energy_known_loss(
                        energy[local_mask],
                        margin=energy[local_mask].detach().mean(),
                    )
                if self.query_diversity_weight > 0:
                    loss = loss + self.query_diversity_weight * query_diversity_loss(
                        self.model2.query_former.type_queries
                    )
                if self.spatial_loss_weight > 0 and data.edge_index.size(1) > 0:
                    loss = loss + self.spatial_loss_weight * spatial_consistency_loss(
                        scores,
                        data.edge_index,
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                valid_batches += 1

            if valid_batches > 0:
                last_progress_epoch = self._refresh_epoch_progress(
                    epoch_bar,
                    epoch,
                    last_progress_epoch,
                    self.epochs2,
                    loss=epoch_loss / valid_batches,
                )
        self._close_progress(epoch_bar)

        inference_loader = ClusterLoader(
            target_dataset,
            batch_size=cluster_batch_size,
            shuffle=False,
            num_workers=0,
        )

        n_nodes = len(self.srt_x)
        logits_sum = torch.zeros(
            n_nodes,
            self.num_classes,
            dtype=torch.float32,
            device=self.device,
        )
        energy_sum = torch.zeros(n_nodes, dtype=torch.float32, device=self.device)
        emb_sum = torch.zeros(
            n_nodes,
            self.hidden_dim,
            dtype=torch.float32,
            device=self.device,
        )
        counts = torch.zeros(n_nodes, dtype=torch.float32, device=self.device)

        self.model2.eval()
        with torch.no_grad():
            for data in inference_loader:
                data = data.to(self.device)
                scores, energy, embeddings = self.model2(
                    data.x,
                    data.edge_index,
                    prototypes,
                )
                global_ids = data.index
                logits_sum.index_add_(0, global_ids, scores)
                energy_sum.index_add_(0, global_ids, energy)
                emb_sum.index_add_(0, global_ids, embeddings)
                counts.index_add_(
                    0,
                    global_ids,
                    torch.ones(
                        global_ids.size(0),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )

        counts = counts.clamp(min=1.0)
        scores = logits_sum / counts.unsqueeze(1)
        energy = energy_sum / counts
        embeddings = emb_sum / counts.unsqueeze(1)
        return scores, energy, embeddings

    def _store_annotation_outputs(self, scores, energy, embeddings, open_set=False):
        probs = torch.softmax(scores, dim=1)
        known_ids = torch.argmax(scores, dim=1)
        if open_set:
            mask = torch.tensor(self.pseudo_mask, dtype=torch.bool, device=scores.device)
            self.energy_threshold = estimate_energy_threshold(
                energy[mask],
                quantile=0.95,
            )
            known_ids, unknown_mask, energy = predict_open_set(
                scores,
                self.energy_threshold,
            )
        else:
            self.energy_threshold = None
            unknown_mask = torch.zeros_like(energy, dtype=torch.bool)

        self.annotation_logits = scores.cpu().numpy()
        self.annotation_probs = probs.cpu().numpy()
        self.annotation_ids = known_ids.cpu().numpy()
        self.srt_qformer_embeddings = embeddings.cpu().numpy()
        self.unknown_scores = energy.cpu().numpy()
        self.unknown_mask = unknown_mask.cpu().numpy()

        self.annotation = [
            self.inverse_dict[int(idx)]
            for idx in self.annotation_ids
        ]
        self.open_set_annotation = [
            "unknown" if is_unknown else label
            for label, is_unknown in zip(self.annotation, self.unknown_mask)
        ]

        if open_set:
            return self.open_set_annotation
        return self.annotation

    def _edge_index_tensor(self):
        edges = torch.tensor(self.srt_edges, dtype=torch.long)
        if edges.dim() != 2:
            raise ValueError("srt_edges must be a 2D array.")
        if edges.size(0) == 2:
            return edges
        if edges.size(1) == 2:
            return edges.t().contiguous()
        raise ValueError("srt_edges must have shape [E, 2] or [2, E].")

    def _epoch_progress(self, total, desc):
        if not self.show_progress or tqdm is None:
            return None
        return tqdm(
            total=total,
            desc=desc,
            unit="epoch",
            dynamic_ncols=True,
            leave=True,
            mininterval=0.5,
        )

    def _refresh_epoch_progress(
        self,
        progress_bar,
        epoch,
        last_epoch,
        total_epochs,
        **postfix,
    ):
        current_epoch = epoch + 1
        refresh_epochs = max(int(self.progress_refresh_epochs), 1)
        should_refresh = (
            current_epoch == total_epochs
            or current_epoch % refresh_epochs == 0
        )
        if progress_bar is None or not should_refresh:
            return last_epoch

        delta = current_epoch - last_epoch
        if delta > 0:
            progress_bar.update(delta)
        progress_bar.set_postfix(**postfix)
        return current_epoch

    @staticmethod
    def _close_progress(progress_bar):
        if progress_bar is not None:
            progress_bar.close()

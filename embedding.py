import os

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataset2 import MyDataset

class CrossModalFusionLayer(nn.Module):
    """
    Cross-modal attention mechanism that allows different feature modalities
    to interact and influence each other.
    """

    def __init__(self, hidden_dim, num_modalities, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.num_heads = num_heads

        # --- FIX START ---
        # The premature 'return' was removed. We now initialize all layers
        # regardless of the number of modalities. The logic to handle the
        # edge case is correctly placed in the forward() method.

        # Multi-head cross-attention for each modality pair
        self.cross_attentions = nn.ModuleDict()
        # Fusion gate to control how much cross-modal influence is added
        self.fusion_gates = nn.ModuleList()

        # Only create attention and gate layers if there is something to fuse
        if self.num_modalities > 1:
            print(self.num_modalities)
            for i in range(num_modalities):
                # Add a gate for each modality
                self.fusion_gates.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.Sigmoid()
                    )
                )
                # Add cross-attention layers for pairs
                for j in range(num_modalities):
                    if i != j:
                        self.cross_attentions[f"{i}_{j}"] = nn.MultiheadAttention(
                            embed_dim=hidden_dim,
                            num_heads=num_heads,
                            dropout=0.3,
                            batch_first=True
                        )

        # LayerNorm should always be initialized for consistent output
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, modality_features):
        """
        Args:
            modality_features: List of [B, L, D] tensors, one for each modality.
        Returns:
            A single [B, L, D] tensor representing the fused embedding.
        """
        # If there's only one modality, just return it. No fusion needed.
        if self.num_modalities <= 1:
            return modality_features[0] if modality_features else None

        enhanced_features = []
        for i, source_feat in enumerate(modality_features):
            cross_modal_influences = []

            # Apply cross-attention from all other modalities
            for j, target_feat in enumerate(modality_features):
                if i != j:
                    attended_feat, _ = self.cross_attentions[f"{i}_{j}"](
                        query=source_feat, key=target_feat, value=target_feat
                    )
                    cross_modal_influences.append(attended_feat)

            # Combine influences and apply gated fusion
            combined_influence = torch.stack(cross_modal_influences).mean(dim=0)
            gate_input = torch.cat([source_feat, combined_influence], dim=-1)
            gate = self.fusion_gates[i](gate_input)
            enhanced_feat = source_feat + gate * combined_influence
            enhanced_features.append(enhanced_feat)

        fused_embedding = torch.stack(enhanced_features).sum(dim=0)
        return self.norm(fused_embedding)


class ModalitySpecificEncoder(nn.Module):
    """
    Specialized encoder for each modality type
    """

    def __init__(self, input_dim, hidden_dim, modality_type="visual"):
        super().__init__()
        self.modality_type = modality_type

        if modality_type == "visual":
            # Visual features often benefit from convolutional-like processing
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif modality_type == "categorical":
            # Categorical features benefit from embedding-like processing
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim)
            )
        elif modality_type == "behavioral":
            # Behavioral features might need temporal modeling
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:  # Generic encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

    def forward(self, x):
        return self.encoder(x)


class SharedEmbeddingModule(nn.Module):
    """
    Shared embedding module that both HSTU and Hydra models can use.
    This ensures consistent item representations across retrieval and reranking.
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args, interaction_vocab):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.args = args
        self.interaction_vocab_dict = interaction_vocab if interaction_vocab is not None else {}
        print(f"Initializing embeddings with user_num={user_num}, item_num={item_num}")
        # Core embeddings - shared across both models
        self.item_emb = nn.Embedding(item_num + 1, args.shared_hidden_units, padding_idx=0)
        self.user_emb = nn.Embedding(user_num + 1, args.shared_hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(2 * args.maxlen + 1, args.shared_hidden_units, padding_idx=0)
        self.action_emb = nn.Embedding(args.num_actions + 1, args.shared_hidden_units, padding_idx=0)
        self.time_span_emb = nn.Embedding(
            getattr(args, 'time_span', 512), args.shared_hidden_units, padding_idx=0
        )
        self.dwell_time_emb = nn.Embedding(7, args.shared_hidden_units, padding_idx=0)

        # Feature processing - shared logic
        self.sparse_emb = nn.ModuleDict()
        self.emb_transform = nn.ModuleDict()
        self.feature_projections = nn.ModuleDict()

        # NEW: Early fusion projection layer (for Hydra-style)
        # This will be initialized after _init_feature_embeddings determines total early_fusion_dim
        self.early_fusion_projection_with_user = None
        self.early_fusion_projection_no_user = None



        self._init_feat_info(feat_statistics, feat_types)

        self.num_item_emb_features = len(feat_types.get('item_emb', []))
        self.cross_modal_fusion = CrossModalFusionLayer(
            hidden_dim=args.shared_hidden_units,
            num_modalities=self.num_item_emb_features,
            num_heads=2  # Or make this configurable in args
        )


        self._init_feature_embeddings(args)
        self._init_early_fusion_projection(args)
        self.feature_gates = nn.ModuleDict()
        all_feature_keys = (
                list(self.USER_SPARSE_FEAT.keys()) + list(self.ITEM_SPARSE_FEAT.keys()) +
                list(self.USER_ARRAY_FEAT.keys()) + list(self.ITEM_ARRAY_FEAT.keys()) +
                list(self.INTERACTION_SPARSE_FEAT.keys()) + list(self.USER_CONTINUAL_FEAT) +
                list(self.ITEM_CONTINUAL_FEAT) + list(self.ITEM_EMB_FEAT.keys())
        )
        for k in all_feature_keys:
            self.feature_gates[k] = nn.Sequential(
                nn.Linear(args.shared_hidden_units, args.shared_hidden_units),
                nn.Sigmoid()
            )

        self.num_item_features = (
                len(feat_types.get('item_sparse', [])) +
                len(feat_types.get('item_array', [])) +
                len(feat_types.get('item_continual', [])) +
                len(feat_types.get('item_emb', []))
        )

        self.num_user_features = (
                len(feat_types.get('user_sparse', [])) +
                len(feat_types.get('user_array', [])) +
                len(feat_types.get('user_continual', []))
        )

    def _get_gated_fusion_features(self, feature_array, feature_groups):
        """Gated fusion: apply a learned gate to each feature before summing."""
        total_feature_emb = None
        max_seq_len = max(len(s) for s in feature_array)
        batch_size = len(feature_array)

        feature_groups['interaction_sparse'] = self.INTERACTION_SPARSE_FEAT
        for group_name, feat_dict_or_list in feature_groups.items():
            if not feat_dict_or_list:
                continue

            keys = feat_dict_or_list.keys() if isinstance(feat_dict_or_list, dict) else feat_dict_or_list

            for k in keys:
                projected_emb = None
                if 'item_emb' in group_name:
                    emb_dim = self.ITEM_EMB_FEAT[k]
                    batch_emb_data = np.zeros((batch_size, max_seq_len, emb_dim), dtype=np.float32)
                    for i, seq_i in enumerate(feature_array):
                        for j, item in enumerate(seq_i):
                            if k in item:
                                batch_emb_data[i, j] = item[k]

                    tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
                    projected_emb = self.emb_transform[k](tensor_feature)
                else:
                    tensor_feature = self.feat2tensor(feature_array, k)
                    if 'sparse' in group_name:
                        raw_emb = self.sparse_emb[k](tensor_feature)
                        projected_emb = self.feature_projections[k](raw_emb)
                    elif 'array' in group_name:
                        raw_emb = self.sparse_emb[k](tensor_feature).sum(2)
                        projected_emb = self.feature_projections[k](raw_emb)
                    elif 'continual' in group_name:
                        raw_feat = tensor_feature.unsqueeze(2).float()
                        projected_emb = self.feature_projections[k](raw_feat)

                if projected_emb is not None:
                    gate = self.feature_gates[k](projected_emb)
                    gated_emb = projected_emb * gate
                    if total_feature_emb is None:
                        total_feature_emb = gated_emb
                    else:
                        total_feature_emb += gated_emb

        return total_feature_emb if total_feature_emb is not None else torch.zeros(
            (batch_size, max_seq_len, self.args.shared_hidden_units),
            device=self.dev
        )
    def _init_feat_info(self, feat_statistics, feat_types):
        """Initialize feature type mappings"""
        max_interaction_vocab = 50000
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        self.INTERACTION_SPARSE_FEAT = {}
        for feature_id, vocab in self.interaction_vocab_dict.items():
            key = f'interaction_user_{feature_id}'
            self.INTERACTION_SPARSE_FEAT[key] = max_interaction_vocab

        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}

    def _init_feature_embeddings(self, args):
        """Initialize all feature embedding layers"""
        # Sparse feature embeddings
        for k in self.USER_SPARSE_FEAT:
            vocab_size = self.USER_SPARSE_FEAT[k] + 1
            self.sparse_emb[k] = nn.Embedding(
                vocab_size, args.shared_hidden_units, padding_idx=0
            )
            nn.init.normal_(self.sparse_emb[k].weight[1], mean=0, std=0.01)
        for k in self.ITEM_SPARSE_FEAT:
            vocab_size = self.ITEM_SPARSE_FEAT[k] + 1
            self.sparse_emb[k] = nn.Embedding(
                vocab_size, args.shared_hidden_units, padding_idx=0
            )
            nn.init.normal_(self.sparse_emb[k].weight[1], mean=0, std=0.01)
        for k in self.ITEM_ARRAY_FEAT:
            vocab_size = self.ITEM_ARRAY_FEAT[k] + 1
            self.sparse_emb[k] = nn.Embedding(
                vocab_size, args.shared_hidden_units, padding_idx=0
            )
            nn.init.normal_(self.sparse_emb[k].weight[1], mean=0, std=0.01)

        for k in self.USER_ARRAY_FEAT:
            vocab_size = self.USER_ARRAY_FEAT[k] + 1
            self.sparse_emb[k] = nn.Embedding(
                vocab_size, args.shared_hidden_units, padding_idx=0
            )
            nn.init.normal_(self.sparse_emb[k].weight[1], mean=0, std=0.01)
        for k in self.INTERACTION_SPARSE_FEAT:
            self.sparse_emb[k] = nn.Embedding(
                self.INTERACTION_SPARSE_FEAT[k] + 1, args.shared_hidden_units, padding_idx=0
            )
        # Pre-computed embedding transformations
        # for k in self.ITEM_EMB_FEAT:
        #     # Using a small MLP for potentially better transformation of raw embeddings
        #     self.emb_transform[k] = nn.Sequential(
        #         nn.Linear(self.ITEM_EMB_FEAT[k], args.shared_hidden_units // 2),
        #         nn.ReLU(),
        #         nn.Linear(args.shared_hidden_units // 2, args.shared_hidden_units)
        #     )
        modality_mapping = {
            "81": "categorical",
            "82": "visual",
            "83": "visual",
            "84": "visual",
            "85": "visual",
            "86": "visual",
        }
        print("Initializing modality-specific encoders for item embeddings...")
        for k in self.ITEM_EMB_FEAT:
            input_dim = self.ITEM_EMB_FEAT[k]

            # Default to a generic encoder if k is not in our map
            modality_type = modality_mapping.get(k, "generic")

            print(f"  - Feature '{k}': input_dim={input_dim}, type='{modality_type}'")

            # 3. Instantiate and assign the correct encoder
            self.emb_transform[k] = ModalitySpecificEncoder(
                input_dim=input_dim,
                hidden_dim=args.shared_hidden_units,
                modality_type=modality_type
            )

        # Feature projections for late fusion (HSTU style)
        all_sparse_array_keys = (
                list(self.USER_SPARSE_FEAT.keys()) + list(self.ITEM_SPARSE_FEAT.keys()) +
                list(self.USER_ARRAY_FEAT.keys()) + list(self.ITEM_ARRAY_FEAT.keys()) +
                list(self.INTERACTION_SPARSE_FEAT.keys())
        )
        for k in all_sparse_array_keys:
            self.feature_projections[k] = nn.Linear(args.shared_hidden_units, args.shared_hidden_units)

        # Continual feature projections
        all_continual_keys = self.USER_CONTINUAL_FEAT + self.ITEM_CONTINUAL_FEAT
        for k in all_continual_keys:
            self.feature_projections[k] = nn.Linear(1, args.shared_hidden_units)

    def _init_early_fusion_projection(self, args):
        """
        Initializes projection layers for early fusion.
        MODIFIED to account for pre-fused multi-modal embeddings.
        """
        # === Base dimension from non-embedding features ===
        base_dim = 0
        base_dim += len(self.ITEM_SPARSE_FEAT) * args.shared_hidden_units
        base_dim += len(self.ITEM_ARRAY_FEAT) * args.shared_hidden_units
        base_dim += len(self.ITEM_CONTINUAL_FEAT) * args.shared_hidden_units  # Assumes projection to hidden_units

        # --- MODIFICATION ---
        # If there are item_emb features, they will be fused into a SINGLE representation
        # of size shared_hidden_units.
        if self.ITEM_EMB_FEAT:
            base_dim += args.shared_hidden_units

        # === Case 1: For standalone items (include_user=False) ===
        item_only_dim = base_dim
        if item_only_dim > 0:
            self.early_fusion_projection_no_user = nn.Linear(item_only_dim, args.shared_hidden_units)
        else:
            # Fallback if there are no features at all
            self.early_fusion_projection_no_user = nn.Identity()

        # === Case 2: For the sequence (include_user=True) ===
        with_user_dim = item_only_dim
        with_user_dim += len(self.USER_SPARSE_FEAT) * args.shared_hidden_units
        with_user_dim += len(self.USER_ARRAY_FEAT) * args.shared_hidden_units
        with_user_dim += len(self.USER_CONTINUAL_FEAT) * args.shared_hidden_units
        with_user_dim += len(self.INTERACTION_SPARSE_FEAT) * args.shared_hidden_units

        if with_user_dim > 0:
            self.early_fusion_projection_with_user = nn.Linear(with_user_dim, args.shared_hidden_units)
        else:
            self.early_fusion_projection_with_user = nn.Identity()

    def feat2tensor(self, seq_feature, k):
        """Convert feature sequences to tensors"""
        batch_size = len(seq_feature)
        device = next(self.parameters()).device
        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # Handle list of lists (array features)
            max_seq_len = max(len(s) for s in seq_feature)
            max_array_len = max(
                (len(item_data) for s in seq_feature for item_data in [item[k] for item in s if k in item]),
                default=1  # Default to 1 if no array features exist to avoid error
            )

            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i] if k in item]  # Only take existing features
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]
            return torch.from_numpy(batch_data).to(self.dev)
        else:
            max_seq_len = max(len(s) for s in seq_feature)
            is_continual = k in self.USER_CONTINUAL_FEAT or k in self.ITEM_CONTINUAL_FEAT
            dtype = np.float32 if is_continual else np.int64
            batch_data = np.zeros((batch_size, max_seq_len), dtype=dtype)

            for i in range(batch_size):
                # Only extract feature if key exists in item dict
                seq_data = [item[k] if k in item else (0.0 if is_continual else 0) for item in seq_feature[i]]
                padded_seq_data = seq_data + [0] * (max_seq_len - len(seq_data))
                batch_data[i] = padded_seq_data
            return torch.from_numpy(batch_data).to(device)

    def get_base_embeddings(self, seq, mask=None, include_user=False):
        """Get base ID embeddings (item + user if specified)"""
        seq = seq.to(self.dev)

        if include_user and mask is not None:
            mask = mask.to(self.dev)  # Ensure mask is on correct device

            user_mask_indices = (mask == 2).nonzero(as_tuple=True)
            item_mask_indices = (mask == 1).nonzero(as_tuple=True)

            # Initialize an empty tensor for combined embeddings
            combined_emb = torch.zeros(seq.shape[0], seq.shape[1], self.args.shared_hidden_units, device=self.dev)

            # Apply user embeddings where mask is 2
            if user_mask_indices[0].numel() > 0:
                user_ids = seq[user_mask_indices].to(self.dev)  # Explicitly move to device
                combined_emb[user_mask_indices] += self.user_emb(user_ids)

            # Apply item embeddings where mask is 1
            if item_mask_indices[0].numel() > 0:
                item_ids = seq[item_mask_indices].to(self.dev)  # Explicitly move to device
                combined_emb[item_mask_indices] += self.item_emb(item_ids)

            return combined_emb
        else:
            return self.item_emb(seq)

    def get_feature_embeddings(self, feature_array, include_user=False, fusion_style='late'):
        """
        Get feature embeddings with specified fusion style

        Args:
            feature_array: Feature sequences
            include_user: Whether to include user features
            fusion_style: 'late' (HSTU-style) or 'early' (Hydra-style)
        """
        feature_groups = {
            'item_sparse': self.ITEM_SPARSE_FEAT,
            'item_array': self.ITEM_ARRAY_FEAT,
            'item_continual': self.ITEM_CONTINUAL_FEAT,
            'item_emb': self.ITEM_EMB_FEAT
        }

        if include_user:
            feature_groups.update({
                'user_sparse': self.USER_SPARSE_FEAT,
                'user_array': self.USER_ARRAY_FEAT,
                'user_continual': self.USER_CONTINUAL_FEAT,
            })

        if fusion_style == 'late':
            return self._get_late_fusion_features(feature_array, feature_groups)
        elif fusion_style == 'gated':
            return self._get_gated_fusion_features(feature_array, feature_groups)
        else:  # early fusion
            return self._get_early_fusion_features(feature_array, feature_groups, include_user)

    def _get_late_fusion_features(self, feature_array, feature_groups):
        """Late fusion: project each feature individually then sum"""
        total_feature_emb = None

        # Determine max sequence length for correct tensor sizing
        if len(feature_array) == 0 or len(feature_array[0]) == 0:
            # Handle empty feature_array or empty sequences within it
            return torch.zeros(
                (len(feature_array), 0, self.args.shared_hidden_units),
                device=self.dev
            )
        max_seq_len = max(len(s) for s in feature_array)
        batch_size = len(feature_array)

        feature_groups['interaction_sparse'] = self.INTERACTION_SPARSE_FEAT
        for group_name, feat_dict_or_list in feature_groups.items():
            if not feat_dict_or_list:
                continue

            keys = feat_dict_or_list.keys() if isinstance(feat_dict_or_list, dict) else feat_dict_or_list

            for k in keys:
                projected_emb = None
                if 'item_emb' in group_name:
                    # Special handling for pre-computed embeddings
                    emb_dim = self.ITEM_EMB_FEAT[k]
                    batch_emb_data = np.zeros((batch_size, max_seq_len, emb_dim), dtype=np.float32)
                    for i, seq_i in enumerate(feature_array):
                        for j, item in enumerate(seq_i):
                            if k in item:
                                batch_emb_data[i, j] = item[k]

                    tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
                    projected_emb = self.emb_transform[k](tensor_feature)
                else:
                    tensor_feature = self.feat2tensor(feature_array, k)
                    if 'sparse' in group_name:
                        raw_emb = self.sparse_emb[k](tensor_feature)
                        projected_emb = self.feature_projections[k](raw_emb)
                    elif 'array' in group_name:
                        # Ensure array features are summed and then projected
                        raw_emb = self.sparse_emb[k](tensor_feature).sum(2)  # Sum along the array dimension
                        projected_emb = self.feature_projections[k](raw_emb)
                    elif 'continual' in group_name:
                        raw_feat = tensor_feature.unsqueeze(2).float()
                        projected_emb = self.feature_projections[k](raw_feat)

                if projected_emb is not None:
                    if total_feature_emb is None:
                        total_feature_emb = projected_emb
                    else:
                        total_feature_emb += projected_emb

        return total_feature_emb if total_feature_emb is not None else torch.zeros(
            (batch_size, max_seq_len, self.args.shared_hidden_units),
            device=self.dev
        )

    def _get_early_fusion_features(self, feature_array, feature_groups, include_user = False):
        """Early fusion: collect all features, concatenate, then project"""
        all_other_features = []
        item_emb_modalities = []
        # Determine max sequence length for correct tensor sizing
        if len(feature_array) == 0 or len(feature_array[0]) == 0:
            # Handle empty feature_array or empty sequences within it
            return torch.zeros(
                (len(feature_array), 0, self.args.shared_hidden_units),
                device=self.dev
            )
        max_seq_len = max(len(s) for s in feature_array)
        batch_size = len(feature_array)
        if include_user:
            feature_groups['interaction_sparse'] = self.INTERACTION_SPARSE_FEAT
        for group_name, feat_dict_or_list in feature_groups.items():
            if not feat_dict_or_list:
                continue

            keys = feat_dict_or_list.keys() if isinstance(feat_dict_or_list, dict) else feat_dict_or_list

            for k in keys:
                if 'item_emb' in group_name:
                    emb_dim = self.ITEM_EMB_FEAT[k]
                    batch_emb_data = np.zeros((batch_size, max_seq_len, emb_dim), dtype=np.float32)
                    for i, seq_i in enumerate(feature_array):
                        for j, item in enumerate(seq_i):
                            if k in item:
                                batch_emb_data[i, j] = item[k]

                    tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
                    normalized_feature = F.normalize(tensor_feature, p=2, dim=-1)
                    item_emb_modalities.append(self.emb_transform[k](normalized_feature))
                    # all_features.append(
                    #     self.emb_transform[k](tensor_feature))  # already projected to shared_hidden_units
                else:
                    tensor_feature = self.feat2tensor(feature_array, k)
                    if 'sparse' in group_name:
                        all_other_features.append(self.sparse_emb[k](tensor_feature))
                    elif 'array' in group_name:
                        all_other_features.append(self.sparse_emb[k](tensor_feature).sum(2))  # Sum along array dimension
                    elif 'continual' in group_name:
                        # Continual features need to be projected to shared_hidden_units
                        raw_feat = tensor_feature.unsqueeze(2).float()
                        all_other_features.append(self.feature_projections[k](raw_feat))

        if item_emb_modalities:
            fused_item_emb = self.cross_modal_fusion(item_emb_modalities)
            # Add the single, powerful fused representation to the list of other features
            all_other_features.append(fused_item_emb)

        if all_other_features:
            # Concatenate all features
            concatenated_features = torch.cat(all_other_features, dim=2)
            # Project to shared_hidden_units
            if include_user:
                # Use the big projection layer (e.g., in_features=1740)
                projected_features = self.early_fusion_projection_with_user(concatenated_features)
            else:
                # Use the smaller projection layer (e.g., in_features=1260)
                projected_features = self.early_fusion_projection_no_user(concatenated_features)

            return F.relu(projected_features)
        else:
            return torch.zeros(
                (batch_size, max_seq_len, self.args.shared_hidden_units),
                device=self.dev
            )

    def get_complete_embeddings(self, seq, feature_array, mask=None, include_user=False,
                                fusion_style='late', model_style='hstu', pos_seq = None):
        """
        Get complete embeddings combining base IDs and features

        Args:
            seq: Sequence of item/user IDs
            feature_array: Feature sequences (list of lists of dicts)
            mask: Mask for user/item distinction
            include_user: Whether to include user features
            fusion_style: 'late' or 'early'
            model_style: 'hstu' or 'hydra' for model-specific processing (determines final projection)
        """
        # Get base embeddings
        base_emb = self.get_base_embeddings(seq, mask, include_user)

        if pos_seq is not None:
            pos_emb = self.pos_emb(pos_seq)
            base_emb += pos_emb

        feature_emb = self.get_feature_embeddings(feature_array, include_user, fusion_style)

        # Ensure feature_emb has the correct batch_size and seq_len, even if empty
        if feature_emb.shape[1] == 0 and seq.shape[1] > 0:  # If features are empty but sequence is not
            feature_emb = torch.zeros(
                seq.shape[0], seq.shape[1], self.args.shared_hidden_units, device=self.dev
            )

        if fusion_style == 'late':
            # HSTU style: sum base + features, then apply activation
            combined_emb = base_emb + feature_emb
            return F.gelu(combined_emb)
        else:  # early fusion
            # Hydra style: concatenate base + features, then project
            # The feature_emb from _get_early_fusion_features is ALREADY projected to shared_hidden_units
            combined_emb = torch.cat([base_emb, feature_emb], dim=2)

            # The final projection for Hydra is handled in the Hydra model's log2feats
            # self.hydra_projection is set by Hydra to project from (shared_hidden_units * 2) to hydra_hidden_units
            return combined_emb

    def save_item_embeddings(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """Save item embeddings for FAISS index"""
        from dataset2 import save_emb  # Ensure save_emb is correctly imported

        all_embs = []
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            # Create a dummy sequence for items. Each item is a sequence of length 1.
            # item_seq shape: (batch_size_chunk, 1)
            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(1)

            # Prepare batch_feat for each item in the chunk.
            # This needs to be a list of lists of dicts, where each inner list has one dict per item.
            batch_feat_processed = []
            for i in range(start_idx, end_idx):
                # feat_dict[i] is already a dict of features for item 'i'
                # We need to wrap it in a list to simulate a sequence of length 1
                batch_feat_processed.append([feat_dict[i]])

                # Use late fusion for consistent embeddings as recommended for retrieval
            batch_emb = self.get_complete_embeddings(
                item_seq, batch_feat_processed, include_user=False, fusion_style='late'
            ).squeeze(1)  # Squeeze the sequence dimension as it's length 1

            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))

    def get_item_embeddings(self, feat_dict, batch_size=1024):
        """
        Generates and returns the complete embeddings for all items in the vocabulary.
        This is intended for use during evaluation for full-vocabulary ranking.

        Args:
            feat_dict (dict): A dictionary mapping item_id (int) to its feature dictionary.
            batch_size (int): The batch size to use for processing to manage memory.

        Returns:
            torch.Tensor: A tensor of shape [item_num + 1, hidden_dim] containing all item embeddings.
        """
        all_embs = []

        # Create a tensor of all item IDs, from 1 to item_num
        all_item_ids = list(range(1, self.item_num + 1))

        for start_idx in tqdm(range(0, len(all_item_ids), batch_size),
                              desc="Generating all item embeddings for evaluation"):
            end_idx = min(start_idx + batch_size, len(all_item_ids))
            item_ids_chunk = all_item_ids[start_idx:end_idx]

            # Prepare tensors and feature arrays for the current chunk
            item_seq = torch.tensor(item_ids_chunk, device=self.dev).unsqueeze(1)  # Shape: [chunk_size, 1]

            # Prepare batch_feat for each item. It needs to be a list of lists of dicts.
            batch_feat_processed = []
            for item_id in item_ids_chunk:
                # Each item is a sequence of length 1, so its feature array is a list containing one dict.
                # We assume feat_dict uses integer keys. If not, convert item_id to str(item_id).
                features = feat_dict.get(item_id, {})
                batch_feat_processed.append([features])

            # Use the same fusion style that produces your final canonical item embeddings.
            # 'late' fusion is a good standard choice for retrieval/ranking embeddings as it's a simple sum.
            batch_emb = self.get_complete_embeddings(
                item_seq, batch_feat_processed, include_user=False, fusion_style='late'
            ).squeeze(1)  # Squeeze the sequence dimension (L=1)

            all_embs.append(batch_emb)

        # Concatenate all chunk embeddings
        final_embs = torch.cat(all_embs, dim=0)

        # Create a zero-vector for the padding index (ID 0)
        padding_emb = torch.zeros(1, self.args.shared_hidden_units, device=self.dev)

        # Combine padding embedding with the rest of the item embeddings
        # The final tensor will have shape [item_num + 1, hidden_dim]
        return torch.cat([padding_emb, final_embs], dim=0)


class SharedEmbeddingConfig:
    """Configuration for shared embeddings"""

    def __init__(self, args):
        # Use the larger of the two hidden dimensions to avoid information loss
        self.shared_hidden_units = max(
            getattr(args, 'hstu_hidden_units', 60),
            getattr(args, 'hydra_hidden_units', 60)
        )
        args.shared_hidden_units = self.shared_hidden_units

        print(f"Using shared hidden units: {self.shared_hidden_units}")




def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--maxlen', default=50, type=int)

    # Baseline Model construction
    parser.add_argument('--hstu_hidden_units', default=60, type=int)
    parser.add_argument('--hstu_num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--hstu_num_heads', default=2, type=int)
    parser.add_argument('--hstu_dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=1e-5, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--loss_type', default=None, type=str)
    parser.add_argument('--num_actions', default=5, type=int, help='Number of distinct action types.')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    parser.add_argument('--num_local_blocks', default=2, type=int)
    parser.add_argument('--num_global_blocks', default=2, type=int)
    parser.add_argument('--time_span', default=512, type=int)  # For time embeddings

    # NEW: Add in-batch negative sampling flag
    parser.add_argument('--use_inbatch_negatives', default= True, help='Use in-batch negative sampling')
    parser.add_argument('--temperature', default=0.05, type=float, help='Temperature for InfoNCE loss')
    parser.add_argument('--eval_batch_size', default=64, type=int)
    args = parser.parse_args()

    return args

def make_embd():
    args = get_args()
    data_path = os.environ.get('TRAIN_DATA_PATH')
    interaction_dir = os.environ.get('TRAIN_CKPT_PATH')
    dataset = MyDataset(data_path, interaction_dir, args)
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    SharedEmbeddingConfig(args)  # Set up shared_hidden_units
    shared_embeddings = SharedEmbeddingModule(usernum, itemnum, feat_statistics, feat_types, args,
                                              dataset.interaction_vocab_dict)

    embedding_path = Path(os.environ.get('TRAIN_CKPT_PATH'), "global_step embedding")
    embedding_path.mkdir(parents=True, exist_ok=True)
    torch.save(shared_embeddings.state_dict(), embedding_path / "embedding.pt")

    print("✅ HSTU training finished. Saving shared embeddings...")
    torch.save(shared_embeddings.state_dict(), "shared_embeddings.pt")
    print("💾 Shared embeddings saved to shared_embeddings.pt")
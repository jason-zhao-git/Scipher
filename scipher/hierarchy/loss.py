import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class MarginalizationLoss(nn.Module):
    """
    Calculates a hierarchical loss by combining a weighted leaf loss (for leaf-labeled data)
    and a parent loss (for all data).
    """
    def __init__(self, marginalization_df, parent_child_df, exclusion_df,
                 leaf_values, internal_values, mapping_dict, leaf_weight=8.0, device='cpu'):
        super().__init__()
        self.leaf_weight = leaf_weight
        self.device = device

        # --- Store integer indices for slicing ---
        self.leaf_indices_set = {mapping_dict[cid] for cid in leaf_values}

        # --- Sort DataFrame columns/index to match mapping_dict order ---
        all_cell_values_sorted = sorted(mapping_dict.keys(), key=lambda k: mapping_dict[k])
        leaf_values_sorted = sorted(leaf_values, key=lambda k: mapping_dict[k])
        internal_values_sorted = sorted(internal_values, key=lambda k: mapping_dict[k])

        # --- Re-index and convert to performant tensors once at initialization ---
        # Marginalization tensor (internal x leaf)
        marginalization_tensor = torch.FloatTensor(
            marginalization_df.loc[internal_values_sorted, leaf_values_sorted].values
        ).to(device)

        # Parent-child tensor (all_cells x internal_nodes)
        parent_child_tensor = torch.FloatTensor(
            parent_child_df.loc[all_cell_values_sorted, internal_values_sorted].values
        ).to(device)

        # Exclusion tensor (all_cells x internal_nodes)
        exclusion_tensor = torch.FloatTensor(
            exclusion_df.loc[all_cell_values_sorted, internal_values_sorted].values
        ).to(device)

        self.marginalization_tensor = marginalization_tensor
        self.parent_child_tensor = parent_child_tensor
        self.exclusion_tensor = exclusion_tensor

        self.criterion_leafs = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, outputs, y_batch):
        """The main forward pass for the loss function.

        NOTE: outputs should be LOGITS (raw scores), not probabilities.
        - CrossEntropyLoss expects logits and applies softmax internally
        - For parent loss, we convert to probabilities via softmax first
        """

        # --- 1. Leaf Loss (for leaf-labeled samples only) ---
        is_leaf_mask = torch.tensor([y.item() in self.leaf_indices_set for y in y_batch], device=self.device)
        leaf_outputs = outputs[is_leaf_mask]
        leaf_y_batch = y_batch[is_leaf_mask]

        loss_leafs = torch.tensor(0.0, device=self.device)
        if leaf_outputs.shape[0] > 0:
            # CrossEntropyLoss expects logits (raw scores), not probabilities
            # It applies softmax internally for numerical stability
            loss_leafs = self.criterion_leafs(leaf_outputs, leaf_y_batch) * self.leaf_weight

        # --- 2. Parent Loss (for all samples) ---

        # Convert logits to probabilities for marginalization
        # Shape stays the same: (batch_size, num_leaf_nodes)
        output_probs = F.softmax(outputs, dim=1)

        # Predicted parent probabilities (shape: batch_size, num_internal_nodes)
        # einsum('ij,kj->ki'): (internal, leaf) @ (batch, leaf) -> (batch, internal)
        output_internal_prob = torch.einsum('ij,kj->ki', self.marginalization_tensor, output_probs)
        # Clamp with epsilon to prevent log(0) and log(1-1) = -inf in BCE loss
        output_internal_prob = torch.clamp(output_internal_prob, 1e-7, 1 - 1e-7)

        # True parent labels (shape: batch_size, num_internal_nodes)
        # Tensors are already pre-sliced, so we can directly index them
        target_internal_prob = self.parent_child_tensor[y_batch] # Fast integer indexing

        # Exclusion mask (shape: batch_size, num_internal_nodes)
        exclusion_mask = self.exclusion_tensor[y_batch] # Fast integer indexing

        # Weighted BCE Loss for internal nodes
        loss_parents = F.binary_cross_entropy(
            output_internal_prob,
            target_internal_prob,
            weight=exclusion_mask,
            reduction='mean'
        )

        # --- Total Loss ---
        total_loss = loss_leafs + loss_parents

        return total_loss, loss_leafs, loss_parents

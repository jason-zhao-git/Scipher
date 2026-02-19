"""ESM-2 gene-level embeddings (1280-dim for 650M model).

Uses Meta's ESM-2 protein language model to generate per-protein
embeddings by mean-pooling last-layer residue representations.

Mirrors the ProteinBERTEmbedder interface: embed_sequence, embed_readthrough,
generate_all_embeddings.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ESM-2 model variants: name -> (num_layers, embedding_dim)
ESM2_MODELS = {
    "esm2_t6_8M_UR50D": (6, 320),
    "esm2_t12_35M_UR50D": (12, 480),
    "esm2_t30_150M_UR50D": (30, 640),
    "esm2_t33_650M_UR50D": (33, 1280),
    "esm2_t36_3B_UR50D": (36, 2560),
    "esm2_t48_15B_UR50D": (48, 5120),
}


class ESM2Embedder:
    """Generate gene-level embeddings using ESM-2.

    Extracts per-protein embeddings by mean-pooling the last-layer
    residue representations (excluding BOS/EOS special tokens).
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: str = None,
        max_length: int = 1022,
    ):
        """
        Args:
            model_name: ESM-2 model variant (default: 650M, 1280-dim)
            device: 'cuda', 'cpu', or None for auto-detect
            max_length: Max residues per sequence. ESM-2 adds BOS+EOS tokens,
                so token count = max_length + 2. Sequences longer than this
                are truncated from the C-terminus.
        """
        import esm

        if model_name not in ESM2_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(ESM2_MODELS.keys())}"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model_name = model_name

        num_layers, self.embedding_dim = ESM2_MODELS[model_name]

        logger.info(f"Loading {model_name}...")
        loader = getattr(esm.pretrained, model_name)
        self.model, self.alphabet = loader()
        self.model = self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.repr_layer = num_layers

        logger.info(
            f"ESM-2 loaded on {self.device} "
            f"(layers={num_layers}, embedding_dim={self.embedding_dim})"
        )

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Generate embedding for a single protein sequence.

        Returns:
            1-D array of shape (embedding_dim,)
        """
        sequence = sequence[: self.max_length]

        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(
                batch_tokens,
                repr_layers=[self.repr_layer],
                return_contacts=False,
            )

        token_repr = results["representations"][self.repr_layer]
        # Mean-pool over residue positions, skipping BOS (idx 0) and EOS
        seq_len = min(len(sequence), self.max_length)
        embedding = token_repr[0, 1 : seq_len + 1].mean(dim=0).cpu().numpy()
        return embedding

    def embed_batch(
        self,
        sequences: List[Tuple[str, str]],
    ) -> Dict[str, np.ndarray]:
        """Embed multiple sequences in a single forward pass.

        Args:
            sequences: List of (gene_name, sequence) tuples

        Returns:
            Dict mapping gene_name to embedding array
        """
        truncated = [(name, seq[: self.max_length]) for name, seq in sequences]

        _, _, batch_tokens = self.batch_converter(truncated)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(
                batch_tokens,
                repr_layers=[self.repr_layer],
                return_contacts=False,
            )

        token_repr = results["representations"][self.repr_layer]

        embeddings = {}
        for i, (name, seq) in enumerate(truncated):
            seq_len = len(seq)
            embeddings[name] = (
                token_repr[i, 1 : seq_len + 1].mean(dim=0).cpu().numpy()
            )

        return embeddings

    def embed_readthrough(
        self,
        component_sequences: Tuple[str, ...],
        strategy: str = "concat",
    ) -> np.ndarray:
        """Generate embedding for readthrough transcript (fusion gene).

        Args:
            component_sequences: Tuple of component gene sequences
            strategy: 'concat' (default), 'mean', 'max', or 'weighted'
        """
        if strategy == "concat":
            return self.embed_sequence("".join(component_sequences))

        component_embeddings = np.array(
            [self.embed_sequence(seq) for seq in component_sequences]
        )

        if strategy == "mean":
            return np.mean(component_embeddings, axis=0)
        elif strategy == "max":
            return np.max(component_embeddings, axis=0)
        elif strategy == "weighted":
            lengths = np.array([len(seq) for seq in component_sequences])
            weights = lengths / lengths.sum()
            return np.average(component_embeddings, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def generate_all_embeddings(
        self,
        gene_to_sequence: Dict[str, Union[str, Tuple[str, ...]]],
        readthrough_strategy: str = "concat",
        cache_path: Path = None,
        batch_size: int = 8,
    ) -> Dict[str, np.ndarray]:
        """Generate embeddings for all genes.

        Uses batched inference for regular genes (much faster than one-by-one).
        Readthrough genes are processed individually since they need special
        aggregation strategies.

        Args:
            gene_to_sequence: Dict mapping gene symbols to sequences (str)
                or component sequences (tuple for readthrough genes)
            readthrough_strategy: Strategy for readthrough transcripts
            cache_path: Optional path to save/load cached embeddings pickle
            batch_size: Sequences per forward pass (reduce if OOM)

        Returns:
            Dict mapping gene symbols to embedding arrays
        """
        if cache_path and cache_path.exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        regular = {k: v for k, v in gene_to_sequence.items() if isinstance(v, str)}
        readthrough = {k: v for k, v in gene_to_sequence.items() if isinstance(v, tuple)}

        logger.info(
            f"Generating ESM-2 embeddings: "
            f"{len(regular)} regular + {len(readthrough)} readthrough genes"
        )

        gene_to_embedding = {}

        # Batch regular genes for efficiency
        regular_items = list(regular.items())
        for i in tqdm(
            range(0, len(regular_items), batch_size),
            desc=f"ESM-2 ({self.model_name})",
        ):
            batch = regular_items[i : i + batch_size]
            batch_results = self.embed_batch(batch)
            gene_to_embedding.update(batch_results)

        # Readthrough genes one-by-one (need per-component logic)
        for gene, seqs in tqdm(readthrough.items(), desc="Readthrough genes"):
            gene_to_embedding[gene] = self.embed_readthrough(
                seqs, strategy=readthrough_strategy
            )

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(gene_to_embedding, f)
            logger.info(f"Cached {len(gene_to_embedding)} embeddings to {cache_path}")

        return gene_to_embedding

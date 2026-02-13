"""ProteinBERT gene-level embeddings (512-dim).

Copied from funcCell/src/model/proteinbert_embeddings.py
- Removed hardcoded paths (model_dir now passed as argument)
- Kept embedding extraction from internal layer 'global-merge2-norm-block6'
- Kept readthrough transcript handling strategies
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from proteinbert import load_pretrained_model

logger = logging.getLogger(__name__)

ZENODO_URL = "https://zenodo.org/records/10371965/files/full_go_epoch_92400_sample_23500000.pkl?download=1"
MODEL_FILENAME = "full_go_epoch_92400_sample_23500000.pkl"


class ProteinBERTEmbedder:
    """Generate gene-level embeddings using ProteinBERT.

    Extracts 512-dim global protein representations from the internal
    layer 'global-merge2-norm-block6', NOT from model output heads.
    """

    EMBEDDING_DIM = 512

    def __init__(self, model_dir: Path, seq_len: int = 2048):
        """
        Args:
            model_dir: Directory containing ProteinBERT model weights
            seq_len: Maximum sequence length for encoding
        """
        model_path = model_dir / MODEL_FILENAME

        if not model_path.exists():
            logger.info("Model not found locally. Downloading from Zenodo...")
            model_dir.mkdir(parents=True, exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(ZENODO_URL, model_path)
            logger.info(f"Downloaded to {model_path}")

        self.pretrained_model_generator, self.input_encoder = load_pretrained_model(
            local_model_dump_dir=str(model_dir),
            local_model_dump_file_name=model_path.name,
            download_model_dump_if_not_exists=False,
        )

        base_model = self.pretrained_model_generator.create_model(seq_len)
        final_global_layer = base_model.get_layer('global-merge2-norm-block6')

        self.model = tf.keras.Model(
            inputs=base_model.inputs,
            outputs=final_global_layer.output,
        )

        self.seq_len = seq_len
        logger.info(f"ProteinBERT loaded (seq_len={seq_len}, embedding_dim={self.EMBEDDING_DIM})")

    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Generate 512-dim embedding for a single protein sequence."""
        if len(sequence) > self.seq_len - 10:
            import random
            start = random.randint(0, len(sequence) - (self.seq_len - 10))
            sequence = sequence[start:start + (self.seq_len - 10)]

        encoded_x = self.input_encoder.encode_X([sequence], self.seq_len)
        global_embedding = self.model.predict(encoded_x, batch_size=1, verbose=0)
        return global_embedding[0]

    def embed_readthrough(
        self,
        component_sequences: Tuple[str, ...],
        strategy: str = 'concat',
    ) -> np.ndarray:
        """Generate embedding for readthrough transcript (fusion gene).

        Args:
            component_sequences: Tuple of component gene sequences
            strategy: 'concat' (default), 'mean', 'max', or 'weighted'
        """
        if strategy == 'concat':
            return self.embed_sequence(''.join(component_sequences))

        component_embeddings = np.array([self.embed_sequence(seq) for seq in component_sequences])

        if strategy == 'mean':
            return np.mean(component_embeddings, axis=0)
        elif strategy == 'max':
            return np.max(component_embeddings, axis=0)
        elif strategy == 'weighted':
            lengths = np.array([len(seq) for seq in component_sequences])
            weights = lengths / lengths.sum()
            return np.average(component_embeddings, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def generate_all_embeddings(
        self,
        gene_to_sequence: Dict[str, Union[str, Tuple[str, ...]]],
        readthrough_strategy: str = 'concat',
        cache_path: Path = None,
    ) -> Dict[str, np.ndarray]:
        """Generate embeddings for all genes.

        Args:
            gene_to_sequence: Dict mapping gene symbols to sequences
            readthrough_strategy: Strategy for readthrough transcripts
            cache_path: Optional path to save/load cached embeddings

        Returns:
            Dict mapping gene symbols to 512-dim embeddings
        """
        if cache_path and cache_path.exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        regular = {k: v for k, v in gene_to_sequence.items() if isinstance(v, str)}
        readthrough = {k: v for k, v in gene_to_sequence.items() if isinstance(v, tuple)}

        logger.info(f"Generating embeddings: {len(regular)} regular + {len(readthrough)} readthrough genes")

        gene_to_embedding = {}

        for gene, seq in tqdm(regular.items(), desc="Regular genes"):
            gene_to_embedding[gene] = self.embed_sequence(seq)

        for gene, seqs in tqdm(readthrough.items(), desc="Readthrough genes"):
            gene_to_embedding[gene] = self.embed_readthrough(seqs, strategy=readthrough_strategy)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(gene_to_embedding, f)
            logger.info(f"Cached embeddings to {cache_path}")

        return gene_to_embedding

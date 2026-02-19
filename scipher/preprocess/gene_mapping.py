"""Map genes to protein sequences using MyGene and UniProt APIs.

Copied from funcCell/src/preprocess_data/gene_mapping.py
- Removed dependency on funcCell config and utils (parameters passed explicitly)
- Saving/reporting logic removed; caller handles persistence
"""

import logging
import time
from typing import Dict, List, Tuple, Union

import requests
import mygene
from tqdm import tqdm

logger = logging.getLogger(__name__)


def query_mygene(gene_symbols: List[str], batch_size: int = 100) -> Dict[str, str]:
    """
    Query MyGene.info to map gene symbols to UniProt IDs.

    Args:
        gene_symbols: List of gene symbols
        batch_size: Number of genes per batch request

    Returns:
        Dictionary mapping gene symbol to UniProt ID
    """
    logger.info(f"Querying MyGene.info for {len(gene_symbols)} genes...")

    mg = mygene.MyGeneInfo()
    gene_to_uniprot = {}

    for i in tqdm(range(0, len(gene_symbols), batch_size), desc="Querying MyGene"):
        batch = gene_symbols[i:i + batch_size]

        try:
            results = mg.querymany(
                batch,
                scopes='symbol',
                fields='uniprot.Swiss-Prot',
                species='human',
                returnall=True,
            )

            for result in results['out']:
                gene_symbol = result.get('query')
                if 'uniprot' in result and 'Swiss-Prot' in result['uniprot']:
                    uniprot_id = result['uniprot']['Swiss-Prot']
                    if isinstance(uniprot_id, list):
                        uniprot_id = uniprot_id[0]
                    gene_to_uniprot[gene_symbol] = uniprot_id

            time.sleep(0.2)

        except Exception as e:
            logger.warning(f"Error querying batch {i//batch_size + 1}: {e}")
            continue

    logger.info(f"Mapped {len(gene_to_uniprot)}/{len(gene_symbols)} genes to UniProt IDs")
    return gene_to_uniprot


def fetch_uniprot_sequences(
    uniprot_ids: List[str],
    batch_size: int = 100,
    max_retries: int = 3,
    timeout: int = 30,
) -> Dict[str, str]:
    """
    Fetch protein sequences from UniProt REST API.

    Args:
        uniprot_ids: List of UniProt IDs
        batch_size: Number of IDs per batch request
        max_retries: Maximum retry attempts for failed requests
        timeout: Request timeout in seconds

    Returns:
        Dictionary mapping UniProt ID to protein sequence
    """
    logger.info(f"Fetching protein sequences from UniProt for {len(uniprot_ids)} IDs...")

    uniprot_to_seq = {}
    base_url = "https://rest.uniprot.org/uniprotkb/stream"

    for i in tqdm(range(0, len(uniprot_ids), batch_size), desc="Fetching UniProt"):
        batch = uniprot_ids[i:i + batch_size]
        query = " OR ".join([f"accession:{uid}" for uid in batch])

        for attempt in range(max_retries):
            try:
                params = {
                    'query': query,
                    'format': 'fasta',
                    'compressed': 'false',
                }

                response = requests.get(base_url, params=params, timeout=timeout)
                response.raise_for_status()

                # Parse FASTA response
                fasta_text = response.text
                current_id = None
                current_seq = []

                for line in fasta_text.split('\n'):
                    if line.startswith('>'):
                        if current_id and current_seq:
                            uniprot_to_seq[current_id] = ''.join(current_seq)
                        parts = line.split('|')
                        if len(parts) >= 2:
                            current_id = parts[1]
                        current_seq = []
                    elif line.strip():
                        current_seq.append(line.strip())

                if current_id and current_seq:
                    uniprot_to_seq[current_id] = ''.join(current_seq)

                break

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for batch {i//batch_size + 1}: {e}")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to fetch batch {i//batch_size + 1} after {max_retries} attempts")

        time.sleep(0.3)

    logger.info(f"Fetched {len(uniprot_to_seq)}/{len(uniprot_ids)} protein sequences")
    return uniprot_to_seq


def map_genes_to_sequences(
    gene_list: List[str],
    gene_descriptions: Dict[str, str] = None,
    batch_size: int = 100,
    max_retries: int = 3,
    timeout: int = 30,
) -> Tuple[Dict[str, Union[str, Tuple[str, ...]]], List[str]]:
    """
    Map gene symbols to protein sequences, including readthrough transcripts.

    Args:
        gene_list: List of gene symbols
        gene_descriptions: Dict mapping gene symbol to description (for readthrough detection)
        batch_size: Genes per API batch
        max_retries: Max retries for failed requests
        timeout: Request timeout in seconds

    Returns:
        Tuple of (gene_to_sequence dict, missing_genes list)
        - gene_to_sequence: Maps gene symbol to sequence (str) or component sequences (tuple)
        - missing_genes: Genes that could not be mapped
    """
    logger.info("=" * 60)
    logger.info("Mapping genes to protein sequences")
    logger.info("=" * 60)

    # Step 1: Gene symbol -> UniProt ID
    gene_to_uniprot = query_mygene(gene_list, batch_size=batch_size)

    # Step 2: UniProt ID -> Protein sequence
    uniprot_ids = list(gene_to_uniprot.values())
    uniprot_to_seq = fetch_uniprot_sequences(
        uniprot_ids, batch_size=batch_size, max_retries=max_retries, timeout=timeout
    )

    # Step 3: Combine: Gene -> Sequence
    gene_to_sequence = {}
    for gene, uniprot_id in gene_to_uniprot.items():
        if uniprot_id in uniprot_to_seq:
            gene_to_sequence[gene] = uniprot_to_seq[uniprot_id]

    # Step 4: Handle readthrough transcripts (fusion genes)
    missing_genes = [g for g in gene_list if g not in gene_to_sequence]

    if gene_descriptions:
        readthrough_genes = [
            g for g in missing_genes
            if g in gene_descriptions and 'readthrough' in gene_descriptions[g].lower()
        ]
    else:
        logger.warning("No gene descriptions provided - using fallback hyphen detection for readthrough genes")
        readthrough_genes = [g for g in missing_genes if '-' in g and not g.startswith('HLA-')]

    if readthrough_genes:
        logger.info(f"Handling {len(readthrough_genes)} readthrough transcripts...")

        component_genes = set()
        for rt_gene in readthrough_genes:
            component_genes.update(rt_gene.split('-'))

        component_to_uniprot = query_mygene(list(component_genes), batch_size=batch_size)
        component_uniprot_ids = list(component_to_uniprot.values())
        component_uniprot_to_seq = fetch_uniprot_sequences(
            component_uniprot_ids, batch_size=batch_size, max_retries=max_retries, timeout=timeout
        )

        component_to_seq = {}
        for gene, uniprot_id in component_to_uniprot.items():
            if uniprot_id in component_uniprot_to_seq:
                component_to_seq[gene] = component_uniprot_to_seq[uniprot_id]

        readthrough_mapped = 0
        for rt_gene in readthrough_genes:
            components = rt_gene.split('-')
            component_seqs = [component_to_seq.get(c) for c in components if c in component_to_seq]
            if component_seqs:
                gene_to_sequence[rt_gene] = tuple(component_seqs)
                readthrough_mapped += 1

        logger.info(f"  Mapped {readthrough_mapped}/{len(readthrough_genes)} readthrough genes")

    missing_genes = [g for g in gene_list if g not in gene_to_sequence]
    n_mapped = len(gene_to_sequence)
    logger.info(f"Final: {n_mapped}/{len(gene_list)} genes mapped ({100*n_mapped/len(gene_list):.1f}%)")

    return gene_to_sequence, missing_genes

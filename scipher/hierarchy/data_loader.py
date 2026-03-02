import cellxgene_census
import pandas as pd
from scipher.hierarchy.ontology_utils import get_sub_DAG
from scipher.hierarchy.config import CENSUS_VERSION


def load_filtered_cell_metadata(cl, root_cl_id: str, min_cell_count: int = 5000) -> pd.DataFrame:
    """
    Loads cell metadata from the CellXGene Census, filters for descendants of a given
    root CL ID with a minimum count, and returns the filtered metadata.

    Args:
        cl (pronto.Ontology): The loaded Cell Ontology object.
        root_cl_id (str): The root Cell Ontology ID (e.g., "CL:0000988") to define the subgraph.
        min_cell_count (int): The minimum number of cells for a cell type to be included.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered cell metadata.
    """
    # Get all descendants of the root CL ID
    print(f"Fetching descendants of {root_cl_id}...")

    descendant_cell_types = set()
    for sub in get_sub_DAG(cl, root_cl_id):
        descendant_cell_types.add(sub.id)

    if not descendant_cell_types:
        print(f"No descendants found for {root_cl_id}. Aborting.")
        return pd.DataFrame()

    print("Connecting to CellXGene Census...")
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        experiment = census["census_data"]["homo_sapiens"]

        print("Reading cell metadata from 10x v3 primary experiments to filter cell types...")
        # First filter for 10x v3 primary experiments, then count cells
        exp_pd = experiment.obs.read(
            value_filter='assay == "10x 3\' v3" and is_primary_data == True',
            column_names=["cell_type_ontology_term_id"]
        ).concat().to_pandas()
        cell_type_counts = exp_pd["cell_type_ontology_term_id"].value_counts()

        # Filter for cell types with a minimum count
        good_mask = cell_type_counts > min_cell_count
        valid_cell_types = cell_type_counts[good_mask].keys().tolist()

        # Find the intersection with our descendant cell list
        intersection_cell_types = list(set(valid_cell_types) & descendant_cell_types)

        print(f"Found {len(intersection_cell_types)} cell types with > {min_cell_count} cells.")

        if not intersection_cell_types:
            print("No cell types matching the criteria. Aborting.")
            return pd.DataFrame()

        # Build the value filter for the final query
        # Use .format() to handle quotes in the assay name safely
        obs_val_filter = '''assay == "10x 3' v3" and is_primary_data == True and cell_type_ontology_term_id in {}'''.format(intersection_cell_types)

        print("Querying for final cell metadata...")
        cell_obs_metadata = (
            experiment.obs.read(value_filter=obs_val_filter,
                                   column_names=['cell_type_ontology_term_id']).concat().to_pandas()
        )

    print("Finished loading and filtering cell metadata.")
    return cell_obs_metadata

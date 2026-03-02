import pickle
import pandas as pd
import pronto
from scipher.hierarchy.paths import HIERARCHY_DATA_DIR

_ontology = None

def load_ontology():
    """
    Loads the cached pronto.Ontology object from the pickle file.
    Caches the object in memory for the session to avoid repeated loading.

    Returns:
        pronto.Ontology: The loaded Cell Ontology object.
    """
    global _ontology
    if _ontology is None:
        ontology_cache_path = HIERARCHY_DATA_DIR / "ontology.pkl"
        print(f"Loading cached ontology from {ontology_cache_path}...")
        try:
            with open(ontology_cache_path, "rb") as f:
                _ontology = pickle.load(f)
            print("Ontology loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Ontology cache file not found at {ontology_cache_path}.")
            print("Please run `python -m scipher.hierarchy.cache_ontology` first.")
            return None
    return _ontology

def get_sub_DAG(cl, cl_number: str) -> set:
    """
    Given a CL number, returns a set of all its downstream nodes (descendants).

    Args:
        cl (pronto.Ontology): The loaded Cell Ontology object.
        cl_number (str): The CL number (e.g., "CL:0000000") for which to find descendants.

    Returns:
        set: A set of all downstream nodes.
    """
    try:
        term = cl[cl_number]
        return {sub for sub in term.subclasses(with_self=True)}
    except KeyError:
        print(f"Warning: Term ID {cl_number} not found in ontology.")
        return set()

def get_cell_info(cl, cl_ids):
    """
    Given a CL ID or a list of CL IDs, returns a DataFrame with their info.

    Args:
        cl (pronto.Ontology): The loaded Cell Ontology object.
        cl_ids (str or list): A single CL ID or a list of CL IDs.

    Returns:
        pd.DataFrame: A DataFrame containing the info for the requested CL IDs.
    """
    if isinstance(cl_ids, str):
        cl_ids = [cl_ids]

    results = []
    for cl_id in cl_ids:
        try:
            term = cl[cl_id]
            results.append({
                'cl_id': term.id,
                'cl_name': term.name,
                'is_leaf': term.is_leaf()
            })
        except KeyError:
            print(f"Warning: Term ID {cl_id} not found in ontology.")
            continue

    return pd.DataFrame(results).set_index('cl_id')

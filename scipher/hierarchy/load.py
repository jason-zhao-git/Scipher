import pandas as pd
import pickle
from scipher.hierarchy.paths import get_data_folder


def load_prebuilt_hierarchy(date, root_cl_id="CL:0000988"):
    """
    Load prebuilt hierarchy artifacts from saved CSVs and pickles.

    Args:
        date (str): Date string in 'YYYY-MM-DD' format (e.g., '2026-01-29').
        root_cl_id (str): Root CL ID (e.g., 'CL:0000988' for blood cells,
                          'CL:0000000' for all human cells).

    Returns:
        tuple: (mapping_dict, leaf_values, internal_values,
                marginalization_df, parent_child_df, exclusion_df)
    """
    folder = get_data_folder(date, root_cl_id)
    if not folder.exists():
        raise FileNotFoundError(
            f"No prebuilt hierarchy data at {folder}. "
            f"Run scripts/run_preprocessing.py first."
        )

    # Load DataFrames
    marginalization_df = pd.read_csv(folder / f"{date}_marginalization_df.csv", index_col=0)
    parent_child_df = pd.read_csv(folder / f"{date}_parent_child_df.csv", index_col=0)
    exclusion_df = pd.read_csv(folder / f"{date}_exclusion_df.csv", index_col=0)

    # Load mapping_dict (saved as DataFrame with CL IDs as index, single column of ints)
    mapping_dict_df = pd.read_csv(folder / f"{date}_mapping_dict_df.csv", index_col=0)
    mapping_dict = {cl_id: int(row.iloc[0]) for cl_id, row in mapping_dict_df.iterrows()}

    # Load leaf and internal values
    with open(folder / f"{date}_leaf_values.pkl", "rb") as fp:
        leaf_values = pickle.load(fp)
    with open(folder / f"{date}_internal_values.pkl", "rb") as fp:
        internal_values = pickle.load(fp)

    print(f"Loaded hierarchy from {folder}: {len(leaf_values)} leaves, {len(internal_values)} internal")
    return mapping_dict, leaf_values, internal_values, marginalization_df, parent_child_df, exclusion_df

import pandas as pd
import pickle
from datetime import datetime
from scipher.hierarchy.ontology_utils import load_ontology
from scipher.hierarchy.data_loader import load_filtered_cell_metadata
from scipher.hierarchy.preprocess_ontology import preprocess_data_ontology
from scipher.hierarchy.paths import HIERARCHY_DATA_DIR
from scipher.hierarchy.config import ROOT_CL_ID

def main():
    """
    Main function to run the full data preprocessing pipeline.
    """
    print("Starting data preprocessing pipeline...")

    # 1. Load the cached ontology object
    cl = load_ontology()
    if cl is None:
        return

    root_cl_id = ROOT_CL_ID

    # 2. Load filtered cell metadata from CellXGene Census
    cell_obs_metadata = load_filtered_cell_metadata(cl, root_cl_id=root_cl_id)

    if cell_obs_metadata.empty:
        print("No cell metadata loaded. Aborting pipeline.")
        return

    # 3. Preprocess the ontology and cell data
    target_column = 'cell_type_ontology_term_id'

    print("Starting ontology preprocessing...")
    mapping_dict, leaf_values, internal_values, \
        marginalization_df, parent_child_df, exclusion_df = preprocess_data_ontology(
            cl, cell_obs_metadata, target_column,
            upper_limit=root_cl_id,
            cl_only=True, include_leafs=False
        )

    print(f"Preprocessing complete. Found {len(leaf_values)} leaf values and {len(internal_values)} internal values.")

    # 4. Save the preprocessed artifacts
    today = datetime.today().strftime('%Y-%m-%d')
    cl_folder = root_cl_id.replace(":", "")  # CL:0000988 -> CL0000988
    today_folder = datetime.today().strftime('%m-%d')

    output_dir = HIERARCHY_DATA_DIR / f"{cl_folder}_{today_folder}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving preprocessed data to {output_dir}...")

    # Save marginalization_df
    marginalization_df.to_csv(output_dir / f"{today}_marginalization_df.csv")

    # Save parent_child_df
    parent_child_df.to_csv(output_dir / f"{today}_parent_child_df.csv")

    # Save exclusion_df
    exclusion_df.to_csv(output_dir / f"{today}_exclusion_df.csv")

    # Save mapping_dict
    mapping_dict_df = pd.DataFrame.from_dict(mapping_dict, orient='index')
    mapping_dict_df.to_csv(output_dir / f"{today}_mapping_dict_df.csv")

    # Save leaf_values and internal_values
    with open(output_dir / f"{today}_leaf_values.pkl", "wb") as fp:
        pickle.dump(leaf_values, fp)
    with open(output_dir / f"{today}_internal_values.pkl", "wb") as fp:
        pickle.dump(internal_values, fp)

    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()

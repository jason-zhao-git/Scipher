import pronto
import pickle
import os
from scipher.hierarchy.paths import HIERARCHY_DATA_DIR

def main():
    """
    Main function to load the Cell Ontology and cache the pronto.Ontology object.
    """
    output_dir = HIERARCHY_DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    ontology_cache_path = output_dir / "ontology.pkl"

    print("Starting ontology caching process...")

    # 1. Load the Cell Ontology using pronto
    # This will download 'cl.owl' from the OBO library if not found locally.
    print("Loading Cell Ontology (will download if necessary)...")
    try:
        cl_ontology = pronto.Ontology.from_obo_library('cl.owl')
        print("Ontology loaded successfully.")
    except Exception as e:
        print(f"Failed to load ontology: {e}")
        return

    # 2. Save the pronto.Ontology object as a pickle file
    print(f"Saving ontology object to {ontology_cache_path}...")
    with open(ontology_cache_path, "wb") as f:
        pickle.dump(cl_ontology, f)
    print("Ontology object cached successfully.")

if __name__ == "__main__":
    main()

import pandas as pd
import pronto


def get_parent_nodes(all_cell_values, cl, upper_limit=None, cl_only=False, include_leafs=False):
    all_parent_nodes = []
    for target in all_cell_values:
        try:
            for term in cl[target].superclasses(with_self=include_leafs):
                all_parent_nodes.append(term.id)
        except KeyError:
            continue

    all_parent_nodes = sorted(list(set(all_parent_nodes)))

    if cl_only:
        all_parent_nodes = [x for x in all_parent_nodes if x.startswith('CL')]

    if upper_limit is not None:
        try:
            upper_limit_nodes = [term.id for term in cl[upper_limit].superclasses(with_self=False)]
            all_parent_nodes = [x for x in all_parent_nodes if x not in upper_limit_nodes]
        except KeyError:
            pass

    return all_parent_nodes


def build_marginalization_df(all_parent_nodes, all_leaf_values, cl):
    """
    Creates the marginalization DataFrame to calculate predicted internal node probabilities.

    Function: To calculate the *predicted* probability of internal nodes by summing
              the probabilities of their descendant leaf nodes.
    Shape: (Number of Internal Nodes, Number of Leaf Nodes)
    Index (Rows): Internal Node CL numbers.
    Columns: Leaf Node CL numbers.
    Values: df.loc[internal_node, leaf_node] = 1 if the leaf is a descendant
            of the internal node, 0 otherwise.
    """
    marginalization_df = pd.DataFrame(data=0, index=all_parent_nodes, columns=all_leaf_values)
    for cell_id in marginalization_df.index:
        try:
            for term in cl[cell_id].subclasses(with_self=True):
                if term.id in marginalization_df.columns:
                    marginalization_df.loc[cell_id, [term.id]] = 1
        except KeyError:
            continue
    return marginalization_df


def build_parent_child_mask(all_cell_values, internal_values, cl, include_self=False):
    """
    Creates the parent-child DataFrame to find the true ancestors for any cell.

    Function: To find the *true* ancestors for any given ground-truth cell label.
              This is used to create the ground truth vector for the BCE loss.
    Shape: (Number of All Cells, Number of Internal Nodes)
    Index (Rows): Child CL numbers (the ground-truth label).
    Columns: Parent CL numbers (internal nodes only).
    Values: df.loc[child_node, parent_node] = 1 if parent_node is an ancestor
            of child_node (or is the child_node itself), 0 otherwise.
    """
    mask_df = pd.DataFrame(0, index=all_cell_values, columns=internal_values)

    for child_id in all_cell_values:
        try:
            ancestors = cl[child_id].superclasses(with_self=include_self)
            # Filter for parents that are in our list of internal nodes
            parent_ids = [p.id for p in ancestors if p.id in internal_values]
            if parent_ids:
                mask_df.loc[child_id, parent_ids] = 1
        except KeyError:
            continue
    return mask_df


def build_exclusion_df(all_cell_values, internal_values, cl):
    """
    Creates the exclusion DataFrame to mask loss calculations.

    Function: To create a weight mask that *excludes* loss calculations for
              descendants of a ground-truth label. This is used when the
              ground-truth is an internal node to avoid penalizing the model
              for its predictions on the children.
    Shape: (Number of All Cells, Number of Internal Nodes)
    Index (Rows): Ground-truth cell CL numbers.
    Columns: All other cell CL numbers (internal nodes only).
    Values: df.loc[true_label, other_node] = 0 if other_node is a descendant
            of true_label. Otherwise, the value is 1. A cell is not its own
            descendant, so the diagonal is 1.
    """
    exclusion_df = pd.DataFrame(1, index=all_cell_values, columns=internal_values)

    for cell_id in all_cell_values:  # This is the "true label" row
        try:
            descendants = cl[cell_id].subclasses(with_self=False)
            # Find which of the descendants are in our internal node columns
            descendant_ids_in_cols = [d.id for d in descendants if d.id in internal_values]
            if descendant_ids_in_cols:
                exclusion_df.loc[cell_id, descendant_ids_in_cols] = 0
        except KeyError:
            continue
    return exclusion_df


def preprocess_data_ontology(cl, labels, target_column, upper_limit=None, cl_only=False, include_leafs=False):
    """
    This function performs preprocessing on an AnnData object to prepare it for modelling.
    It returns all the data structures needed by the training pipeline.

    Returns:
        mapping_dict (dict):
            Maps CL numbers to integer indices. The dictionary is structured so that
            all effective leaf nodes are indexed first (from 0 to N_leaves-1), followed by
            all internal nodes with children.
        leaf_values (list):
            A sorted list of CL numbers for all EFFECTIVE leaf nodes in the dataset.
            This includes both true ontology leaves AND internal nodes that have no
            children within the dataset (childless internal nodes).
        internal_values (list):
            A sorted list of CL numbers for internal nodes that have at least one
            child (descendant) within the dataset.
        marginalization_df (pd.DataFrame):
            DataFrame for calculating predicted internal node probabilities.
            Shape: (internal_nodes, leaf_nodes).
        parent_child_df (pd.DataFrame):
            DataFrame for finding the true ancestors of any given cell.
            Shape: (all_cells, internal_nodes).
        exclusion_df (pd.DataFrame):
            DataFrame for masking loss calculations for descendants of internal nodes.
            Shape: (all_cells, internal_nodes).
    """
    all_cell_values_from_data = labels[target_column].astype('category').unique().tolist()

    # Step 1: Initial classification using full ontology
    ontology_leaf_values = [term_id for term_id in all_cell_values_from_data if cl[term_id].is_leaf()]
    ontology_internal_values = [term_id for term_id in all_cell_values_from_data if not cl[term_id].is_leaf()]

    # Step 2: Find internal nodes that actually have children IN THE DATASET
    # An internal node has children if any other node in dataset lists it as an ancestor
    internal_with_children = set()
    for node_id in all_cell_values_from_data:
        for ancestor in cl[node_id].superclasses(with_self=False):
            if ancestor.id in ontology_internal_values:
                internal_with_children.add(ancestor.id)

    # Step 3: Childless internal nodes become effective leaves
    childless_internal = [nid for nid in ontology_internal_values if nid not in internal_with_children]

    # Step 4: Final classification - SORTED for consistent ordering
    leaf_values = sorted(ontology_leaf_values + childless_internal)
    internal_values = sorted(internal_with_children)

    print(f"Ontology classification: {len(ontology_leaf_values)} leaves, {len(ontology_internal_values)} internal")
    print(f"Reclassified {len(childless_internal)} childless internal nodes as effective leaves")
    print(f"Final: {len(leaf_values)} effective leaves, {len(internal_values)} internal with children")

    # Create the final ordered list of all cells and the mapping_dict from it
    all_cell_values = leaf_values + internal_values
    mapping_dict = {term_id: i for i, term_id in enumerate(all_cell_values)}

    labels['encoded_labels'] = labels[target_column].map(mapping_dict)

    # Build the ontology dataframe for marginalization (internal nodes x leaf nodes)
    marginalization_df = build_marginalization_df(internal_values, leaf_values, cl)
    print(f"{len(all_cell_values)} cell types in dataset: {len(leaf_values)} effective leaves, {len(internal_values)} internal with children")

    # Build the optimized (all_cells x internal_nodes) matrices directly
    parent_child_df = build_parent_child_mask(all_cell_values, internal_values, cl, include_self=True)
    exclusion_df = build_exclusion_df(all_cell_values, internal_values, cl)

    return mapping_dict, leaf_values, internal_values, marginalization_df, parent_child_df, exclusion_df

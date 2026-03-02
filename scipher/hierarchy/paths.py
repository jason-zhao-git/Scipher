from pathlib import Path

# Resolve the project root by finding the directory containing the 'scipher' folder.
# This makes the path independent of where the script is run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Hierarchy-specific data lives under data/hierarchy/
HIERARCHY_DATA_DIR = PROJECT_ROOT / "data" / "hierarchy" / "processed"


def get_data_folder(date: str, root_cl_id: str = "CL:0000988") -> Path:
    """
    Get the path to the processed hierarchy data folder for a given date and root CL ID.

    Args:
        date (str): Date string in 'YYYY-MM-DD' format (e.g., '2025-10-24')
        root_cl_id (str): Root CL ID (e.g., 'CL:0000988'). Defaults to blood cells.

    Returns:
        Path: Path to data/hierarchy/processed/{CL_ID}_{MM-DD}/ folder

    Example:
        >>> get_data_folder('2025-10-24', 'CL:0000988')
        Path('/path/to/project/data/hierarchy/processed/CL0000988_10-24')
    """
    cl_folder = root_cl_id.replace(":", "")
    date_folder = date[5:]  # Gets 'MM-DD' from 'YYYY-MM-DD'
    return HIERARCHY_DATA_DIR / f"{cl_folder}_{date_folder}"

import json
from importlib import resources
import os

from fusionrate.load_data import DEFAULT_DATA_DIR

__all__ = ["ion_mass"]

file_path = resources.files(DEFAULT_DATA_DIR) / "ions.json"

with open(file_path, 'r') as f:
    ion_data = json.load(f)

def ion_mass(s):
    r"""
    Parameters
    ----------
    s: string
       One of the canonical ion names

    Returns
    -------
    Mass in amu
    """
    return ion_data[s]["mass"]


if __name__ == "__main__":
    print(ion_data)

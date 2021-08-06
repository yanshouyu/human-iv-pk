"""Chemical utils module.
"""
import numpy as np
from pandas.core.algorithms import isin
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from typing import List, Tuple, Union

def is_valid_smiles(sm):
    try:
        Chem.SanitizeMol(Chem.MolFromSmiles(sm))
    except:
        return False
    else:
        return True

def get_fp_mat(
    mols: Union[List[Mol], List[str]], 
    fpType: str = 'morgan', 
    fpSize: int = 256) -> np.ndarray:
    """Generate fingerprint at given fpSize. 
    Args:
        mols: Element of list can be either Mol object or SMILES string.
        fpType: fingerprint type, {morgan, rdkit}
        fpSize: bit vector length.
    Returns:
        2d array, shape (len(mols), fpSize), dtype 'float64'
    """
    assert len(mols) > 0
    if isinstance(mols[0], str):
        mols = [Chem.MolFromSmiles(sm) for sm in mols]
    elif not isinstance(mols[0], Mol):
        raise ValueError(f"element type incompatible, got {type(mols[0])}")
    arr = np.zeros((len(mols), fpSize))
    if fpType == "morgan":
        fp_func = lambda mol: GetMorganFingerprintAsBitVect(mol, 2, nBits=fpSize)
    elif fpType == 'rdkit':
        fp_func = lambda mol: Chem.RDKFingerprint(mol, fpSize=fpSize)
    else:
        raise NotImplementedError(f"Fingerprint {fpType} not implemented")
    
    for i, mol in enumerate(mols):
        fp = fp_func(mol)
        ConvertToNumpyArray(fp, arr[i])
    return arr

def morgan_bit_atoms(mol: Mol, nBits: int, bit: int) -> Tuple[Tuple[int, int]]:
    """Return a tuple of (atom_id, radius) for the bit in morgan fingerprint. 
    If the bit is not on, return None.
    """
    bi = {}
    _ = GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits, bitInfo=bi)
    if bit in bi:
        return bi[bit]
    else:
        return None
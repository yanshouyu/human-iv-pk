"""Visualization module, including both data and molecule visualization.
"""
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from typing import Union
from .utils import morgan_bit_atoms

def draw_highlighted_bit(
    fname: str, 
    mol: Union[Mol, str], 
    bit: int, 
    nBits: int) -> None:
    """Draw the molecule, highlighting a bit on its Morgan fingerprint if 
    the bit is on.
    Args:
        fname: path of the image to be saved
        mol: molecule object or smiles string
        bit: the bit to highlight
        nBits: nBits in generating morgan fingerprint
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    hl_atoms = []
    hl_bonds = []
    atom_radius = morgan_bit_atoms(mol, nBits=nBits, bit=bit)
    if atom_radius:
        for a_id, r in morgan_bit_atoms(mol, nBits=nBits, bit=bit):
            hl_atoms.append(a_id)
            if r > 0:
                atom = mol.GetAtomWithIdx(a_id)
                for bond in atom.GetBonds():
                    hl_bonds.append(bond.GetIdx())
    d = rdMolDraw2D.MolDraw2DCairo(500, 500)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=hl_atoms, highlightBonds=hl_bonds)
    d.FinishDrawing()
    d.WriteDrawingText(fname)
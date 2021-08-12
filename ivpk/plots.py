"""Visualization module, including both data and molecule visualization.
"""
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
from typing import Union, Tuple
from matplotlib import pyplot as plt
from .utils import morgan_bit_atoms
from .evalutaion import Evaluation

#------------ chem plot -------------#

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


#------------ stat plot -------------#

def plot_model_eval(ev: Evaluation):
    def plot_scatter(x, y, lim: Tuple):
        plt.plot(x, y, ".")
        plt.xlim(*lim)
        plt.ylim(*lim)
        plt.xlabel("real")
        plt.ylabel("predicted")
    
    # set lim
    min_val = np.min([ev.y_val, ev.y_val_pred])
    max_val = np.max([ev.y_val, ev.y_val_pred])
    interval = max_val - min_val
    lim = (
        np.floor(min_val - 0.05*interval), 
        np.ceil(max_val + 0.05*interval)
    )

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plot_scatter(ev.y_val, ev.y_val_pred, lim)
    plt.title((
        "validation", 
        f"MAE: {ev.evaluation['MAE_val']:.3f},", 
        f"Pearson: {ev.evaluation['Pearsonr_val']:.3f}"
    ))
    plt.subplot(122)
    plot_scatter(ev.y_train_val, ev.y_train_val_pred, lim)
    plt.title((
        f"CV pred", 
        f"MAE: {ev.evaluation['MAE_cv']:.3f}", 
        f"Pearson: {ev.evaluation['Pearsonr_cv']:.3f}"
    ))
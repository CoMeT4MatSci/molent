


# the XYZ files from QM9 aren't really XYZ... clean them up:
def cleanup_qm9_xyz(fname):
    """ Reads a xyz file from the QM9 dataset. Taken from  
        https://greglandrum.github.io/rdkit-blog/posts/2022-12-18-introducing-rdDetermineBonds.html

        Args:
            fname(string): file name of xyz file.

        Returns:
            string: Clean version of the xyz data.
            string: SMILES string extracted from header info (GDB).
            string: SMILES string extracted from header info (OpenBabel).

    """    
    ind = open(fname).readlines()
    nAts = int(ind[0])
    # There are two smiles in the data: the one from GDB and the one assigned from the
    # 3D coordinates in the QM9 paper using OpenBabel (I think).
    gdb_smi,relax_smi = ind[-2].split()[:2]
    ind[1] = '\n'
    ind = ind[:nAts+2]
    for i in range(2,nAts+2):
        l = ind[i].replace('*^', 'e')
        l = l.split('\t')
        l.pop(-1)
        ind[i] = '\t'.join(l)+'\n'
    ind = ''.join(ind)
    
    return ind, gdb_smi, relax_smi


def atoms2mol(atoms, charge=0):
    """ Converts ASE Atoms representation to rdkit molecule.
        Taken from  
        https://greglandrum.github.io/rdkit-blog/posts/2022-12-18-introducing-rdDetermineBonds.html

        Args:
            atoms(ase.Atoms): Input molecule.
            charge(int): Total charge of the molecule.
        
        Returns:
            rdchem.Mol: rdkit molecule or None if an error occurs.
    """
    import io
    from ase.io import write
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    
    ## xyz_to_mol
    xyz_string = io.StringIO()
    write(xyz_string, atoms, format='xyz')
    raw_mol = Chem.MolFromXYZBlock(xyz_string.getvalue())
    xyz_mol = Chem.Mol(raw_mol)
    try:
        rdDetermineBonds.DetermineBonds(xyz_mol, charge=charge)
    except ValueError:
        xyz_mol = None
        
    return xyz_mol


def plot_similarities(S, symbols=None, digits=1, fig=None, ax=None):
    """ Plots similarity matrix as a heatmap.
    
        Args:
            S(np.ndarray): Similarity matrix.
            symbols(list): List of chemical symbols.
            digits(int): Number of digits to be used for displaying the similarity values.
            fig (optional): fig object for subplots. Defaults to None. If None, new one is inilialized.
            ax (optional): ax object for subplots. Defaults to None. If None, new one is initialized.
            
        Returns:
            fig, ax
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if not fig:
        fig = plt.figure(figsize=(4,4))
    if not ax:
        ax = fig.add_subplot(111, aspect='equal')
    
    ax.imshow(S, cmap='Blues', aspect='equal')

    N = S.shape[0]
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            c = 'w'
            if S[i,j] == 0:
                c = 'k'
            text = ax.text(j, i, np.round(S[i,j],digits),
                           ha="center", va="center", color=c)

    if symbols is None:
        symbols = [str(i+1) for i in range(N)]

    ax.set_xticks(np.arange(N), labels=symbols)
    ax.set_yticks(np.arange(N), labels=symbols)
    ax.tick_params(top=True, bottom=False,
                       labeltop=True, labelbottom=False)

    ax.set_xticks(np.arange(N+1)-.5, minor=True)
    ax.set_yticks(np.arange(N+1)-.5, minor=True)
    ax.grid(which='minor', color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    
    return fig, ax

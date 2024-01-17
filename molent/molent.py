import numpy as np
from scipy.linalg import logm

def atomic_smiles(m, max_radius = 1):
    """ Determine fragments of molecule m by starting from each atom and taking atoms 
        which are at most max_radius bonds away.
    
        Args:
            m(rdchem.Mol): Molecule.
            max_radius(int): Number of bonds to include in fragment.
            
        Returns:
            list: List of SMILES strings for each atomic environment in molecule.
    """
    from rdkit import Chem
    
    smiles = []
    for at in range(m.GetNumAtoms()):
        if max_radius == 0:
            sm = m.GetAtomWithIdx(at).GetSymbol()
        else:
            for rad in range(max_radius,0,-1):
                env = Chem.FindAtomEnvironmentOfRadiusN(m, rad, at, useHs=True)
                amap={}
                submol = Chem.PathToSubmol(m, env, atomMap=amap)

                if len(amap)>0:
                    break

            if submol.GetNumAtoms() == 0:
                submol = m
                amap = {0: 0}

            sm = Chem.MolToSmiles(submol, rootedAtAtom=amap[at], canonical=True)
        smiles.append(sm)

    return smiles


def fragment_smiles(mol_smiles, N_rad=1, useHs=False):
    """ Determine fragments of each molecule given in the list of SMILES strings.
    
        Args:
            mol_smiles(list): List of SMILES strings representing molecules.
            N_rad(int): Number of bonds to include in fragment.
            useHs(boolean): Use explicit hydrogens to determine fragments.
            
        Returns:
            list of lists: List of SMILES strings for each atomic environment in each molecule.
    """
    from rdkit import Chem
    
    fragsmiles = []
    for smi in mol_smiles:
        mol = Chem.MolFromSmiles(smi)

        if useHs:
            mol = Chem.AddHs(mol)

        fragsmiles.append(atomic_smiles(mol, max_radius=N_rad))
        
    return fragsmiles


def binary_similarity(smiles):
    """ Calculates the similarity matrix by pairwise comparing a list of strings (or other objects).
    
        Args:
            smiles(list): List of objects that can be compared using `==`.
            
        Results:
            np.ndarray: Similarity matrix.
    """
    S = np.zeros((len(smiles), len(smiles)))
    
    for i in range(len(smiles)):
        for j in range(len(smiles)):
            S[i,j] = int(smiles[i] == smiles[j])

    return S


def cosine_similarity(desc, at_nums=None):
    """ Calculates the similarity matrix by pairwise taking the scalar product of the given descriptors.
    
        Args:
            desc(list): List of descriptor arrays.
            at_nums(list): List of atom types which are used in addition to the scalar product to determine the similarity.
            
        Results:
            np.ndarray: Similarity matrix.
    """    
    S = np.zeros((desc.shape[0], desc.shape[0]))
    
    if at_nums is None:
        at_nums = np.zeros(desc.shape[0], dtype=int)
    
    for i in range(S.shape[0]):
        for j in range(i, S.shape[0]):
            if at_nums[i] == at_nums[j]:
                S[i,j] = np.dot(desc[i], desc[j])/np.sqrt(np.dot(desc[i], desc[i])*np.dot(desc[j], desc[j]))
            S[j,i] = S[i,j]
            
    return S


def entropy(sim):
    """ Calculates the (Shanon) information entropy from the given similarity matrix.
    
        Args:
            sim(np.ndarray): Similarity matrix.
            
        Returns:
            float: Information entropy.
    """
    A = sim / sim.shape[0]
    
    es = np.linalg.eigvals(A)
    return -np.dot(np.log(es[es > 0.]), es[es > 0.])/np.log(2)
#    return -np.trace(A @ logm(A))/np.log(2)


def cross_entropy(sim_p, sim_q):
    """ Calculates the cross entropy from the given similarity matrices.
    
        Args:
            sim_p(np.ndarray): Reference similarity matrix.
            sim_q(np.ndarray): Second similarity matrix.
            
        Returns:
            float: Cross entropy.
    """
    A = sim_p / sim_p.shape[0]
    B = sim_q / sim_q.shape[0]
        
    return -np.trace(A @ logm(B))/np.log(2)
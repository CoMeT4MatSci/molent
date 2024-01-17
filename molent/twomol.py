import numpy as np

from .molent import entropy

def mixing_entropy(n1, n2):
    """ Calculates the mixing entropy of two molecules with n1 and n2 atoms, respectively. 
        The mixing entropy assumes that the two molecules have no similar environments.
    
        Args:
            n1(float): Number of atoms in 1st molecule.
            n2(float): Number of atoms in 2nd molecule.
        
        Returns:
            float: Mixing entropy.
    """
    p1 = n1/(n1+n2)
    p2 = n2/(n1+n2)
    
    return -p1*np.log2(p1) -p2*np.log2(p2)


def mixing_gain(sim12, n12):
    """ Calculates the gain of entropy due to mixing of two molecules.
    
        Args:
            sim12(np.ndarray): Similarity matrix of the combination of two molecules.
            n12(list or tuple): Number of atoms in 1st and 2nd molecule, i.e. [n1,n2].
    
        Returns:
            float: Gain of entropy.
    """
    n1, n2 = n12

    sim1 = sim12[0:n1, 0:n1]
    H1 = np.real(entropy(sim1))

    sim2 = sim12[n1:, n1:]
    H2 = np.real(entropy(sim2))
    
    return np.real(entropy(sim12)) - (n1*H1 + n2*H2)/(n1+n2)


def average_kernel(sim12, n12, p=1):
    """ Calculates the average of mutual similarites of environments of two molecules.
        See S. De et al, PCCP 18(20):13754–13769, 2016. arXiv:1601.04077
    
        Args:
            sim12(np.ndarray): Similarity matrix of the combination of two molecules.
            n12(list or tuple): Number of atoms in 1st and 2nd molecule, i.e. [n1,n2].
            p(int): Power of matrix elements. p=1 means matrix elements are averaged,
            p=2 the square of elements are averaged, etc.
    
        Returns:
            float: Average kernel.
    """
    n1, n2 = n12
    sim12_sub = sim12[0:n1, n1:]
    
    return np.mean(sim12_sub**p)


def bestmatch_kernel(sim12, n12, p=1):
    """ Calculates the best-match kernel from mutual similarites of environments of two molecules.
        See S. De et al, PCCP 18(20):13754–13769, 2016. arXiv:1601.04077
    
        Args:
            sim12(np.ndarray): Similarity matrix of the combination of two molecules.
            n12(list or tuple): Number of atoms in 1st and 2nd molecule, i.e. [n1,n2].
            p(int): Power of matrix elements. p=1 means matrix elements are used,
            p=2 the square of elements are used, etc.
    
        Returns:
            float: Best-match kernel.
    """    
    from scipy.optimize import linear_sum_assignment
    
    n1, n2 = n12
    sim12_sub = sim12[0:n1, n1:]

    row_ind, col_ind = linear_sum_assignment(sim12_sub, maximize=True)
    
    return (sim12_sub[row_ind, col_ind]**p).sum() / row_ind.shape[0]
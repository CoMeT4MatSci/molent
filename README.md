# molent
Computing molecular information entropies and similarities.

## requirements
- numpy and scipy
- some functions use [rdkit](https://www.rdkit.org/)
- the SOAP descriptors are calculated using [DScribe](https://github.com/SINGROUP/dscribe)

## example(s)
A simple example looks like this:
```python
from rdkit import Chem
from molent.molent import entropy, binary_similarity, atomic_smiles

# construct molecule representation
mol = Chem.MolFromSmiles("CCO")
mol = Chem.AddHs(mol)

# get environments with 1-bond "radius"
frag_smiles = atomic_smiles(mol, max_radius=1)

# get similarity matrix
sim = binary_similarity(frag_smiles)

# compute entropy
entropy(sim)
```
More examples can be found in the [examples/](examples/) folder.

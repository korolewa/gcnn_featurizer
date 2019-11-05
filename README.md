# gcnn_featurizer

featurizer.structure_to_convmol() is designed to generate binary adjacency and vertex feature matrices for input pymatgen.Structure:

```python
from featurizer import structure_to_convmol

adjacency_matrix, vertex_matrix, num_atoms = structure_to_convmol(structure)
```

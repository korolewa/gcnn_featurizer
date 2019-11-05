import json
import numpy as np
from pymatgen.core.periodic_table import Element
from matminer.featurizers.site import VoronoiFingerprint, CoordinationNumber

ELEMENTAL_PROPERTIES = list(json.load(open('properties_from_tables_full.json')).values())


def zfill(matrix, dim_v, dim_h):
    matrix = np.hstack([matrix, np.zeros((matrix.shape[0], dim_h - matrix.shape[1]))])
    matrix = np.vstack([matrix, np.zeros((dim_v - matrix.shape[0], dim_h))])
    return matrix


def structure_to_convmol(structure, properties=ELEMENTAL_PROPERTIES, max_atoms=200, max_features=41, tolerance_distance=0.25):
    atomic_radii = {
        'At': 1.50,
        'Bk': 1.70,
        'Cm': 1.74,
        'Fr': 2.60,
        'He': 0.28,
        'Kr': 1.16,
        'Lr': 1.71,
        'Md': 1.94,
        'Ne': 0.58,
        'No': 1.97,
        'Rn': 1.50,
        'Xe': 1.40,       
    }

    distance_matrix = structure.distance_matrix

    for index, x in np.ndenumerate(distance_matrix):
        radius_1 = Element(structure._sites[index[0]].specie).atomic_radius or atomic_radii[str(structure._sites[index[0]].specie)]
        radius_2 = Element(structure._sites[index[1]].specie).atomic_radius or atomic_radii[str(structure._sites[index[1]].specie)]
        max_distance = radius_1 + radius_2 + tolerance_distance
        if x > max_distance:
            distance_matrix[index] = 0
        else:
            distance_matrix[index] = 1
    np.fill_diagonal(distance_matrix, 1)
    atom_features = []
    
    for i, site in enumerate(structure._sites):
        atom_feature_vector = []
        for atom_property in properties:
            min_value = np.nanmin(np.array(list(atom_property.values()), dtype=float))
            max_value = np.nanmax(np.array(list(atom_property.values()), dtype=float))
            if atom_property[str(Element(site.specie))] is not None:
                atom_feature_vector.append((atom_property[str(Element(site.specie))] - min_value) / (max_value - min_value))
            else:
                atom_feature_vector.append(None)
        
        voronoi_min = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        voronoi_max = np.array([120.0, 135.0, 11.0, 3.0, 11.0, 12.0, 18.0, 7.0, 17.0, 17.0, 6.0, 2.0, 6.0, 7.0])
        voronoi_fps = VoronoiFingerprint().featurize(structure, i)
        i_fold_symmetry_indices = voronoi_fps[8:16]
        voronoi_stats = (np.array(voronoi_fps[16:]) - voronoi_min) / (voronoi_max - voronoi_min)
        atom_feature_vector.extend(i_fold_symmetry_indices + voronoi_stats.tolist())        

        coord_min = np.array([1])
        coord_max = np.array([36])
        coord_fps = ((CoordinationNumber.from_preset(
                "MinimumDistanceNN").featurize(structure, i) - coord_min) / (coord_max - coord_min)).tolist()
        atom_feature_vector.extend(coord_fps)        
        
        atom_features.append(atom_feature_vector)
        
    atom_features = np.array(atom_features, dtype=np.float)
    
    if np.isnan(atom_features).any():
        raise ValueError('feature vector contains nan value')    
    
    return (zfill(distance_matrix, max_atoms, max_atoms), zfill(atom_features, max_atoms, max_features), len(structure.sites))

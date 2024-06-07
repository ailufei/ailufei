from Bio.PDB import PDBParser
from scipy.spatial import Delaunay
import numpy as np
import json
from collections import OrderedDict

def calculate_surface_curvature(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]

    coords = []
    for atom in model.get_atoms():
        coords.append(atom.get_coord())

    coords = np.array(coords)
    tri = Delaunay(coords[:, :3])
    tetra = tri.simplices

    curvature_sum = 0.0
    count = 0

    for tet in tetra:
        x1, y1, z1 = coords[tet[0]][:3]
        x2, y2, z2 = coords[tet[1]][:3]
        x3, y3, z3 = coords[tet[2]][:3]
        x4, y4, z4 = coords[tet[3]][:3]

        a = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        b = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2 + (z1 - z3) ** 2)
        c = np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2 + (z1 - z4) ** 2)
        d = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2 + (z2 - z3) ** 2)
        e = np.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2 + (z2 - z4) ** 2)
        f = np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2 + (z3 - z4) ** 2)

        V = (a * d * f) + (b * e * f) + (c * d * e) - (a * e * e) - (b * d * d) - (c * f * f)
        if V == 0:
            continue

        R = (a * a * (e * e + f * f - d * d)) + (b * b * (d * d + f * f - e * e)) + (c * c * (d * d + e * e - f * f))
        R /= 16 * V
        curvature_sum += 1 / R
        count += 1

    if count == 0:
        return None

    mean_curvature = curvature_sum / count
    return mean_curvature


output_file='/home/ntu/Documents/MY_Module/protein_information/curve.txt'
dict_file_curv = '/home/ntu/Documents/MY_Module/protein_information/curve _dict.txt'
path ='/home/ntu/Documents/MY_Module/fusion/data_kiba/'
proteins = json.load(open(path + "proteins.txt"), object_pairs_hook=OrderedDict)
XT = []
XT_key = []

def CURVE(pro):
    pdb_file = '/home/ntu/Documents/MY_Module/protein_information/PDB/Davis_PDB (copy)/' + str(pro) + '.pdb'

    mean_curvature = calculate_surface_curvature(pdb_file)


    # if mean_curvature is not None:
    #print(f"{t}: {mean_curvature:.2f} Å^-1")
    #print(f"{t}:{mean_curvature:.2f}")
    # else:
    #     print("Failed to calculate surface curvature.")

    return '%.4f'%mean_curvature

# for t in proteins.keys():
#     print("protein:"+t+"   mean_curvature:"+CURVE(t))


       # with open(output_file, 'w') as f:
    #     for t in proteins.keys():
    #         f.write(t+'\t'+str(CURVE('KIBA_PDB',t))+'\n')


    # with open(output_file, 'r') as f:
    #     with open(dict_file_curv, 'w') as file:
    #         for lines in f:
    #             parts=lines.split()
    #             key=parts[0]
    #             value=float(parts[1])
    #             dictionary={key:value}
    #             #file.write(str(dictionary))
    #
    #
    # print(dictionary.keys["O00141"])

import pandas as pd

dict_compound_infomation={}

def count_atoms(chformula):
    atoms = ['C', 'H', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']
    atoms_count = {atom: 0 for atom in atoms}
    for atom in atoms_count.keys():
        count = chformula.count(atom)
        atoms_count[atom] = count
    return list(atoms_count.values())


def all_drug_chemical_feature(P_cid):
    data = pd.read_csv('/home/ntu/Documents/MY_Module/p_c_information/Data/Davis/drug_info.csv')
    print(data)
    pubchem_cid = data['cid'].tolist()
    print(pubchem_cid)
    atoms = ['C', 'H', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']
    features = []
    not_found = []

    for cid in P_cid:
        cid = str(cid)
        if cid not in pubchem_cid:
            not_found.append(cid)
            continue

        row = data.loc[data['cid'] == cid]
        feature = [
            row['mw'].values[0],
            row['polararea'].values[0],
            row['complexity'].values[0],
            row['heavycnt'].values[0],
            row['hbonddonor'].values[0],
            row['hbondacc'].values[0],
            row['rotbonds'].values[0]
        ]
        chformula = row['mf'].values[0]
        print(chformula)
        feature += count_atoms(chformula)
        features.append(feature)

    return features, not_found


# Example usage
P_cid = ['10184653']  # Example PubChem CIDs
features, not_found = all_drug_chemical_feature(P_cid)

print('Features:')
for feature in features:
    print(feature)

print('Not found:', not_found)

b=count_atoms('CC1=CC(=C(C=C1)F)NC(=O)NC2=CC=C(C=C2)C3=C4C(=CC=C3)NN=C4N')
print(b)
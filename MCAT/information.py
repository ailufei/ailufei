import pubchempy as pcp
import csv
import torch
from pubchempy import get_compounds
import pandas as pd
import math
import json
import pickle
import numpy as np
from collections import OrderedDict

path='/home/ntu/Documents/MY_Module/p_c_information/'
ligands = json.load(open(path + "ligands_can4.1.txt"), object_pairs_hook=OrderedDict)

XD = []
XT = []
for d in ligands.keys():
    XD.append(ligands[d])
print(XD)


#file_path = r'/home/ntu/Documents/MY_Module/DATA/DATAdrug1.csv'  # r对路径进行转义，windows需要
#raw_data = pd.read_csv(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
#raw_data = pd.read_csv(file_path, header=None)  # 不设置表头
#print(raw_data)



#csv_path = "/home/ntu/Documents/MY_Module/p_c_information/info.csv"

def write_csv(csv_path, M):
    with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(M)

def ite_write_csv(csv_path, row):
    '''
    追加写入，每次只写入一行
    '''
    with open(csv_path, 'a+', newline='', encoding='utf-8') as csvfile:
        csv_write = csv.writer(csvfile)
        csv_write.writerow(row)

def count_atoms(chformula):
    atoms = ['C', 'H', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']
    atoms_count = {atom: 0 for atom in atoms}
    for atom in atoms_count.keys():
        count = chformula.count(atom)
        atoms_count[atom] = count
    return list(atoms_count.values())

save_lsit=[]
'''a_lane1_list = []  # 车道1

#for i in range(0, 115184):
for i in range(0, 20):
    a_lane1_list.append(raw_data.values[i, 6])  # 读取excel第一列的值，并添加到列表
print(a_lane1_list)
#print("a_lane1_list = " + str(a_lane1_list))'''




dict_compound_information={}


def get_infomatrion(smile_list):
    #for i in range(0, 115184):
    #for i in range(0, 20):
    features=[]
    for smile in smile_list:
        #for compound in get_compounds(input[i], 'smiles'):
        for compound in get_compounds(smile, 'smiles'):
            print(smile)
            if smile=="CCOC1=CC(=C(C=C1)C=C2C(=O)N=C(S2)N)O":
                b1='4589745'
            elif smile=="C1CN(CC1O)CC2=NC(=O)C3=C(N2)C4=C(S3)C=CC(=C4)C5=CC=C(C=C5)O":
                b1='135927974'
            elif smile=='C1C2C(C3C(C(O2)O)OC(=O)C4=CC(=C(C(=C4C5=C(C(=C(C=C5C(=O)O3)O)O)O)O)O)O)OC(=O)C6=CC(=C(C(=C6C7=C(C(=C8C9=C7C(=O)OC2=C(C(=C(C3=C(C(=C(C=C3C1=O)O)O)O)C(=C92)C(=O)O8)O)O)O)O)O)O)O':
                b1='44460933'
            elif smile =='C1C2C(C3C4C(C5=C(C(=C(C(=C5C(=O)O4)C6=C(C(=C(C(=C6C(=O)O3)C7=C(C(=C(C=C7C(=O)O2)O)O)O)O)O)O)O)O)O)O)OC(=O)C8=CC(=C(C(=C8C9=C(C(=C2C3=C9C(=O)OC4=C(C(=C(C5=C(C(=C(C=C5C(=O)O1)O)O)O)C(=C34)C(=O)O2)O)O)O)O)O)O)O':
                b1='44460932'
            else:
                b1 = compound.cid
            #print("compound:",compound)
            #print("b1:",b1)
            c1 = compound.isomeric_smiles
            automs=count_atoms(c1)
            #print(automs)


        # 获取物CID

        c = pcp.get_compounds(b1, 'cid')[0]
        cid = c.cid

        # 获取物理化学信息
        props = ['xlogp', 'tpsa', 'complexity', 'h_bond_donor_count',
                 'heavy_atom_count', 'charge',
                 'h_bond_acceptor_count']  # molecular_weight,xlogp:亲脂性    TPSA:拓扑分子极性表面积 是常用于药物化学的一个参数，其定义为化合物内极性分子的总表面积 complexity:复杂性

        #props = ['isomeric_smiles', 'xlogp', 'tpsa','complexity','molecular_weight','h_bond_donor_count','heavy_atom_count','charge','h_bond_acceptor_count']     #xlogp:亲脂性    TPSA:拓扑分子极性表面积 是常用于药物化学的一个参数，其定义为化合物内极性分子的总表面积 complexity:复杂性
        #MolecularWeight,charge,HBondDonorCount，HeavyAtomCount，IsotopeAtomCount，
        compound = pcp.Compound.from_cid(cid)
        #print(dir(compound))
        info = [compound.__getattribute__(prop)  for prop in props ]
        #for i,a in enumerate(info):
        #  if a==None
        # 将信息转化为矩阵形式
        middle_c=info+automs
        #print("middle_c:",middle_c)
        #dict_compound_information[smile]=middle_c
        #print("dict_compound_information:",dict_compound_information)

        #features.append(dict_compound_information[smile])
        #print('middle_c:',middle_c)
        features.append(middle_c)
        df_features = pd.DataFrame(features)
        df_features = df_features.fillna(0)
    #print('df_features:',df_features)

    np_features = df_features.to_numpy()
    tensor_feature=torch.tensor(np_features)
    float_feature = torch.tensor(tensor_feature, dtype=torch.float32)
    #print(a.size())
    return float_feature


#get_infomatrion(XD)
#print(dict_compound_information)

# path_in='/home/ntu/Documents/feiailu/MY_Module/p_c_information/compound_information4.1txt'
# with open(path_in,"w") as f:
#     f.write(json.dumps(dict_compound_information))






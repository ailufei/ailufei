
import pubchempy as pcp
import csv
import torch
from pubchempy import get_compounds
import pandas as pd
import json


path_infomation_KIBA='/home/ntu/Documents/MY_Module_Finallshell/p_c_information/KIBA/'
com_info_KIBA=json.load(open(path_infomation_KIBA+"all_compound_information1.txt"))

path_infomation_DAVIS='/home/ntu/Documents/MY_Module_Finallshell/p_c_information/Davis/'
com_info_DAVIS=json.load(open(path_infomation_DAVIS+"all_compound_information.txt"))


#file_path = r'/home/ntu/Documents/MY_Module/DATA/DATAdrug1.csv'  # r对路径进行转义，windows需要
#raw_data = pd.read_csv(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
#raw_data = pd.read_csv(file_path, header=None)  # 不设置表头
#print(raw_data)



#csv_path = "/home/ntu/Documents/feiailu/MY_Module/p_c_information/info.csv"

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

'''
def get_infomatrion(smile_list):
    #for i in range(0, 115184):
    #for i in range(0, 20):
    features=[]
    for smile in smile_list:
        #for compound in get_compounds(input[i], 'smiles'):
        for compound in get_compounds(smile, 'smiles'):
            b1 = compound.cid
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
        features.append(middle_c)
        #print('middle_c:',middle_c)
    df_features = pd.DataFrame(features)
    df_features = df_features.fillna(0)
    #print('df_features:',df_features)

    np_features = df_features.to_numpy()
    tensor_feature=torch.tensor(np_features)
    float_feature = torch.tensor(tensor_feature, dtype=torch.float32)
    #print(a.size())
    return float_feature'''


def get_infomatrion_KIBA(smiles_list):
    features = []
    for smiles in smiles_list:
        #print(protein_list)
        features.append(com_info_KIBA[smiles])
            #print('middle_c:',middle_c)
        df_features = pd.DataFrame(features)
        df_features = df_features.fillna(0)
        #print('df_features:',df_features)

        np_features = df_features.to_numpy()
        #tensor_feature=torch.tensor(np_features)
        float_feature = torch.as_tensor(np_features, dtype=torch.float32)
        #print(float_feature.size())
    return   float_feature

def get_infomatrion_DAVIS(smiles_list):
    features = []
    for smiles in smiles_list:
        #print(protein_list)
        features.append(com_info_DAVIS[smiles])
            #print('middle_c:',middle_c)
        df_features = pd.DataFrame(features)
        df_features = df_features.fillna(0)
        #print('df_features:',df_features)

        np_features = df_features.to_numpy()
        #tensor_feature=torch.tensor(np_features)
        float_feature = torch.as_tensor(np_features, dtype=torch.float32)
        #print(float_feature.size())
    return   float_feature











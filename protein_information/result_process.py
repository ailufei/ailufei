# author:emo
# creatTime:8/17/23,9:15 AM
# filename:result_process


'''
#cpmpute all protein infomation

path_in='/home/ntu/Documents/MY_Module/protein_information/Davis_result_process.txt'
dict_2={}

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from protein_curvature import CURVE
from P_dainhe import total_energy
import json


import json
from collections import OrderedDict
path='/home/ntu/Documents/MY_Module/fusion/data_davis/'
proteins = json.load(open(path + "proteins.txt"), object_pairs_hook=OrderedDict)
XT = []
XT_key=[]
for t in proteins.keys():
    XT.append(proteins[t])




def count_atoms(chformula):
    atoms = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K','L','M','N','P','Q','R','S','T','V','W','Y']
    atoms_count = {atom: 0 for atom in atoms}
    for atom in atoms_count.keys():
        count = chformula.count(atom)
        atoms_count[atom] = count
    return list(atoms_count.values())


def get_protein_information(proteins):
    # 定义蛋白质序列
    i=0

    for t in proteins.keys():
        print(t)
        protein = ProteinAnalysis(proteins[t])         # 创建蛋白质分析对象
        hydrophobicity = protein.gravy()                    # 计算蛋白质氨基酸的疏水性指标
        molecular_weight = protein.molecular_weight()        # 计算蛋白质的分子量
        isoelectric_point = protein.isoelectric_point()   # 计算蛋白质的等电点
        pro_info_list = [molecular_weight,isoelectric_point,hydrophobicity,float(CURVE(t)),float(total_energy(t))]
        pro_reduse=count_atoms(proteins[t])#[分子量,等电点,疏水性指标]
        pro_info_list=pro_info_list+pro_reduse
        dict_2[proteins[t]] = pro_info_list
        # 打印结果
        #print("蛋白质分子量：", molecular_weight)
        #print("蛋白质等电点：", isoelectric_point)
    return pro_info_list


#cun chu de dao de shu ju

get_protein_information(a)
print(dict_2)


with open(path_in,"w") as f:
    f.write(json.dumps(dict_2))
'''


#use all dataset

import json
from collections import OrderedDict
path='/home/ntu/Documents/MY_Module/fusion/data_davis/'
proteins = json.load(open(path + "proteins.txt"), object_pairs_hook=OrderedDict)





import torch
import pandas as pd
import json
from collections import OrderedDict

path_infomation='/home/ntu/Documents/MY_Module/protein_information/'

pro_info=json.load(open(path_infomation+"Davis_result_process.txt"), object_pairs_hook=OrderedDict)
features=[]


def protein_infomation(protein_list):
    for protein in protein_list:
        print(protein)
        features.append(pro_info[protein])
            #print('middle_c:',middle_c)
        df_features = pd.DataFrame(features)
        df_features = df_features.fillna(0)
        #print('df_features:',df_features)

        np_features = df_features.to_numpy()
        tensor_feature=torch.tensor(np_features)
        float_feature = torch.tensor(tensor_feature, dtype=torch.float32)
        #print(a.size())
    return   float_feature


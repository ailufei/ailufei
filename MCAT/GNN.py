import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from rdkit import Chem
from rdkit.Chem import rdDepictor
import pandas as pd
import numpy as np







def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


# one ont encoding with unknown symbol
def one_hot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_hot_encoding_unk(atom.GetSymbol(),
                                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                          'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                          'Pt', 'Hg', 'Pb', 'X']) +
                    one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv


def ch(smiles_list):
    data_list = []
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        try:
            mol_size = mol.GetNumAtoms()
        except:
            continue
        mol_features = []
        for atom in mol.GetAtoms():
            feature = atom_features(atom)
            mol_features.append(feature / sum(feature))
        edges = []
        bond_type_np = np.zeros((mol_size, mol_size))
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            bond_type_np[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond.GetBondTypeAsDouble()
            bond_type_np[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond.GetBondTypeAsDouble()
            # bond_type.append(bond.GetBondTypeAsDouble())
        g = nx.Graph(edges).to_directed()


        mol_adj = np.zeros((mol_size, mol_size))
        for e1, e2 in g.edges:
            mol_adj[e1, e2] = 1
            # edge_index.append([e1, e2])
        # print(np.array(mol_adj).shape,'mol_adj')
        mol_adj += np.matrix(np.eye(mol_adj.shape[0]))

        bond_edge_index = []
        bond_type = []
        index_row, index_col = np.where(mol_adj >= 0.5)
        for i, j in zip(index_row, index_col):
            bond_edge_index.append([i, j])
            bond_type.append(bond_type_np[i, j])
        # print(bond_edge_index)
        # print('smile_to_graph')
        #print('mol_features',np.array(mol_features).shape)
        #print('bond_edge_index',np.array(bond_edge_index).shape)
        #print('bond_type',np.array(bond_type).shape)
        atom_features1 = torch.tensor(np.array(mol_features), dtype=torch.float).view(-1, 1)
        edge_index = torch.tensor(bond_edge_index, dtype=torch.long).t().contiguous()
        data = Data(x=atom_features1, edge_index=edge_index)
        data_list.append(data)
        #print(len(data_list))
        #print(data_list)
    return data_list


class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def GNN_feature(smiles_list,i):

  # 示例SMILES列表
    graph_data_list = ch(smiles_list)

    # 构造Batch对象
    batch = Batch.from_data_list(graph_data_list)

    input_dim = 1  # 输入特征维度
    hidden_dim = 64  # 隐层特征维度
    output_dim = 128  # 输出特征维度设置为128

    model = GCNModel(input_dim, hidden_dim, output_dim)

    # 扩展Batch对象以匹配32个样本
    expanded_batch = Batch.from_data_list([batch])
    expanded_batch.x=expanded_batch.x.unsqueeze(0).expand(i,-1,-1)
    #print(expanded_batch)
    output = model(expanded_batch).cuda()
    output=output.mean(dim=1)
    # 输出特征的维度：[32, 128]
    return output

def GNN_feature_8(smiles_list):
      # 示例SMILES列表
    graph_data_list = ch(smiles_list)

    # 构造Batch对象
    batch = Batch.from_data_list(graph_data_list)

    input_dim = 1  # 输入特征维度
    hidden_dim = 64  # 隐层特征维度
    output_dim = 128  # 输出特征维度设置为128

    model = GCNModel(input_dim, hidden_dim, output_dim)

    # 扩展Batch对象以匹配32个样本
    expanded_batch = Batch.from_data_list([batch])
    expanded_batch.x=expanded_batch.x.unsqueeze(0).expand(8,-1,-1)
    #print(expanded_batch)
    output = model(expanded_batch).cuda()
    output=output.mean(dim=1)
    # 输出特征的维度：[32, 128]
    return output

def GNN_feature_17(smiles_list):
  
  # 示例SMILES列表
    graph_data_list = ch(smiles_list)

    # 构造Batch对象
    batch = Batch.from_data_list(graph_data_list)

    input_dim = 1  # 输入特征维度
    hidden_dim = 64  # 隐层特征维度
    output_dim = 128  # 输出特征维度设置为128

    model = GCNModel(input_dim, hidden_dim, output_dim)

    # 扩展Batch对象以匹配32个样本
    expanded_batch = Batch.from_data_list([batch])
    expanded_batch.x=expanded_batch.x.unsqueeze(0).expand(17,-1,-1)
    #print(expanded_batch)
    output = model(expanded_batch).cuda()
    output=output.mean(dim=1)
    # 输出特征的维度：[32, 128]
    return output
'''

# SMILES表示的化合物

#file_path = r'/home/ntu/Documents/MY_Module/DATA/DATAdrug1.csv'  # r对路径进行转义，windows需要
file_path = r'/home/ntu/Documents/MY_Module/fusion/compound.csv'
raw_data = pd.read_csv(file_path, header=None)  # header=0表示第一行是表头，就自动去除了
# raw_data = pd.read_csv(file_path, header=None)  # 不设置表头


a_lane1_list = []  # 车道1

for i in range(0, 2110):
    a_lane1_list.append(raw_data.values[i,0])  # 读取excel第6列的值，并添加到列表

print(a_lane1_list[1])'''


'''a_lane1_list = []

from torch_geometric.nn import GCNConv, \
    global_mean_pool as gep
import torch.nn as nn'''




'''class DTA_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_mol=32, output_dim=128):
        super(DTA_GCN, self).__init__()

        print('DTA_GCN Loading ...')
        self.n_output = n_output
        self.mol_conv1 = GCNConv(num_features_mol, 128)
        #self.mol_conv2 = GCNConv(128, 64)
        self.relu = nn.ReLU()


        # combined layers
        self.fc1 = nn.Linear(output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self,data):
        x,edge_index,batch= data.x,data.edge_index,data.batch

        X=self.mol_conv1(x,edge_index)
        #X =self.mol_conv2(X)
        X=self.relu(X)

        #X =gep(X, )
        #out = self.out(X)
        return X

features_list = []
#a = len(sml)
#model = GCN(in_channels=1, out_channels=1)
def GNNSHI(sml):
    model1 = DTA_GCN()
    a = len(sml)
    for i in range(0,a):
        mol_features,bond_edge_index=ch(sml[i])
        #print(sml)
        # 构建分子图数据
        data= Data(x=mol_features, edge_index=bond_edge_index)
        # 提取特征
        result=model1(data).cuda()
        features_list.append(result.squeeze(0))
        print(features_list)
        #result=torch.cat(output1)
        print(result)
    features_tensor = torch.stack(features_list, dim=0)

    return features_tensor

print()'''


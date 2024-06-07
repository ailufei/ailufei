from rdkit import Chem
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader
import time
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


def GET_GRAPH_FEATURE(smiles_list):
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

        atom_features1 = torch.tensor(np.array(mol_features), dtype=torch.float).view(-1, 1)
        edge_index = torch.tensor(bond_edge_index, dtype=torch.long).t().contiguous()
        data = Data(x=atom_features1, edge_index=edge_index)
        data_list.append(data)
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

    graph_data_list = GET_GRAPH_FEATURE(smiles_list)

    # 构造Batch对象
    batch = Batch.from_data_list(graph_data_list)

    input_dim = 1  # 输入特征维度
    hidden_dim = 64  # 隐层特征维度
    output_dim = 128  # 输出特征维度设置为128

    model = GCNModel(input_dim, hidden_dim, output_dim).cuda()

    # 扩展Batch对象以匹配32个样本

    expanded_batch = Batch.from_data_list([batch])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    expanded_batch = expanded_batch.to(device)
    expanded_batch.x=expanded_batch.x.unsqueeze(0).expand(i,-1,-1)
    #print(expanded_batch)
    output = model(expanded_batch).cuda()
    output=output.mean(dim=1)
    # 输出特征的维度：[32, 128]
    return output


'''
t = time.time()
features = GNN_feature(smiles_list,32)
print(features.shape)

print(f'coast:{time.time() - t:.4f}s')
'''
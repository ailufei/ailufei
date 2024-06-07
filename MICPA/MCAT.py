import torch
import torch.nn as nn
from  GNN import GNN_feature
from information import get_infomatrion_KIBA
from information import get_infomatrion_DAVIS


num_heads=4
query_dim=key_dim=128
value_dim=14

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, num_heads, query_dim, key_dim, value_dim,random,dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()

        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.random=random


        self.query_transform = nn.Linear(query_dim, key_dim * num_heads)
        self.key_transform = nn.Linear(key_dim, key_dim * num_heads)
        self.value_transform = nn.Linear(8*value_dim, value_dim * num_heads)

        self.dropout = nn.Dropout(dropout)
        self.output_transform = nn.Linear(value_dim * num_heads,8* value_dim)
        self.line=nn.Linear(value_dim,8*value_dim)
    def forward(self, query, key, value,random):
        value=self.line(value)
        assert query.size(-2) == random and query.size(-1) == 128, "Input query size must be 32x128"
        assert key.size(-2) == random and key.size(-1) == 128, "Input key size must be 32x128"
        assert value.size(-2) == random and value.size(-1) == 128, "Input value size must be 32x128"

        batch_size = query.size(0)

        transformed_query = self.query_transform(query).view(batch_size, self.num_heads, -1, self.key_dim)
        transformed_key = self.key_transform(key).view(batch_size, self.num_heads, -1, self.key_dim)
        transformed_value = self.value_transform(value).view(batch_size, self.num_heads, -1, self.value_dim)

        # Compute attention scores
        scores = torch.matmul(transformed_query, transformed_key.transpose(-2, -1))

        # Normalize scores
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Compute weighted sum of values
        attended_values = torch.matmul(attention_weights, transformed_value)

        # Reshape and linear transformation
        attended_values = attended_values.view(batch_size, -1, self.value_dim * self.num_heads)
        output = self.output_transform(attended_values)
        output = output.mean(dim=1)
        return output, attention_weights

def COMPOUND_MCAT_KIBA(smiles_list,i,smiles):

    ##32 hou mian xu yao  yon bian liang i ti dai
    model=MultiHeadCrossAttention(num_heads,query_dim,key_dim,value_dim,i).cuda()
    output,attention=model(GNN_feature(smiles,i),GNN_feature(smiles,i).cuda(),get_infomatrion_KIBA(smiles_list).cuda(),i)
    #print(get_infomatrion(smiles_list),i)
    return output


def COMPOUND_MCAT_DAVIS(smiles_list,i,smiles):

    ##32 hou mian xu yao  yon bian liang i ti dai
    model=MultiHeadCrossAttention(num_heads,query_dim,key_dim,value_dim,i).cuda()
    output,attention=model(GNN_feature(smiles,i),GNN_feature(smiles,i).cuda(),get_infomatrion_DAVIS(smiles_list).cuda(),i)
    #print(get_infomatrion(smiles_list),i)
    return output

#print(COMPOUND_MCAT(smiles_list,32).shape)





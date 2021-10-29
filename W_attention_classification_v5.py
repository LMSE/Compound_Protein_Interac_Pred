import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import scipy
import torch
from torch import nn
from torch.utils import data
from torch.nn.utils.weight_norm import weight_norm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection

def padding(all_hiddens, max_len, embedding_dim):
    seq_input = []
    seq_mask = []
    for seq in all_hiddens:
        padding_len = max_len - len(seq)
        seq_mask.append(np.concatenate((np.ones(len(seq)),np.zeros(padding_len))).reshape(-1,max_len))
        seq_input.append(np.concatenate(seq,np.zeros((padding_len,embedding_dim))).reshape(-1,max_len,embedding_dim))
    seq_input = np.concatenate(seq_input, axis=0)
    seq_mask = np.concatenate(seq_mask, axis=0)
    return seq_input, seq_mask

class ATT_dataset(data.Dataset):
    def __init__(self, embedding, substrate, label, max_len):
        super().__init__()
        self.embedding = embedding
        self.substrate = substrate
        self.label = label
        self.max_len = max_len

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, idx):
        return self.embedding[idx], self.substrate[idx], self.label[idx]

    def collate_fn(self, batch:List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        embedding, substrate, target = zip(*batch)
        batch_size = len(embedding)
        emb_dim = embedding[0].shape[1]
        arra = np.full([batch_size,self.max_len,emb_dim], 0.0)
        seq_mask = []
        for arr, seq in zip(arra, embedding):
            padding_len = self.max_len - len(seq)
            seq_mask.append(np.concatenate((np.ones(len(seq)),np.zeros(padding_len))).reshape(-1,self.max_len))
            arrslice = tuple(slice(dim) for dim in seq.shape)
            arr[arrslice] = seq
        seq_mask = np.concatenate(seq_mask, axis=0)        
        return {'embedding': torch.from_numpy(arra), 'mask': torch.from_numpy(seq_mask), 'substrate': torch.tensor(list(substrate)), 'target': torch.tensor(list(target))}

class CNN_dataset(data.Dataset):
    def __init__(self, embeding, substrate, target, max_len):
        super().__init__()
        self.embedding = embeding
        self.substrate = substrate
        self.target = target
        self.max_len = max_len

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, idx):
        return self.embedding[idx], self.substrate[idx], self.target[idx]

    def collate_fn(self, batch:List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        embedding, substrate, target = zip(*batch)
        batch_size = len(embedding)
        emb_dim = embedding[0].shape[1]
        arra = np.full([batch_size,self.max_len,emb_dim], 0.0)
        for arr, seq in zip(arra, embedding):
            arrslice = tuple(slice(dim) for dim in seq.shape)
            arr[arrslice] = seq        
        return {'embedding': torch.from_numpy(arra), 'substrate': torch.tensor(list(substrate)), 'target': torch.tensor(list(target))}

def Attention_Classification_loader(training_embedding,training_mask,training_target,batch_size,validation_embedding,validation_mask,validation_target,test_embedding,test_mask,test_target):
    
    trainloader = data.DataLoader(MyDataSet(training_embedding,training_mask,training_target),batch_size,True)
    validation_loader = data.DataLoader(MyDataSet(validation_embedding,validation_mask,validation_target),batch_size,False)
    test_loader = data.DataLoader(MyDataSet(test_embedding,test_mask,test_target),batch_size,False)
    return trainloader, validation_loader, test_loader

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1,-2))
        scores.masked_fill_(attn_mask,-1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttentionwithonekey(nn.Module):
    def __init__(self,d_model,d_k,n_heads,d_v,out_dim):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.out_dim = out_dim
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, out_dim, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        Q = self.W_Q(input_Q).view(input_Q.size(0),-1, self.n_heads, self.d_k).transpose(1,2)
        K = self.W_K(input_K).unsqueeze(1).repeat(1,self.n_heads,1,1)
        V = self.W_V(input_V).view(input_V.size(0),-1, self.n_heads, self.d_k).transpose(1,2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1,self.n_heads,1,1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(input_Q.size(0), -1, self.n_heads * self.d_v)
        output = self.fc(context) # [batch_size, len_q, out_dim]
        return nn.LayerNorm(self.out_dim).double().cuda()(output), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,out_dim,max_len,sub_dim,d_ff):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(out_dim*max_len+sub_dim, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, 1, bias=False)
            )

    def forward(self, inputs, substrate):
        '''
        inputs: [batch_size, max_len, out_dim]
        '''
        inputs = torch.cat((torch.flatten(inputs, start_dim=1),substrate),1)
        output = self.fc(inputs)
        return output

class EncoderLayer(nn.Module):
    def __init__(self,d_model,d_k,n_heads,d_v,out_dim,max_len,sub_dim,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttentionwithonekey(d_model,d_k,n_heads,d_v,out_dim)
        self.pos_ffn = PoswiseFeedForwardNet(out_dim,max_len,sub_dim,d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask, substrate):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs,substrate) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self,d_model,d_k,n_heads,d_v,out_dim,max_len,sub_dim,d_ff):
        super(Encoder, self).__init__()
        self.layers = EncoderLayer(d_model,d_k,n_heads,d_v,out_dim,max_len,sub_dim,d_ff)

    def get_attn_pad_mask(self, seq_mask):
        batch_size, len_q = seq_mask.size()
        _, len_k = seq_mask.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_mask.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
        return pad_attn_mask.expand(batch_size, len_q, len_k)        

    def forward(self, enc_inputs, input_mask, substrate):
        '''
        enc_inputs: [batch_size, max_len, embedding_dim]
        input_mask: [batch_size, max_len]
        '''

        enc_self_attn_mask = self.get_attn_pad_mask(input_mask) # [batch_size, src_len, src_len]
        # enc_outputs: [batch_size, src_len, out_dim], enc_self_attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attn = self.layers(enc_inputs, enc_self_attn_mask, substrate)
        return enc_outputs, enc_self_attn

def example_training(model,lr,opti,epoch_num,trainloader,validation_loader,save_model,test_loader):
    model.double()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    if opti == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    critation = nn.BCELoss()

    for epoch in range(epoch_num):#1500
        model.train()
        for seq in trainloader:
            enc_inputs, input_mask, target = seq
            enc_inputs, input_mask, target = enc_inputs.double().cuda(),input_mask.double().cuda(), target.double().cuda()
            output, _ = model(enc_inputs,input_mask)
            loss = critation(output,target.view(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        pre = []
        gt = []
        for seq in validation_loader:
            enc_inputs, input_mask, target = seq
            enc_inputs, input_mask = enc_inputs.double().cuda(), input_mask.double().cuda()
            output, _ = model(enc_inputs,input_mask)
            output = output.cpu().detach().numpy().reshape(-1)
            target = target.numpy()
            pre.append(output)
            gt.append(target)
        pre = np.concatenate(pre)
        gt = np.concatenate(gt)
        validation_auc = roc_auc_score(gt,pre)  
        print("epoch: {} | loss: {} | valiloss: {}".format(epoch,loss,validation_auc))
    if save_model:
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, str(epoch)+"_epoch_trial_training.pt")
    test_pre = []
    test_gt = []
    for seq in test_loader:
            enc_inputs, input_mask, target = seq
            input, input_mask = input.double().cuda(), input_mask.double().cuda()
            output, _ = model(input, input_mask)
            output = output.cpu().detach().numpy().reshape(-1)
            target = target.numpy()
            test_pre.append(output)
            test_gt.append(target)
    test_pre = np.concatenate(test_pre)
    test_gt = np.concatenate(test_gt)
    fpr,tpr,_ = roc_curve(test_gt,test_pre)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()       

class CNN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 kernal_1: int,
                 out_dim: int,
                 kernal_2: int,
                 max_len: int,
                 sub_dim: int,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.conv1 = nn.Conv1d(in_dim, hid_dim, kernal_1, padding=int((kernal_1-1)/2))
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.fc_early = nn.Linear(max_len*hid_dim+sub_dim,1)
        self.conv2 = nn.Conv1d(hid_dim, out_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.fc = nn.Linear(max_len*out_dim+sub_dim,1)
        self.cls = nn.Sigmoid()

    def forward(self,enc_inputs, substrate):
        """
        input:[batch_size,seq_len,embed_dim]
        """
        output = enc_inputs.transpose(1, 2)
        output = nn.functional.relu(self.conv1(self.norm(output)))
        output = self.dropout(output)
        single_conv = torch.cat((torch.flatten(output, 1),substrate),1)
        single_conv = self.fc_early(single_conv)
        output = nn.functional.relu(self.conv2(output))
        output = self.dropout2(output)
        output = torch.cat((torch.flatten(output, 1),substrate),1)
        output = self.fc(output)
        return output, single_conv




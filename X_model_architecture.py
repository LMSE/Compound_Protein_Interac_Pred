#!/usr/bin/env python
# coding: utf-8

from torch import nn
from torch.utils import data
from torch.nn.utils.weight_norm import weight_norm

class Model_C_pooling(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 kernal_1: int,
                 out_dim: int,
                 kernal_2: int,
                 max_len: int,
                 sub_dim: int,
                 last_hid: int,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.conv1 = nn.Conv1d(in_dim, hid_dim, kernal_1, padding=int((kernal_1-1)/2))
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        #--------------------------------------------------#
        self.conv2_1 = nn.Conv1d(hid_dim, out_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout2_1 = nn.Dropout(dropout, inplace=True)
        #--------------------------------------------------#
        self.conv2_2 = nn.Conv1d(hid_dim, hid_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout2_2 = nn.Dropout(dropout, inplace=True)
        #--------------------------------------------------#
        self.fc_early = nn.Linear(max_len*hid_dim+sub_dim,1)
        #--------------------------------------------------#
        self.conv3 = nn.Conv1d(hid_dim, out_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout3 = nn.Dropout(dropout, inplace=True)
        #self.pooling = nn.MaxPool1d(3, stride=3, padding=1)
        #--------------------------------------------------#
        self.fc_1 = nn.Linear(int(2*max_len*out_dim+sub_dim),last_hid)
        self.fc_2 = nn.Linear(last_hid,last_hid)
        self.fc_3 = nn.Linear(last_hid,1)
        self.cls = nn.Sigmoid()

    def forward(self, enc_inputs, substrate):
        #--------------------------------------------------#
        output = enc_inputs.transpose(1, 2)
        output = nn.functional.relu(self.conv1(self.norm(output)))
        output = self.dropout1(output)
        #--------------------------------------------------#
        output_1 = nn.functional.relu(self.conv2_1(output))
        output_1 = self.dropout2_1(output_1)
        #--------------------------------------------------#
        output_2 = nn.functional.relu(self.conv2_2(output)) + output
        output_2 = self.dropout2_2(output_2)
        #--------------------------------------------------#
        single_conv = torch.cat( (torch.flatten(output_2,1),substrate) ,1)
        single_conv = self.cls(self.fc_early(single_conv))
        #--------------------------------------------------#
        output_2 = nn.functional.relu(self.conv3(output_2))
        output_2 = self.dropout3(output_2)
        #--------------------------------------------------#
        output = torch.cat((output_1,output_2),1)
        #--------------------------------------------------#
        #output = self.pooling(output)
        #--------------------------------------------------#
        output = torch.cat( (torch.flatten(output,1), substrate) ,1)
        #--------------------------------------------------#
        output = self.fc_1(output)
        output = nn.functional.relu(output)
        output = self.fc_2(output)
        output = nn.functional.relu(output)
        output = self.fc_3(output)
        return output, single_conv

#====================================================================================================#
model = Model_C_pooling(
                        in_dim = NN_input_dim,
                        hid_dim = hid_dim,      # 256
                        kernal_1 = kernal_1,    # 3
                        out_dim = out_dim,      # 2
                        kernal_2 = kernal_2,    # 3
                        max_len = seqs_max_len, # 295 for phosphatase dataset, with 14 seq removed
                        sub_dim = X_subs_encodings_dim, #1413 for ECFP6
                        last_hid = last_hid,    # 1024
                        dropout = 0.
                        )
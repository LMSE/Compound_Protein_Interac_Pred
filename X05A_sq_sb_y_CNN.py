#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Microsoft VS header 
import os, os.path
# import winsound
from sys import platform
if os.name == 'nt' or platform == 'win32':
    try:
        os.chdir(os.path.dirname(__file__))
        print("Running in Microsoft VS!")
    except:
        print("Not Running in Microsoft VS")
#########################################################################################################
#########################################################################################################
import sys
import time
import torch
import numpy as np
import pandas as pd
import pickle
import argparse
import scipy
import random
import subprocess
#--------------------------------------------------#
from torch import nn
from torch.utils import data
from torch.nn.utils.weight_norm import weight_norm
#from torchvision import models
#from torchsummary import summary
#--------------------------------------------------#
from tape import datasets
from tape import TAPETokenizer
#--------------------------------------------------#
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#--------------------------------------------------#
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
#--------------------------------------------------#
#from sklearn import svm
#from sklearn.model_selection import GridSearchCV
#from sklearn.tree import DecisionTreeRegressor
#--------------------------------------------------#
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#--------------------------------------------------#
from tpot import TPOTRegressor
from ipywidgets import IntProgress
from pathlib import Path
from copy import deepcopy
#--------------------------------------------------#
from datetime import datetime
#--------------------------------------------------#
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
## Args
Step_code="X05A_"
data_folder = Path("X_DataProcessing/")
embedding_file_list = ["X03_embedding_ESM_1B.p", 
                       "X03_embedding_BERT.p", 
                       "X03_embedding_TAPE.p", 
                       "X03_embedding_ALBERT.p", 
                       "X03_embedding_T5.p", 
                       "X03_embedding_TAPE_FT.p", 
                       "X03_embedding_Xlnet.p"]
embedding_file = embedding_file_list[1]
properties_file= "X00_substrates_properties_list.p"
# embedding_file is a dict, {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
# properties_file is a list, [ SUBSTRATE_name, Y_Property_value, Y_Property_Class_#1, Y_Property_Class_#2, SMILES_str ]
#====================================================================================================#
# Select properties (Y) of the model 
screen_bool = False
classification_threshold_type = 2 # 2: 1e-5, 3: 1e-2
log_value = False ##### !!!!! If value is True, screen_bool will be changed
#====================================================================================================#
# Select substrate encodings
subs_encodings_list = ["ECFP2", "ECFP4", "ECFP6", "JTVAE", "MorganFP"]
subs_encodings = subs_encodings_list[2]
#---------- ECFP
ECFP_type = subs_encodings[-1] if subs_encodings in ["ECFP2", "ECFP4", "ECFP6",] else 2 # 2, 4, 6
#---------- JTVAE
data_folder_2 = Path("X_DataProcessing/X00_enzyme_datasets_processed/")
subs_JTVAE_file="phosphatase_JTVAE_features.p"
#---------- Morgan
data_folder = Path("X_DataProcessing/")
subs_Morgan1024_file = "X00_phosphatase_Morgan1024_features.p"
subs_Morgan2048_file = "X00_phosphatase_Morgan2048_features.p"
#====================================================================================================#
# Data Split Methods
split_type = 0 # 0, 1, 2, 3
# split_type = 0, train/test/split completely randomly selected
# split_type = 1, train/test/split looks at different seq-subs pairs
# split_type = 2, train/test/split looks at different seqs
# split_type = 3, train/test/split looks at different subs
custom_split = True
customized_idx_file = "X02_customized_idx_selected.p"
#====================================================================================================#
# Prediction NN settings
epoch_num=100
batch_size=256
learning_rate=0.0001
NN_type_list=["Reg", "Clf"]
NN_type=NN_type_list[0]
#====================================================================================================#
hid_dim = 256   # 256
kernal_1 = 3    # 5
out_dim = 1     # 2
kernal_2 = 3    # 3
last_hid = 1024  # 1024
dropout = 0.    # 0
#--------------------------------------------------#
'''
model = CNN(
            in_dim = NN_input_dim,
            hid_dim = 1024,
            kernal_1 = 5,
            out_dim = 2, #2
            kernal_2 = 3,
            max_len = seqs_max_len,
            sub_dim = X_subs_encodings_dim,
            last_hid = 2048, #256
            dropout = 0.
            )
            '''
#====================================================================================================#
if log_value==True:
    screen_bool = True
if NN_type=="Clf": ##### !!!!! If value is "Clf", log_value will be changed
    screen_bool = False # Actually Useless
    log_value==False
#====================================================================================================#
# Results
results_folder = Path("X_DataProcessing/" + Step_code +"intermediate_results/")
i_o_put_file_1 = "X04B_all_ecfps" + str(ECFP_type) + ".p"
i_o_put_file_2 = "X04B_all_cmpds_ecfps" + str(ECFP_type) + "_dict.p"
output_file_3 = Step_code + "_all_X_y.p"
output_file_header = Step_code + "_result_"
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Create Temp Folder for Saving Results
print(">>>>> Creating temporary subfolder and clear past empty folders! <<<<<")
now = datetime.now()
#d_t_string = now.strftime("%Y%m%d_%H%M%S")
d_t_string = now.strftime("%m%d-%H%M%S")
#====================================================================================================#
results_folder_contents = os.listdir(results_folder)
for item in results_folder_contents:
    if os.path.isdir(results_folder / item):
        try:
            os.rmdir(results_folder / item)
            print("Remove empty folder " + item + "!")
        except:
            print("Found Non-empty folder " + item + "!")
embedding_code=embedding_file.replace("X03_embedding_", "")
embedding_code=embedding_code.replace(".p", "")
temp_folder_name = Step_code + d_t_string + "_" + embedding_code.replace("_","") + "_" + subs_encodings + "_" + NN_type + "_Split" + str(split_type) + "_screen" + str(screen_bool) + "_log" + str(log_value) + "_threshold" + str(classification_threshold_type)
results_sub_folder=Path("X_DataProcessing/" + Step_code + "intermediate_results/" + temp_folder_name +"/")
if not os.path.exists(results_sub_folder):
    os.makedirs(results_sub_folder)
print(">>>>> Temporary subfolder created! <<<<<")
#########################################################################################################
#########################################################################################################
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()
#--------------------------------------------------#
orig_stdout = sys.stdout
f = open(results_sub_folder / 'print_out.txt', 'w')
sys.stdout = Tee(sys.stdout, f)
print("="*50)
#--------------------------------------------------#
print("embedding_file: ", embedding_file)
#--------------------------------------------------#
print("log_value: ", log_value," --- Use log values of Y.")
print("screen_bool: ", screen_bool, " --- Whether zeros shall be removed")
print("classification_threshold_type: ", classification_threshold_type, " --- 2: 1e-5, 3: 1e-2")
#--------------------------------------------------#
print("subs_encodings: ", subs_encodings)
print("ECFP_type: ", ECFP_type)
#--------------------------------------------------#
print("split_type: ", split_type)
print("custom_split: ", custom_split)
#--------------------------------------------------#
print("epoch_num: ", epoch_num)
print("batch_size: ", batch_size)
print("learning_rate: ", learning_rate)
print("NN_type: ", NN_type)
#--------------------------------------------------#
print("-"*50)
for i in ['hid_dim', 'kernal_1', 'out_dim', 'kernal_2', 'last_hid', 'dropout']:
    print(i, ": ", locals()[i])
print("-"*50)
#########################################################################################################
#########################################################################################################
# Get Input files
# Get Sequence Embeddings from X03 pickles.
with open( data_folder / embedding_file, 'rb') as seqs_embeddings:
    seqs_embeddings_pkl = pickle.load(seqs_embeddings)
X_seqs_all_hiddens_list = seqs_embeddings_pkl['seq_all_hiddens'] # new: X_seqs_all_hiddens_list, old: seq_all_hiddens_list
#====================================================================================================#
# Get subs_properties_list.
with open( data_folder / properties_file, 'rb') as subs_properties:
    subs_properties_list = pickle.load(subs_properties) # [[one_substrate, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES], [], [], ...[] ]
#====================================================================================================#
# Get Substrate Encodings (JTVAE/Morgan) from processed data.
with open( data_folder_2 / subs_JTVAE_file, 'rb') as subs_JTVAE_info:
    subs_SMILES_JTVAE_dict = pickle.load(subs_JTVAE_info)
#====================================================================================================#
# Get Substrate Encodings (JTVAE/Morgan) from processed data.
with open( data_folder / subs_Morgan1024_file, 'rb') as subs_Morgan1024_info:
    subs_SMILES_Morgan1024_dict = pickle.load(subs_Morgan1024_info)
#====================================================================================================#
with open( data_folder / customized_idx_file, 'rb') as customized_idx_list:
    customized_idx_list = pickle.load(customized_idx_list)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# ECFP from CDK java file
def CDK_ECFP(smiles_str,ecfp_type,iteration_number):    
    # Use java file CDKImpl class to get ECFP from cmd line
    query_str1='java -cp .;cdk-2.2.jar CDKImpl ' + smiles_str + ' ' + ecfp_type + ' ' + str(iteration_number)
    query_result = subprocess.check_output(query_str1, shell=True)
    query_result = query_result.decode("gb2312")
    query_result=query_result.replace('[','')
    query_result=query_result.replace(']','')
    query_result=query_result.replace(' ','')
    query_result=query_result.replace('\n','')
    query_result=query_result.replace('\r','')
    if query_result!="":
        if query_result[-1]==',':
            query_result=query_result[0:-1]
        list_of_ecfp=query_result.split(",")
    else:
        list_of_ecfp=[]
    return list_of_ecfp 
#====================================================================================================#
def get_full_ecfp(smiles_str,ecfp_type,iteration_number):   
    # ECFP4 + itr2 or ECFP2 + itr1
    full_ecfp_list=[]
    for i in range(iteration_number+1):
        full_ecfp_list=full_ecfp_list+CDK_ECFP(smiles_str,ecfp_type,i)
    return full_ecfp_list
#====================================================================================================#
def generate_all_ECFPs(list_smiles,ecfp_type="ECFP2",iteration_number=1):
# return a list of ECFPs of all depth for a list of compounds (UNIQUE!!!)
    all_ecfps=set([])
    for smiles_a in list_smiles:
        discriptors = get_full_ecfp(smiles_a,ecfp_type,iteration_number)
        #print(smiles_a)
        all_ecfps=all_ecfps.union(set(discriptors))
    return all_ecfps
#====================================================================================================#
def generate_all_smiles_ecfps_dict(list_smiles,ecfp_type="ECFP2",iteration_number=1):
    all_smiles_ecfps_dict=dict([])
    for smiles_a in list_smiles:
        #print(smiles_a)
        all_smiles_ecfps_dict[smiles_a]=get_full_ecfp(smiles_a,ecfp_type,iteration_number)
    return all_smiles_ecfps_dict
#====================================================================================================#
def generate_all_smiles_ecfps_list_dict(list_smiles,ecfp_type="ECFP2",iteration_number=1):
    all_ecfps=set([])
    all_smiles_ecfps_dict=dict([])
    for smiles_a in list_smiles:
        discriptors = get_full_ecfp(smiles_a,ecfp_type,iteration_number)
        #print(smiles_a)
        all_smiles_ecfps_dict[smiles_a]=discriptors
        all_ecfps=all_ecfps.union(set(discriptors))
    return list(all_ecfps),all_smiles_ecfps_dict
#====================================================================================================#
# Get all substrates ECFP encoded.
all_smiles_list=[]
for one_list_prpt in subs_properties_list:
    all_smiles_list.append(one_list_prpt[-1])
#print(all_smiles_list)
#--------------------------------------------------#
#all_ecfps,all_smiles_ecfps_dict=generate_all_smiles_ecfps_list_dict(all_smiles_list,ecfp_type="ECFP4",iteration_number=1)
#pickle.dump(all_ecfps, open(data_folder / i_o_put_file_1,"wb") )
#pickle.dump(all_smiles_ecfps_dict, open(data_folder / i_o_put_file_2,"wb"))
#====================================================================================================#
with open( data_folder / i_o_put_file_1, 'rb') as all_ecfps:
    all_ecfps = pickle.load(all_ecfps)
with open( data_folder / i_o_put_file_2, 'rb') as all_smiles_ecfps_dict:
    all_smiles_ecfps_dict = pickle.load(all_smiles_ecfps_dict)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def Get_represented_X_y_data(X_seqs_all_hiddens_list, subs_properties_list, screen_bool, classification_threshold_type):
    # new: X_seqs_all_hiddens_list, old: seq_all_hiddens_list
    # subs_properties_list: [[one_substrate, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES], [], [], ...[] ]
    X_seqs_all_hiddens = []
    X_subs_representations=[]
    y_data = []
    seqs_subs_idx_book=[]
    print("len(X_seqs_all_hiddens_list): ", len(X_seqs_all_hiddens_list))
    for i in range(len(subs_properties_list)):
        for j in range(len(X_seqs_all_hiddens_list)):
            X_smiles_rep = subs_properties_list[i][-1] # list of SMILES
            X_one_all_hiddens = X_seqs_all_hiddens_list[j]
            if not (screen_bool==True and subs_properties_list[i][classification_threshold_type][j]==False):
                # classification_threshold_type ----> # 2: Threshold = 1e-5 | 3: Threshold = 1e-2 (from Original CSV File)
                # subs_properties_list[i] ----> [one_substrate, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES]
                # subs_properties_list[i][classification_threshold_type] ----> y_prpty_cls OR y_prpty_cls2
                # subs_properties_list[i][classification_threshold_type][j] ----> y_prpty_cls[j] OR y_prpty_cls2[j]
                # subs_properties_list[i][1][j] ----> y_prpty_reg[j]
                X_seqs_all_hiddens.append(X_one_all_hiddens)
                X_subs_representations.append(X_smiles_rep)
                y_data.append(subs_properties_list[i][1][j])
                seqs_subs_idx_book.append([j, i]) # [seqs_idx, subs_idx]
    return X_seqs_all_hiddens, X_subs_representations, y_data, seqs_subs_idx_book
#====================================================================================================#
def Get_represented_X_y_data_clf(X_seqs_all_hiddens_list, subs_properties_list, screen_bool, classification_threshold_type):
    # new: X_seqs_all_hiddens_list, old: seq_all_hiddens_list
    # subs_properties_list: [[one_substrate, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES], [], [], ...[] ]
    X_seqs_all_hiddens = []
    X_subs_representations=[]
    y_data = []
    seqs_subs_idx_book=[]
    print("len(X_seqs_all_hiddens_list): ", len(X_seqs_all_hiddens_list))
    for i in range(len(subs_properties_list)):
        for j in range(len(X_seqs_all_hiddens_list)):
            X_smiles_rep = subs_properties_list[i][-1] # list of SMILES
            X_one_all_hiddens = X_seqs_all_hiddens_list[j]
            # classification_threshold_type ----> # 2: Threshold = 1e-5 | 3: Threshold = 1e-2 (from Original CSV File)
            # subs_properties_list[i] ----> [one_substrate, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES]
            # subs_properties_list[i][classification_threshold_type] ----> y_prpty_cls OR y_prpty_cls2
            # subs_properties_list[i][classification_threshold_type][j] ----> y_prpty_cls[j] OR y_prpty_cls2[j]
            X_seqs_all_hiddens.append(X_one_all_hiddens)
            X_subs_representations.append(X_smiles_rep)
            y_data.append(subs_properties_list[i][classification_threshold_type][j])
            seqs_subs_idx_book.append([j, i]) # [seqs_idx, subs_idx]
    return X_seqs_all_hiddens, X_subs_representations, y_data, seqs_subs_idx_book
#########################################################################################################
#########################################################################################################
# Encode Substrates.
#====================================================================================================#
def list_smiles_to_ecfp_through_dict(smiles_list, all_smiles_ecfps_dict):
    ecfp_list=[]
    for one_smiles in smiles_list:
        ecfp_list=ecfp_list + all_smiles_ecfps_dict[one_smiles]
    return ecfp_list
#====================================================================================================#
def smiles_to_ECFP_vec( smiles_x, all_ecfps, all_smiles_ecfps_dict):
    dimension=len(all_ecfps)
    Xi=[0]*dimension
    Xi_ecfp_list=list_smiles_to_ecfp_through_dict( [smiles_x, ] ,all_smiles_ecfps_dict)
    for one_ecfp in Xi_ecfp_list:
        Xi[all_ecfps.index(one_ecfp)]=Xi_ecfp_list.count(one_ecfp)
    return np.array(Xi)
#====================================================================================================#
def Get_ECFPs_encoding(X_subs_representations, all_ecfps, all_smiles_ecfps_dict):
    X_subs_encodings=[]
    for one_smiles in X_subs_representations:
        one_subs_encoding = smiles_to_ECFP_vec(one_smiles, all_ecfps, all_smiles_ecfps_dict) # substrate_encoding
        X_subs_encodings.append(one_subs_encoding)
    return X_subs_encodings
#====================================================================================================#
def Get_JTVAE_encoding(X_subs_representations, subs_SMILES_JTVAE_dict):
    X_subs_encodings=[]
    for one_smiles in X_subs_representations:
        one_subs_encoding = subs_SMILES_JTVAE_dict[one_smiles] # substrate_encoding
        X_subs_encodings.append(one_subs_encoding)
    return X_subs_encodings
#====================================================================================================#
def Get_Morgan_encoding(X_subs_representations, subs_SMILES_Morgan1024_dict):
    X_subs_encodings=[]
    for one_smiles in X_subs_representations:
        one_subs_encoding = subs_SMILES_Morgan1024_dict[one_smiles] # substrate_encoding
        X_subs_encodings.append(one_subs_encoding)
    return X_subs_encodings
#########################################################################################################
#########################################################################################################
if NN_type=="Reg":
    X_seqs_all_hiddens, X_subs_representations, y_data, seqs_subs_idx_book = Get_represented_X_y_data(X_seqs_all_hiddens_list, subs_properties_list, screen_bool, classification_threshold_type)
if NN_type=="Clf":
    X_seqs_all_hiddens, X_subs_representations, y_data, seqs_subs_idx_book = Get_represented_X_y_data_clf(X_seqs_all_hiddens_list, subs_properties_list, classification_threshold_type)
#====================================================================================================#
subs_encodings_list = ["ECFP2", "ECFP4", "ECFP6", "JTVAE", "MorganFP"]
if subs_encodings == "ECFP2" or subs_encodings == "ECFP4"  or subs_encodings == "ECFP6" :
    X_subs_encodings = Get_ECFPs_encoding(X_subs_representations, all_ecfps, all_smiles_ecfps_dict)
if subs_encodings == "JTVAE":
    X_subs_encodings = Get_JTVAE_encoding(X_subs_representations, subs_SMILES_JTVAE_dict)
if subs_encodings == "MorganFP":
    X_subs_encodings = Get_Morgan_encoding(X_subs_representations, subs_SMILES_Morgan1024_dict)
#====================================================================================================#
#save_dict=dict([])
#save_dict["X_seqs_all_hiddens"] = X_seqs_all_hiddens
#save_dict["X_subs_encodings"] = X_subs_encodings
#save_dict["y_data"] = y_data
#pickle.dump( save_dict , open( results_folder / output_file_3, "wb" ) )
#print("Done getting X_seqs_all_hiddens, X_subs_encodings and y_data!")
print("len(X_seqs_all_hiddens): ", len(X_seqs_all_hiddens), ", len(X_subs_encodings): ", len(X_subs_encodings), ", len(y_data): ", len(y_data) )
#====================================================================================================#
# Get size of some interests
X_seqs_all_hiddens_dim = [ max([ X_seqs_all_hiddens_list[i].shape[0] for i in range(len(X_seqs_all_hiddens_list)) ]), X_seqs_all_hiddens_list[0].shape[1], ]
X_subs_encodings_dim = len(X_subs_encodings[0])
X_seqs_num = len(X_seqs_all_hiddens_list)
X_subs_num = len(subs_properties_list)
print("seqs, subs dimensions: ", X_seqs_all_hiddens_dim, ", ", X_subs_encodings_dim)
print("seqs, subs counts: ", X_seqs_num, ", ", X_subs_num)

seqs_max_len = max([  X_seqs_all_hiddens_list[i].shape[0] for i in range(len(X_seqs_all_hiddens_list))  ])
print("seqs_max_len: ", seqs_max_len)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Get Separate SEQS index and SUBS index.
def split_idx(X_num, train_split, test_split, random_state=42):
    # X_seqs_idx = y_seqs_idx = list(range(len(X_seqs_all_hiddens_list)))
    # X_subs_idx = y_subs_idx = list(range(len(subs_properties_list)))
    X_idx = y_idx = list(range(X_num))
    X_tr_idx, X_ts_idx, y_tr_idx, y_ts_idx = train_test_split(X_idx, y_idx, test_size=(1-train_split), random_state=42)
    X_va_idx, X_ts_idx, y_va_idx, y_ts_idx = train_test_split(X_ts_idx, y_ts_idx, test_size = (test_split/(1.0-train_split)) , random_state=42)
    return X_tr_idx, X_ts_idx, X_va_idx
#====================================================================================================#
tr_idx_seqs, ts_idx_seqs, va_idx_seqs = split_idx(X_seqs_num, train_split=0.8, test_split=0.1, random_state=42)
tr_idx_subs, ts_idx_subs, va_idx_subs = split_idx(X_subs_num, train_split=0.8, test_split=0.1, random_state=42)
#########################################################################################################
#########################################################################################################
# Get Customized SEQS index if needed.
def split_seqs_idx_custom(X_num, customized_idx_list, valid_test_split=0.5, random_state=42):
    X_idx = y_idx = list(range(X_num))
    X_tr_idx = [idx for idx in X_idx if (idx not in customized_idx_list)]
    #--------------------------------------------------#
    X_ts_idx = y_ts_idx = customized_idx_list
    X_va_idx, X_ts_idx, y_va_idx, y_ts_idx = train_test_split(X_ts_idx, y_ts_idx, test_size = valid_test_split , random_state=42)
    return X_tr_idx, X_ts_idx, X_va_idx
#====================================================================================================#
if custom_split == True:
    tr_idx_seqs, ts_idx_seqs, va_idx_seqs = split_seqs_idx_custom(X_seqs_num, customized_idx_list, valid_test_split=0.5, random_state=42)
#====================================================================================================#
print("len(tr_idx_seqs): ", len(tr_idx_seqs))
print("len(ts_idx_seqs): ", len(ts_idx_seqs))
print("len(va_idx_seqs): ", len(va_idx_seqs))
#########################################################################################################
#########################################################################################################
# Get splitted index of the entire combined dataset.
def split_seqs_subs_idx_book(tr_idx_seqs, ts_idx_seqs, va_idx_seqs, tr_idx_subs, ts_idx_subs, va_idx_subs, seqs_subs_idx_book, split_type):
    #--------------------------------------------------#
    # split_type = 0, train/test/split completely randomly selected
    # split_type = 1, train/test/split looks at different seq-subs pairs
    # split_type = 2, train/test/split looks at different seqs
    # split_type = 3, train/test/split looks at different subs
    #--------------------------------------------------#
    tr_idx, ts_idx, va_idx = [], [], []
    if split_type==0:
        dataset_size = len(seqs_subs_idx_book)
        X_data_idx = np.array(list(range(dataset_size)))
        tr_idx, ts_idx, y_train, y_test = train_test_split(X_data_idx, y_data, test_size=0.3, random_state=42)
        va_idx, ts_idx, y_valid, y_test = train_test_split(ts_idx, y_test, test_size=0.6667, random_state=42)
    for one_pair_idx in seqs_subs_idx_book:
        if split_type==1:
            if one_pair_idx[0] in tr_idx_seqs and one_pair_idx[1] in tr_idx_subs:
                tr_idx.append(seqs_subs_idx_book.index(one_pair_idx))
            if one_pair_idx[0] in va_idx_seqs and one_pair_idx[1] in va_idx_subs:
                va_idx.append(seqs_subs_idx_book.index(one_pair_idx))
            if one_pair_idx[0] in ts_idx_seqs and one_pair_idx[1] in ts_idx_subs:
                ts_idx.append(seqs_subs_idx_book.index(one_pair_idx))
        if split_type==2: 
            if one_pair_idx[0] in tr_idx_seqs:
                tr_idx.append(seqs_subs_idx_book.index(one_pair_idx))
            if one_pair_idx[0] in va_idx_seqs:
                va_idx.append(seqs_subs_idx_book.index(one_pair_idx))
            if one_pair_idx[0] in ts_idx_seqs:
                ts_idx.append(seqs_subs_idx_book.index(one_pair_idx))
        if split_type==3:
            if one_pair_idx[1] in tr_idx_subs:
                tr_idx.append(seqs_subs_idx_book.index(one_pair_idx))
            if one_pair_idx[1] in va_idx_subs:
                va_idx.append(seqs_subs_idx_book.index(one_pair_idx))
            if one_pair_idx[1] in ts_idx_subs:
                ts_idx.append(seqs_subs_idx_book.index(one_pair_idx))
    return tr_idx, ts_idx, va_idx

X_train_idx, X_test_idx, X_valid_idx = split_seqs_subs_idx_book(tr_idx_seqs, ts_idx_seqs, va_idx_seqs, tr_idx_subs, ts_idx_subs, va_idx_subs, seqs_subs_idx_book, split_type)
dataset_size = len(seqs_subs_idx_book)
print("dataset_size: ", dataset_size)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Get splitted data of the combined dataset through the index.
def Get_X_y_data_selected(X_idx, X_seqs_all_hiddens, X_subs_encodings, y_data):
    X_seqs_selected=[]
    X_subs_selected=[]
    y_data_selected=[]
    for idx in X_idx:
        X_seqs_selected.append(X_seqs_all_hiddens[idx])
        X_subs_selected.append(X_subs_encodings[idx])
        y_data_selected.append(y_data[idx])
    X_seqs_selected = X_seqs_selected
    X_subs_selected = X_subs_selected
    y_data_selected = np.array(y_data_selected)
    if log_value==True:
        y_data_selected=np.log10(y_data_selected)
    return X_seqs_selected, X_subs_selected, y_data_selected
#====================================================================================================#
X_tr_seqs, X_tr_subs, y_tr = Get_X_y_data_selected(X_train_idx, X_seqs_all_hiddens, X_subs_encodings, y_data)
X_ts_seqs, X_ts_subs, y_ts = Get_X_y_data_selected(X_test_idx, X_seqs_all_hiddens, X_subs_encodings, y_data)
X_va_seqs, X_va_subs, y_va = Get_X_y_data_selected(X_valid_idx, X_seqs_all_hiddens, X_subs_encodings, y_data)
#print("Done getting X_data and y_data!")
print("X_tr_seqs_dimension: ", len(X_tr_seqs), ", X_tr_subs_dimension: ", len(X_tr_subs), ", y_tr_dimension: ", y_tr.shape )
print("X_ts_seqs_dimension: ", len(X_ts_seqs), ", X_ts_subs_dimension: ", len(X_ts_subs), ", y_ts_dimension: ", y_ts.shape )
print("X_va_seqs_dimension: ", len(X_va_seqs), ", X_va_subs_dimension: ", len(X_va_subs), ", y_va_dimension: ", y_va.shape )
#########################################################################################################
#########################################################################################################
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
        return {'seqs_embeddings': torch.from_numpy(arra), 'subs_encodings': torch.tensor(list(substrate)), 'y_property': torch.tensor(list(target))}
#########################################################################################################
#########################################################################################################
def generate_CNN_loader(X_tr_seqs, X_tr_subs, y_tr,
                        X_va_seqs, X_va_subs, y_va,
                        X_ts_seqs, X_ts_subs, y_ts,
                        seqs_max_len, batch_size):
    X_y_tr = CNN_dataset(list(X_tr_seqs), list(X_tr_subs), y_tr, seqs_max_len)
    X_y_va = CNN_dataset(list(X_va_seqs), list(X_va_subs), y_va, seqs_max_len)
    X_y_ts = CNN_dataset(list(X_ts_seqs), list(X_ts_subs), y_ts, seqs_max_len)
    train_loader = data.DataLoader(X_y_tr, batch_size, True,  collate_fn=X_y_tr.collate_fn)
    valid_loader = data.DataLoader(X_y_va, batch_size, False, collate_fn=X_y_va.collate_fn)
    test_loader  = data.DataLoader(X_y_ts, batch_size, False, collate_fn=X_y_ts.collate_fn)
    return train_loader, valid_loader, test_loader

train_loader, valid_loader, test_loader = generate_CNN_loader(X_tr_seqs, X_tr_subs, y_tr, X_va_seqs, X_va_subs, y_va, X_ts_seqs, X_ts_subs, y_ts, seqs_max_len, batch_size)
train_loader_list = [train_loader, ]
valid_loader_list = [valid_loader, ]
test_loader_list  = [test_loader, ]
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
NN_input_dim=X_seqs_all_hiddens_dim[1]
print("NN_input_dim: ", NN_input_dim)
#########################################################################################################
#########################################################################################################
def padding_X_seqs_all_hiddens(X_seqs_all_hiddens, seqs_max_len, embedding_dim):
    seqs_input_rep = []
    for one_seq_all_hidden in X_seqs_all_hiddens:
        padding_len = seqs_max_len - len(one_seq_all_hidden)
        seqs_input_rep.append(np.pad(one_seq_all_hidden, ((0,padding_len),(0,0))).reshape(-1,seqs_max_len,embedding_dim))
    seqs_input_rep = np.concatenate(seqs_input_rep, axis=0)
    return seqs_input_rep
#====================================================================================================#
class LoaderClass(data.Dataset):
    def __init__(self, seqs_embeddings, subs_encodings, y_property):
        super(LoaderClass, self).__init__()
        self.seqs_embeddings = seqs_embeddings
        self.subs_encodings = subs_encodings
        self.y_property = y_property
    def __len__(self):
        return self.seqs_embeddings.shape[0]
    def __getitem__(self, idx):
        return self.seqs_embeddings[idx], self.subs_encodings[idx], self.y_property[idx]
    def collate_fn(self, batch:List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        seqs_embeddings, subs_encodings, y_property = zip(*batch)
        batch_size = len(seqs_embeddings)
        seqs_embeddings_dim = seqs_embeddings[0].shape[1]    
        return {'seqs_embeddings': torch.tensor(seqs_embeddings), 'subs_encodings': torch.tensor(list(subs_encodings)), 'y_property': torch.tensor(list(y_property))}
#====================================================================================================#
def LoaderOrganizer(loader_num, loader_type, ith_loader, dataset_size, seqs_max_len, seqs_embeddings_dim, data_idx, batch_size=128 ):
    # Want to divide loaders to [loader_num] number of train_loaders, test_loaders, or valid_loaders.
    # Return #[ith_loader] loader of type [loader_type]
    # Inputs seq_max_len, embedding_dim are for the padding function
    def divide_list(list_a, num): # num is number of divided sub-list.
        count_a = int(len(list_a)/num) + 1 # count_a is number of items in one divided list.
        for i in range(0, len(list_a), count_a): 
            yield list_a[i:i+count_a]
    #--------------------------------------------------#
    list_index=list(data_idx)
    divided_list_index=list(divide_list(list_index, loader_num))
    print(len(divided_list_index))
    selected_index=divided_list_index[ith_loader]
    selected_X_seqs_all_hiddens = padding_X_seqs_all_hiddens([X_seqs_all_hiddens[i] for i in selected_index], seqs_max_len, seqs_embeddings_dim)
    selected_X_subs_encodings = np.array([X_subs_encodings[i] for i in selected_index])
    selected_y_data = np.array([y_data[i] for i in selected_index])
    selected_X_y_Loader = LoaderClass(selected_X_seqs_all_hiddens, selected_X_subs_encodings, selected_y_data)
    selected_loader = data.DataLoader(selected_X_y_Loader, batch_size, True if (loader_type == "train") else False, collate_fn = selected_X_y_Loader.collate_fn)
    return selected_loader
#########################################################################################################
#########################################################################################################
#train_loader_list = [LoaderOrganizer(10, "train", i, dataset_size, seqs_max_len, X_seqs_all_hiddens_dim[1], X_train_idx) for i in range(10)]
#valid_loader_list = [LoaderOrganizer(5,  "valid", i, dataset_size, seqs_max_len, X_seqs_all_hiddens_dim[1], X_valid_idx) for i in range(5)]
#test_loader_list  = [LoaderOrganizer(2,  "test",  i, dataset_size, seqs_max_len, X_seqs_all_hiddens_dim[1], X_test_idx)  for i in range(2)]

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
class CNN(nn.Module):
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
        #self.pooling = nn.MaxPool1d(3, stride=3,padding=1)
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
model = CNN(
            in_dim = NN_input_dim,
            hid_dim = hid_dim,
            kernal_1 = kernal_1,
            out_dim = out_dim, #2
            kernal_2 = kernal_2,
            max_len = seqs_max_len,
            sub_dim = X_subs_encodings_dim,
            last_hid = last_hid, #256
            dropout = 0.
            )
#########################################################################################################
#########################################################################################################
model.double()
model.cuda()
#--------------------------------------------------#
print("#"*50)
print(model)
#model.float()
#print( summary( model,[(seqs_max_len, NN_input_dim), (X_subs_encodings_dim, )] )  )
#model.double()
print("#"*50)
#--------------------------------------------------#
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
criterion = nn.MSELoss()
#====================================================================================================#
for epoch in range(epoch_num): 
    model.train()
    #====================================================================================================#
    count_x=0
    for train_loader in train_loader_list:
        for one_seqsubs_ppt_group in train_loader:
            len_train_loader=len(train_loader)
            count_x+=1
            if ((count_x) % 10) == 0:
                print(str(count_x)+"/"+str(len_train_loader)+"->", end=" ")
            #--------------------------------------------------#
            seq_rep, subs_rep, target = one_seqsubs_ppt_group["seqs_embeddings"], one_seqsubs_ppt_group["subs_encodings"], one_seqsubs_ppt_group["y_property"]
            seq_rep, subs_rep, target = seq_rep.double().cuda(), subs_rep.double().cuda(), target.double().cuda()
            output, _ = model(seq_rep, subs_rep)
            loss = criterion(output,target.view(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #====================================================================================================#
    model.eval()
    y_pred_valid = []
    y_real_valid = []
    #--------------------------------------------------#
    for valid_loader in valid_loader_list:
        for one_seqsubs_ppt_group in valid_loader:
            seq_rep, subs_rep, target = one_seqsubs_ppt_group["seqs_embeddings"], one_seqsubs_ppt_group["subs_encodings"], one_seqsubs_ppt_group["y_property"]
            seq_rep, subs_rep = seq_rep.double().cuda(), subs_rep.double().cuda()
            output, _ = model(seq_rep, subs_rep)
            output = output.cpu().detach().numpy().reshape(-1)
            target = target.numpy()
            y_pred_valid.append(output)
            y_real_valid.append(target)
    y_pred_valid = np.concatenate(y_pred_valid)
    y_real_valid = np.concatenate(y_real_valid)
    slope, intercept, r_value_va, p_value, std_err = scipy.stats.linregress(y_pred_valid, y_real_valid)
    #====================================================================================================#
    y_pred = []
    y_real = []
    #--------------------------------------------------#
    for test_loader in test_loader_list:
        for one_seqsubs_ppt_group in test_loader:
            seq_rep, subs_rep, target = one_seqsubs_ppt_group["seqs_embeddings"], one_seqsubs_ppt_group["subs_encodings"], one_seqsubs_ppt_group["y_property"]
            seq_rep, subs_rep = seq_rep.double().cuda(), subs_rep.double().cuda()
            output, _ = model(seq_rep, subs_rep)
            output = output.cpu().detach().numpy().reshape(-1)
            target = target.numpy()
            y_pred.append(output)
            y_real.append(target)
    y_pred = np.concatenate(y_pred)
    y_real = np.concatenate(y_real)
    #--------------------------------------------------#
    if log_value == False:
        y_pred[y_pred<0]=0
    #--------------------------------------------------#
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_pred, y_real)
    print()
    print("epoch: {} | vali_r_value: {} | loss: {} | test_r_value: {} ".format( str((epoch+1)+1000).replace("1","",1) , np.round(r_value_va,4), loss, np.round(r_value, 4)))
    #====================================================================================================#
    if ((epoch+1) % 1) == 0:
        _, _, r_value, _ , _ = scipy.stats.linregress(y_pred, y_real)
        pred_vs_actual_df = pd.DataFrame(np.ones(len(y_pred)))
        pred_vs_actual_df["actual"] = y_real
        pred_vs_actual_df["predicted"] = y_pred
        pred_vs_actual_df.drop(columns=0, inplace=True)
        pred_vs_actual_df.head()
        #--------------------------------------------------#
        sns.set_theme(style="darkgrid")
        y_interval=max(np.concatenate((y_pred, y_real),axis=0))-min(np.concatenate((y_pred, y_real),axis=0))
        x_y_range=(min(np.concatenate((y_pred, y_real),axis=0))-0.1*y_interval, max(np.concatenate((y_pred, y_real),axis=0))+0.1*y_interval)
        g = sns.jointplot(x="actual", y="predicted", data=pred_vs_actual_df,
                            kind="reg", truncate=False,
                            xlim=x_y_range, ylim=x_y_range,
                            color="blue",height=7)

        g.fig.suptitle("Predictions vs. Actual Values, R = " + str(round(r_value,3)) + ", Epoch: " + str(epoch+1) , fontsize=18, fontweight='bold')
        g.fig.tight_layout()
        g.fig.subplots_adjust(top=0.95)
        g.ax_joint.text(0.4,0.6,"", fontsize=12)
        g.ax_marg_x.set_axis_off()
        g.ax_marg_y.set_axis_off()
        g.ax_joint.set_xlabel('Actual Values',fontsize=18 ,fontweight='bold')
        g.ax_joint.set_ylabel('Predictions',fontsize=18 ,fontweight='bold')
        g.savefig(results_sub_folder / (output_file_header + "epoch_" + str(epoch+1)) )
    #====================================================================================================#
        if log_value == False and screen_bool==True:

            y_real = np.delete(y_real, np.where(y_pred == 0.0))
            y_pred = np.delete(y_pred, np.where(y_pred == 0.0))

            y_real = np.log10(y_real)
            y_pred = np.log10(y_pred)

            log_pred_vs_actual_df = pd.DataFrame(np.ones(len(y_pred)))
            log_pred_vs_actual_df["log(actual)"] = y_real
            log_pred_vs_actual_df["log(predicted)"] = y_pred
            log_pred_vs_actual_df.drop(columns=0, inplace=True)

            y_interval = max(np.concatenate((y_pred, y_real),axis=0))-min(np.concatenate((y_pred, y_real),axis=0))
            x_y_range = (min(np.concatenate((y_pred, y_real),axis=0))-0.1*y_interval, max(np.concatenate((y_pred, y_real),axis=0))+0.1*y_interval)
            g = sns.jointplot(x="log(actual)", y="log(predicted)", data=log_pred_vs_actual_df,
                                kind="reg", truncate=False,
                                xlim=x_y_range, ylim=x_y_range,
                                color="blue",height=7)

            g.fig.suptitle("Predictions vs. Actual Values, R = " + str(round(r_value,3)) + ", Epoch: " + str(epoch+1) , fontsize=18, fontweight='bold')
            g.fig.tight_layout()
            g.fig.subplots_adjust(top=0.95)
            g.ax_joint.text(0.4,0.6,"", fontsize=12)
            g.ax_marg_x.set_axis_off()
            g.ax_marg_y.set_axis_off()
            g.ax_joint.set_xlabel('Log(Actual Values)',fontsize=18 ,fontweight='bold')
            g.ax_joint.set_ylabel('Log(Predictions)',fontsize=18 ,fontweight='bold')
            g.savefig(results_sub_folder / (output_file_header + "_log_plot, epoch_" + str(epoch+1)) )
#########################################################################################################
#########################################################################################################












#########################################################################################################
#########################################################################################################
class CNN_old1(nn.Module):
    def __init__(self,
                 in_dim: int,
                 kernal_1: int,
                 out_dim: int,
                 max_len: int,
                 sub_dim: int,
                 last_hid: int,
                 dropout: float = 0.
                 ):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernal_1, padding=int((kernal_1-1)/2)) 
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.fc_1 = nn.Linear(max_len*out_dim+sub_dim,last_hid)
        self.fc_2 = nn.Linear(last_hid,1)
        self.cls = nn.Sigmoid()

    def forward(self,enc_inputs, substrate):
        #input:[batch_size,seq_len,embed_dim]
        output = enc_inputs.transpose(1, 2)
        output = nn.functional.relu(self.conv1(self.norm(output)))
        output = self.dropout(output)
        single_conv = torch.cat(  (torch.flatten(output,1),  substrate) ,1)
        output = torch.cat(  (torch.flatten(output,1),  substrate) ,1)
        output = self.fc_1(output)
        output = nn.functional.relu(output)
        output = self.fc_2(output)
        return output, single_conv

#########################################################################################################
#########################################################################################################
class CNN_old2(nn.Module):
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
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.fc_early = nn.Linear(max_len*hid_dim+sub_dim,1)
        self.conv2 = nn.Conv1d(hid_dim, out_dim, kernal_2, padding=int((kernal_2-1)/2))
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        self.fc_1 = nn.Linear(max_len*out_dim+sub_dim,last_hid)
        self.fc_2 = nn.Linear(last_hid,1)
        self.cls = nn.Sigmoid()
    def forward(self, enc_inputs, substrate):
        output = enc_inputs.transpose(1, 2)
        output = nn.functional.relu(self.conv1(self.norm(output)))
        output = self.dropout(output)
        single_conv = torch.cat(  (torch.flatten(output,1),  substrate) ,1)
        single_conv = self.cls(self.fc_early(single_conv))
        output = nn.functional.relu(self.conv2(output))
        output = self.dropout2(output)
        output = torch.cat(  (torch.flatten(output,1),  substrate) ,1)
        output = self.fc_1(output)
        output = nn.functional.relu(output)
        output = self.fc_2(output)
        return output, single_conv

#########################################################################################################
#########################################################################################################
# For Attention Layer (will NOT be called now!)
def padding(all_hiddens, seq_max_len, embedding_dim):
    seq_input_rep = []
    seq_mask = []
    for seq in all_hiddens:
        padding_len = seq_max_len - seq.shape[0]
        seq_mask.append(np.concatenate((np.ones(seq.shape[0]),np.zeros(padding_len))).reshape(-1,seq_max_len))
        seq_input_rep.append(np.pad(seq, ((0,padding_len),(0,0))).reshape(-1,seq_max_len,embedding_dim))
    seq_input_rep = np.concatenate(seq_input_rep, axis=0)
    seq_mask = np.concatenate(seq_mask, axis=0)
    return seq_input_rep, seq_mask


#====================================================================================================#
'''
model = CNN_old1(
            in_dim = NN_input_dim,
            kernal_1 = 3,
            out_dim = 2, #2
            max_len = seqs_max_len,
            sub_dim = X_subs_encodings_dim,
            last_hid = 256, #256
            dropout = 0.
            )
'''
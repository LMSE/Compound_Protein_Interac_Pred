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
#--------------------------------------------------#
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
#--------------------------------------------------#
# Seqs_embeddings and properties
Step_code = "X04B_"
data_folder = Path("X_DataProcessing/")
embedding_file_list = ["X03_embedding_ESM_1B.p", 
                       "X03_embedding_BERT.p", 
                       "X03_embedding_TAPE.p", 
                       "X03_embedding_ALBERT.p", 
                       "X03_embedding_T5.p", 
                       "X03_embedding_TAPE_FT.p", 
                       "X03_embedding_Xlnet.p"]
embedding_file = embedding_file_list[0]
properties_file= "X00_substrates_properties_list.p" # [[one_substrate, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES], [], [], ...[] ]
# embedding_file is a dict, {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
# properties_file is a list, [ SUBSTRATE_name, Y_Property_value, Y_Property_Class_#1, Y_Property_Class_#2, SMILES_str ]
#--------------------------------------------------#
# Select properties (Y) of the model 
screen_bool = False
classification_threshold_type = 3 # 2: 1e-5, 3: 1e-2
log_value = False ##### !!!!! If value is True, screen_bool will be changed
#--------------------------------------------------#
# Select substrate encodings
#-------------------      0        1        2        3          4         5        6     
subs_encodings_list = ["ECFP2", "ECFP4", "ECFP6", "JTVAE", "MorganFP", "ECFP8", "ECFPX"]
subs_encodings = subs_encodings_list[1]
#---------- ECFP
ECFP_type = subs_encodings[-1] if subs_encodings in ["ECFP2", "ECFP4", "ECFP6", "ECFP8", "ECFPX"] else "4" # 2, 4, 6, 8, X
#---------- JTVAE
data_folder_2 = Path("X_DataProcessing/X00_enzyme_datasets_processed/")
subs_JTVAE_file="phosphatase_JTVAE_features.p"
#---------- Morgan
data_folder = Path("X_DataProcessing/")
subs_Morgan1024_file = "X00_phosphatase_Morgan1024_features.p"
subs_Morgan2048_file = "X00_phosphatase_Morgan2048_features.p"
#--------------------------------------------------#
# Data Split Methods
split_type = 2 # 0, 1, 2, 3
# split_type = 0, train/test/split completely randomly selected
# split_type = 1, train/test/split looks at different seq-subs pairs
# split_type = 2, train/test/split looks at different seqs
# split_type = 3, train/test/split looks at different subs
custom_split = True
customized_idx_file = "X02_customized_idx_selected.p"
#--------------------------------------------------#
# Prediction NN settings
epoch_num=200
batch_size=256
learning_rate=0.0001
NN_type_list=["Reg", "Clf"]
NN_type=NN_type_list[1]
#--------------------------------------------------#
hid_1=1024
hid_2=1024
#--------------------------------------------------#
if log_value==True:
    screen_bool = True
if NN_type=="Clf": ##### !!!!! If value is "Clf", log_value will be changed
    screen_bool = False # Actually Useless
    log_value==False
#--------------------------------------------------#
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
print("learning_rate: ", learning_rate)
print("batch_size: ", batch_size)
print("NN_type: ", NN_type)
#--------------------------------------------------#
print("hid_1: ", hid_1)
print("hid_2: ", hid_2)
#########################################################################################################
#########################################################################################################
# Get Input files
# Get Sequence Embeddings from X03 pickles.
with open( data_folder / embedding_file, 'rb') as seqs_embeddings:
    seqs_embeddings_pkl = pickle.load(seqs_embeddings)
X_seqs_embeddings_list = seqs_embeddings_pkl['seq_embeddings'] 
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
    print(query_str1)
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
#print(get_full_ecfp(smiles_str="C([C@H](COP(=O)(O)O)O)O",ecfp_type="ECFP6",iteration_number=3))
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
#====================================================================================================#
# Generate ECFP encodings here !
#all_ecfps, all_smiles_ecfps_dict = generate_all_smiles_ecfps_list_dict(all_smiles_list, ecfp_type="ECFP6", iteration_number=3)
#pickle.dump(all_ecfps, open(data_folder / i_o_put_file_1,"wb") )
#pickle.dump(all_smiles_ecfps_dict, open(data_folder / i_o_put_file_2,"wb"))
#====================================================================================================#
# Load already generated ECFP encodings !
with open( data_folder / i_o_put_file_1, 'rb') as all_ecfps:
    all_ecfps = pickle.load(all_ecfps)
with open( data_folder / i_o_put_file_2, 'rb') as all_smiles_ecfps_dict:
    all_smiles_ecfps_dict = pickle.load(all_smiles_ecfps_dict)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def Get_represented_X_y_data(X_seqs_embeddings_list, subs_properties_list, screen_bool, classification_threshold_type):
    # subs_properties_list: [[one_substrate, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES], [], [], ...[] ]
    X_seqs_embeddings = []
    X_subs_representations=[]
    y_data = []
    seqs_subs_idx_book=[]
    print("len(X_seqs_embeddings_list): ", len(X_seqs_embeddings_list))
    for i in range(len(subs_properties_list)):
        for j in range(len(X_seqs_embeddings_list)):
            X_smiles_rep = subs_properties_list[i][-1] # list of SMILES
            X_one_seqs_embedding = X_seqs_embeddings_list[j]
            if not (screen_bool==True and subs_properties_list[i][classification_threshold_type][j]==False):
                # classification_threshold_type ----> # 2: Threshold = 1e-5 | 3: Threshold = 1e-2 (from Original CSV File)
                # subs_properties_list[i] ----> [one_substrate, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES]
                # subs_properties_list[i][classification_threshold_type] ----> y_prpty_cls OR y_prpty_cls2
                # subs_properties_list[i][classification_threshold_type][j] ----> y_prpty_cls[j] OR y_prpty_cls2[j]
                # subs_properties_list[i][1][j] ----> y_prpty_reg[j]
                X_seqs_embeddings.append(X_one_seqs_embedding)
                X_subs_representations.append(X_smiles_rep)
                y_data.append(subs_properties_list[i][1][j])
                seqs_subs_idx_book.append([j, i]) # [seqs_idx, subs_idx]
    return X_seqs_embeddings, X_subs_representations, y_data, seqs_subs_idx_book
#====================================================================================================#
def Get_represented_X_y_data_clf(X_seqs_embeddings_list, subs_properties_list, classification_threshold_type):
    # subs_properties_list: [[one_substrate, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES], [], [], ...[] ]
    X_seqs_embeddings = []
    X_subs_representations=[]
    y_data = []
    seqs_subs_idx_book=[]
    print("len(X_seqs_embeddings_list): ", len(X_seqs_embeddings_list))
    for i in range(len(subs_properties_list)):
        for j in range(len(X_seqs_embeddings_list)):
            X_smiles_rep = subs_properties_list[i][-1] # list of SMILES
            X_one_seqs_embedding = X_seqs_embeddings_list[j]
            # classification_threshold_type ----> # 2: Threshold = 1e-5 | 3: Threshold = 1e-2 (from Original CSV File)
            # subs_properties_list[i] ----> [one_substrate, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES]
            # subs_properties_list[i][classification_threshold_type] ----> y_prpty_cls OR y_prpty_cls2
            # subs_properties_list[i][classification_threshold_type][j] ----> y_prpty_cls[j] OR y_prpty_cls2[j]
            X_seqs_embeddings.append(X_one_seqs_embedding)
            X_subs_representations.append(X_smiles_rep)
            y_data.append(subs_properties_list[i][classification_threshold_type][j])
            seqs_subs_idx_book.append([j, i]) # [seqs_idx, subs_idx]
    return X_seqs_embeddings, X_subs_representations, y_data, seqs_subs_idx_book
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
    X_seqs_embeddings, X_subs_representations, y_data, seqs_subs_idx_book = Get_represented_X_y_data(X_seqs_embeddings_list, subs_properties_list, screen_bool, classification_threshold_type)
if NN_type=="Clf":
    X_seqs_embeddings, X_subs_representations, y_data, seqs_subs_idx_book = Get_represented_X_y_data_clf(X_seqs_embeddings_list, subs_properties_list, classification_threshold_type)

#====================================================================================================#
subs_encodings_list = ["ECFP2", "ECFP4", "ECFP6", "JTVAE", "MorganFP", "ECFP8"]
if subs_encodings == "ECFP2" or subs_encodings == "ECFP4"  or subs_encodings == "ECFP6" or subs_encodings == "ECFP8" :
    X_subs_encodings = Get_ECFPs_encoding(X_subs_representations, all_ecfps, all_smiles_ecfps_dict)
if subs_encodings == "JTVAE":
    X_subs_encodings = Get_JTVAE_encoding(X_subs_representations, subs_SMILES_JTVAE_dict)
if subs_encodings == "MorganFP":
    X_subs_encodings = Get_Morgan_encoding(X_subs_representations, subs_SMILES_Morgan1024_dict)
#====================================================================================================#
#save_dict=dict([])
#save_dict["X_seqs_embeddings"] = X_seqs_embeddings
#save_dict["X_subs_encodings"] = X_subs_encodings
#save_dict["y_data"] = y_data
#pickle.dump( save_dict , open( results_folder / output_file_3, "wb" ) )
#print("Done getting X_seqs_embeddings, X_subs_encodings and y_data!")
print("len(X_seqs_embeddings): ", len(X_seqs_embeddings), ", len(X_subs_encodings): ", len(X_subs_encodings), ", len(y_data): ", len(y_data) )
#====================================================================================================#
# Get size of some interests
X_seqs_embeddings_dim = X_seqs_embeddings[0].shape[0]
X_subs_encodings_dim = len(X_subs_encodings[0])
X_seqs_num = len(X_seqs_embeddings_list)
X_subs_num = len(subs_properties_list)
print("seqs,subs dimensions: ", X_seqs_embeddings_dim, ", ", X_subs_encodings_dim)
print("seqs,subs counts: ", X_seqs_num, ", ", X_subs_num)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def split_idx(X_num, train_split, test_split, random_state=42):
    #X_seqs_idx=y_seqs_idx=list(range(len(X_seqs_embeddings_list)))
    #X_subs_idx=y_subs_idx=list(range(len(subs_properties_list)))
    X_idx = y_idx = list(range(X_num))
    X_tr_idx, X_ts_idx, y_tr_idx, y_ts_idx = train_test_split(X_idx, y_idx, test_size=(1-train_split), random_state=42)
    X_va_idx, X_ts_idx, y_va_idx, y_ts_idx = train_test_split(X_ts_idx, y_ts_idx, test_size = (test_split/(1.0-train_split)) , random_state=42)
    return X_tr_idx, X_ts_idx, X_va_idx
#====================================================================================================#
tr_idx_seqs, ts_idx_seqs, va_idx_seqs = split_idx(X_seqs_num, train_split=0.8, test_split=0.1, random_state=42)
tr_idx_subs, ts_idx_subs, va_idx_subs = split_idx(X_subs_num, train_split=0.8, test_split=0.1, random_state=42)
#====================================================================================================#
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
#########################################################################################################
#########################################################################################################
def Get_X_y_data_selected(X_idx, X_seqs_embeddings, X_subs_encodings, y_data):
    X_data_selected=[]
    y_data_selected=[]
    for idx in X_idx:
        Xi_extended=list(np.concatenate(  (X_seqs_embeddings[idx], X_subs_encodings[idx])  ))
        X_data_selected.append(Xi_extended)
        y_data_selected.append(y_data[idx])
    X_data_selected = np.array(X_data_selected)
    y_data_selected = np.array(y_data_selected)
    if log_value==True:
        y_data_selected=np.log10(y_data_selected)
    return X_data_selected, y_data_selected
X_train, y_train = Get_X_y_data_selected(X_train_idx, X_seqs_embeddings, X_subs_encodings, y_data)
X_test , y_test  = Get_X_y_data_selected(X_test_idx, X_seqs_embeddings, X_subs_encodings, y_data)
X_valid, y_valid = Get_X_y_data_selected(X_valid_idx, X_seqs_embeddings, X_subs_encodings, y_data)

#print("Done getting X_data and y_data!")
print("X_train_dimension: ", X_train.shape, "y_train_dimension: ", y_train.shape )
print("X_test_dimension: ", X_test.shape, "y_test_dimension: ", y_test.shape )
print("X_valid_dimension: ", X_valid.shape, "y_valid_dimension: ", y_valid.shape )
#########################################################################################################
#########################################################################################################
NN_input_dim=X_train.shape[1]
print("NN_input_dim: ", NN_input_dim)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Y04B Learning AND Prediction
class LoaderClass(data.Dataset):
    def __init__(self, embeding, label):
        super(LoaderClass, self).__init__()
        self.embedding = embeding
        self.label = label
    def __len__(self):
        return self.embedding.shape[0]
    def __getitem__(self, idx):
        return self.embedding[idx], self.label[idx]
#====================================================================================================#
train_loader = data.DataLoader(LoaderClass(X_train,y_train),batch_size,True)
valid_loader = data.DataLoader(LoaderClass(X_valid,y_valid),batch_size,False)
test_loader  = data.DataLoader(LoaderClass(X_test,y_test),batch_size,False)
#########################################################################################################
#########################################################################################################
class MLP2(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_1: int,
                 hid_2: int                 
                 ):
        super(MLP2,self).__init__()
        self.fc1 = weight_norm(nn.Linear(in_dim,hid_1),dim=None) 
        self.dropout1 = nn.Dropout(p=0.) 
        self.fc2 = weight_norm(nn.Linear(hid_1,hid_2),dim=None)
        self.fc3 = weight_norm(nn.Linear(hid_2,1),dim=None)

    def forward(self,input):
        output = nn.functional.leaky_relu(self.fc1(input))
        output = self.dropout1(output)
        output = nn.functional.leaky_relu(self.fc2(output))
        output = self.fc3(output)
        return output
#########################################################################################################
#########################################################################################################
class MLP2_clf(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_2: int                 
                 ):
        super(MLP2_clf,self).__init__()
        self.fc1 = weight_norm(nn.Linear(in_dim,hid_2),dim=None)
        self.dropout1 = nn.Dropout(p=0.0)
        self.fc2 = weight_norm(nn.Linear(hid_2,1),dim=None)
    #--------------------------------------------------#
    def forward(self, input):
        output = nn.functional.leaky_relu(self.fc1(input))
        output = self.dropout1(output)
        output = torch.sigmoid(self.fc2(output))
        return output
#########################################################################################################
#########################################################################################################
if NN_type == "Reg":
    #====================================================================================================#
    model = MLP2(in_dim=NN_input_dim, hid_1=hid_1, hid_2=hid_2)
    model.double()
    model.cuda()
    print("#"*50)
    print(model.eval())
    print("#"*50)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    criterion = nn.MSELoss()
    #====================================================================================================#
    for epoch in range(epoch_num): 
        model.train()
        for one_seqsubs_ppt_pair in train_loader:
            #print(one_seqsubs_ppt_pair)
            input, target = one_seqsubs_ppt_pair
            input, target = input.double().cuda(), target.cuda()
            output = model(input)
            loss = criterion(output,target.view(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #--------------------------------------------------#
        model.eval()
        y_pred_valid = []
        y_real_valid = []
        for one_seqsubs_ppt_pair in valid_loader:
            input,target = one_seqsubs_ppt_pair
            input = input.double().cuda()
            output = model(input)
            output = output.cpu().detach().numpy().reshape(-1)
            target = target.numpy()
            y_pred_valid.append(output)
            y_real_valid.append(target)
        y_pred_valid = np.concatenate(y_pred_valid)
        y_real_valid = np.concatenate(y_real_valid)
        slope, intercept, r_value_va, p_value, std_err = scipy.stats.linregress(y_pred_valid, y_real_valid)
        #--------------------------------------------------#
        y_pred = []
        y_real = []
        for one_seqsubs_ppt_pair in test_loader:
            input,target = one_seqsubs_ppt_pair
            input = input.double().cuda()
            output = model(input)
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
        print("epoch: {} | vali_r_value: {} | loss: {} | test_r_value: {} ".format( str((epoch+1)+1000).replace("1","",1) , np.round(r_value_va,4), loss, np.round(r_value,4)))
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
if NN_type == "Clf":
    #====================================================================================================#
    model = MLP2_clf(in_dim=NN_input_dim, hid_2=hid_2)
    model.double()
    model.cuda()
    print("#"*50)
    print(model.eval())
    print("#"*50)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    criterion = nn.BCELoss()
    #====================================================================================================#
    for epoch in range(epoch_num):
        model.train()
        for one_seqsubs_ppt_pair in train_loader:
            input, target = one_seqsubs_ppt_pair
            input, target = input.double().cuda(), target.double().cuda()
            output = model(input)
            loss = criterion(output,target.view(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #--------------------------------------------------#
        model.eval()
        y_pred_valid = []
        y_real_valid = []
        for one_seqsubs_ppt_pair in valid_loader:
            input,target = one_seqsubs_ppt_pair
            input = input.double().cuda()
            output = model(input)
            output = output.cpu().detach().numpy().reshape(-1)
            target = target.numpy()
            y_pred_valid.append(output)
            y_real_valid.append(target)
        y_pred_valid = np.concatenate(y_pred_valid)
        y_real_valid = np.concatenate(y_real_valid)
        vali_AUC = roc_auc_score(y_real_valid,y_pred_valid)
        #====================================================================================================#
        y_pred = []
        y_real = []
        for one_seqsubs_ppt_pair in test_loader:
            input,target = one_seqsubs_ppt_pair
            input = input.double().cuda()
            output = model(input)
            output = output.cpu().detach().numpy().reshape(-1)
            target = target.numpy()
            y_pred.append(output)
            y_real.append(target)
        y_pred = np.concatenate(y_pred)
        y_real = np.concatenate(y_real)
        test_AUC = roc_auc_score(y_real,y_pred)  
        print("epoch: {} | loss: {} | vali_AUC: {} | test_AUC: {}".format(str((epoch+1)+1000).replace("1","",1), loss, np.round(vali_AUC,3), np.round(test_AUC,3)))
        #====================================================================================================#
        if ((epoch+1) % 5) == 0:
            fpr,tpr,_ = roc_curve(y_real,y_pred)
            roc_auc = auc(fpr,tpr)
            #--------------------------------------------------#
            lw = 2
            fig = plt.figure(figsize=(6,6))
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlabel('False Positive Rate',fontsize=18 ,fontweight='bold')
            plt.ylabel('True Positive Rate',fontsize=18 ,fontweight='bold')
            plt.title('ROC-' + "AUC: " + str(np.round(test_AUC,3)) + ", Epoch: " + str(epoch+1) ,fontsize=18 ,fontweight='bold')
            plt.legend(loc="lower right")
            #plt.axis('equal')
            plt.ylim([-0.05, 1.05])
            plt.xlim([-0.05, 1.05])
            fig.savefig(    results_sub_folder / (output_file_header + "_ROC-AUC_" + "epoch_" + str(epoch+1))   ) 
        #====================================================================================================#
            lr_precision, lr_recall, _ = precision_recall_curve(y_real,y_pred)
            no_skill=len(y_real[y_real==1])/len(y_real)
            fig = plt.figure(figsize=(6,6))
            AUPRC= np.round(metrics.auc(lr_recall, lr_precision),3)

            plt.plot( [0,1], [no_skill,no_skill], linestyle='--',label='No Skill')
            plt.plot(lr_recall, lr_precision,marker='.', label='Classifier')
            plt.xlabel('Recall',fontsize=18 ,fontweight='bold')
            plt.ylabel('Precision',fontsize=18 ,fontweight='bold')
            plt.title("AU-PRC: " + str(AUPRC) + ", Epoch: " + str(epoch+1) ,fontsize=18 ,fontweight='bold')
            plt.legend(loc="lower right")
            plt.ylim([no_skill - 0.1, 1.05])
            plt.xlim([-0.05, 1.05])
            fig.savefig(    results_sub_folder / (output_file_header + "_AU-PRC_" + "epoch_" + str(epoch+1))    ) 

#########################################################################################################
#########################################################################################################
print(Step_code, " Done!")
sys.stdout = orig_stdout
f.close()
print(Step_code, " Done!")

#====================================================================================================#
class MLP3(nn.Module):
    def __init__(self):
        super(MLP3,self).__init__()
        self.fc1 = weight_norm(nn.Linear(1024,1536),dim=None) 
        self.dropout1 = nn.Dropout(p=0.05) 
        self.fc2 = weight_norm(nn.Linear(1536,1024),dim=None)
        self.dropout2 = nn.Dropout(p=0.05) 
        self.fc3 = weight_norm(nn.Linear(1024,1024),dim=None)
        self.fc4 = weight_norm(nn.Linear(1024,1),dim=None)

    def forward(self,input):
        output = nn.functional.leaky_relu(self.fc1(input))
        output = self.dropout1(output)
        output = nn.functional.leaky_relu(self.fc2(output))
        output = self.dropout2(output)
        output = nn.functional.leaky_relu(self.fc3(output))
        output = self.fc4(output)
        return output
#!/usr/bin/env python
# coding: utf-8
#########################################################################################################
#########################################################################################################
# Microsoft VS header 
import os, os.path
#import winsound
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
import pickle
import argparse
import numpy as ny
import pandas as pd
from Bio import SeqIO
#--------------------------------------------------#
from pathlib import Path
from torch import nn
from torch.utils import data as data
import torch.optim as optim
#--------------------------------------------------#
from tape import datasets
from tape.datasets import *
#--------------------------------------------------#
from ModifiedModels import *
#====================================================================================================#
from tape import ProteinBertModel, TAPETokenizer, ProteinBertForMaskedLM
model = ProteinBertForMaskedLM.from_pretrained('bert-base')
tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model
#from tape import UniRepModel, UniRepForLM
#########################################################################################################
#########################################################################################################
# Args
Step_code = "X02_"
output_folder = Path("X_DataProcessing/")
input_seq_fasta_file = "X00_phosphatase.fasta"
#--------------------------------------------------#
data_folder = Path("X_DataProcessing/X00_enzyme_datasets_processed/")
#input_seq_fasta_file_1 = "X00_DB_clu_rep_218.fasta"
input_seq_fasta_file_1 = "X00_DB_clu_rep_204.fasta"

#--------------------------------------------------#

output_file_name_header = Step_code + "customized_idx_"
#--------------------------------------------------#
seed=42
#--------------------------------------------------#

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
input_file_all_seq = output_folder / input_seq_fasta_file #data path (fasta) 
all_seq_list=[]
for seq_record in SeqIO.parse(input_file_all_seq, "fasta"):
    seq = str(seq_record.seq)
    all_seq_list.append(seq)
    #print(seq)


input_file_selected = data_folder / input_seq_fasta_file_1 #data path (fasta)
print("input_file_selected: ", input_file_selected)
selected_seq_list=[]
for seq_record in SeqIO.parse(input_file_selected, "fasta"):
    seq = str(seq_record.seq)
    selected_seq_list.append(seq)
print("Index count for test/validation set: ", len(selected_seq_list))

selected_idx_list=[]
for one_selected_seq in selected_seq_list:
    if one_selected_seq in all_seq_list:
        selected_idx_list.append(all_seq_list.index(one_selected_seq))


pickle.dump( selected_idx_list, open( output_folder / (output_file_name_header + "selected.p"), "wb" ) )









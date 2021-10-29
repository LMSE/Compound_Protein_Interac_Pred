#!/usr/bin/env python
# coding: utf-8
#########################################################################################################
#########################################################################################################
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit import DataStructs
from rdkit.Chem.AtomPairs import Torsions
#########################################################################################################
#########################################################################################################
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
import numpy as np
import pickle
import pandas as pd
#--------------------------------------------------#
from chemconvert import *
from chemconvert import unique_canonical_smiles_zx as unis
#--------------------------------------------------#
from pathlib import Path
#########################################################################################################
#########################################################################################################
# Args
Step_code = "X00_"
data_folder = Path("X_DataProcessing/X00_enzyme_datasets_processed/")
data_file = "phosphatase_chiral.csv"
data_file_binary = "phosphatase_chiral_binary.csv" # y_prpty_cls_threshold = around 1e-2
smiles_file = "phosphatase_smiles.dat"
#--------------------------------------------------#
max_seq_len=300
#--------------------------------------------------#
output_folder = Path("X_DataProcessing/")
output_file_1 = "X00_substrates_properties_list.p"
output_file_2 = "X00_phosphatase.fasta"
output_file_3 = "X00_phosphatase_Morgan1024_features.p"
output_file_4 = "X00_phosphatase_Morgan2048_features.p"
#====================================================================================================#
subs_df = pd.read_csv(data_folder / data_file, index_col=0, header=0)
subs_df_bi = pd.read_csv(data_folder / data_file_binary, index_col=0, header=0)
#====================================================================================================#
# 
subs_df["seq_length"]=subs_df.SEQ.str.len()
subs_df = subs_df[subs_df.seq_length<=max_seq_len]
subs_df.reset_index(drop=True, inplace=True)

subs_df_bi["seq_length"]=subs_df_bi.SEQ.str.len()
subs_df_bi = subs_df_bi[subs_df_bi.seq_length<=max_seq_len]
subs_df_bi.reset_index(drop=True, inplace=True)

#====================================================================================================#
#Once the display.max_rows is exceeded, 
#the display.min_rows options determines 
#how many rows are shown in the truncated repr.
with pd.option_context('display.max_rows', 20, 
                       'display.min_rows', 20, 
                       'display.max_columns', None, 
                       #"display.max_colwidth", None,
                       "display.width", None,
                       "expand_frame_repr", True,
                       "max_seq_items", None,):  # more options can be specified
    print(subs_df)
print()
#====================================================================================================#
subs_df_row_num = subs_df.shape[0]-1
y_prpty_cls_threshold = 1e-5 # Used for type II screening
#====================================================================================================#
# Write a fasta file including all sequences
# Also get a seq_list including all sequences
with open(output_folder / output_file_2, 'w') as f:
    count_x=0
    seq_list=[]
    for i in range(subs_df_row_num):
        if subs_df.loc[i,"SEQ"] not in seq_list and len(subs_df.loc[i,"SEQ"])<=max_seq_len:
            seq_list.append(subs_df.loc[i,"SEQ"])
            count_x+=1
            f.write(">seq"+str(count_x)+"\n")
            f.write(subs_df.loc[i,"SEQ"].upper()+"\n")
print("number of seqs: ", len(seq_list))
#====================================================================================================#
# Also get a substrate_list including all substrates (substrates here use SMILES representation)
subs_list=[] # Non-unique SMILES
subs_smiles_list=[] # Unique SMILES
for i in range(subs_df_row_num):
    if subs_df.loc[i,"SUBSTRATES"] not in subs_list:
        one_substrate_smiles=unis(subs_df.loc[i,"SUBSTRATES"])
        subs_list.append(subs_df.loc[i,"SUBSTRATES"])
        subs_smiles_list.append(one_substrate_smiles)
print("number of subs: ", len(subs_smiles_list)) # actually SMILES



from rdkit.Chem import Draw
fig_folder = Path("X_DataProcessing//X00_figs/")
size = (520, 520) 
count_x=0
for one_sub in subs_list:
    count_x+=1
    m = Chem.MolFromSmiles(one_sub)
    img = Draw.MolsToGridImage([m, ],molsPerRow=1,subImgSize=(200,200)) 
    img.save(fig_folder / (str(count_x)+ '.png'))    


#====================================================================================================#
# Morgan Function #1
subs_SMILES_MorganFP1024_dict=dict([])
for one_smiles in subs_smiles_list:
    rd_mol = Chem.MolFromSmiles(one_smiles)
    MorganFP = AllChem.GetMorganFingerprintAsBitVect(rd_mol, 2, nBits=1024)
    MorganFP_features = np.array(MorganFP)
    subs_SMILES_MorganFP1024_dict[one_smiles]=MorganFP_features
pickle.dump(subs_SMILES_MorganFP1024_dict, open(output_folder / output_file_3,"wb"))
#====================================================================================================#
# Morgan Function #2
subs_SMILES_MorganFP2048_dict=dict([])
for one_smiles in subs_smiles_list:
    rd_mol = Chem.MolFromSmiles(one_smiles)
    MorganFP = AllChem.GetMorganFingerprintAsBitVect(rd_mol, 2, nBits=2048)
    MorganFP_features = np.array(MorganFP)
    subs_SMILES_MorganFP2048_dict[one_smiles]=MorganFP_features
pickle.dump(subs_SMILES_MorganFP2048_dict, open(output_folder / output_file_4,"wb"))
#====================================================================================================#
# Morgan Function #3
def ECFP_from_SMILES(smiles, radius=2, bit_len=1024, scaffold=0, index=None): # Not useful here !
    fps = np.zeros((len(smiles), bit_len))
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        arr = np.zeros((1,))
        try:
            if scaffold == 1:
                mol = MurckoScaffold.GetScaffoldForMol(mol)
            elif scaffold == 2:
                mol = MurckoScaffold.MakeScaffoldGeneric(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps[i, :] = arr
        except:
            print(smile)
            fps[i, :] = [0] * bit_len
    return pd.DataFrame(fps, index=(smiles if index is None else index)) 
#====================================================================================================#
# Morgan Function #4
def morgan_fingerprint(smiles: str, radius: int = 2, num_bits: int = 1024, use_counts: bool = False) -> np.ndarray:
    """
    Generates a morgan fingerprint for a smiles string.
    :param smiles: A smiles string for a molecule.
    :param radius: The radius of the fingerprint.
    :param num_bits: The number of bits to use in the fingerprint.
    :param use_counts: Whether to use counts or just a bit vector for the fingerprint
    :return: A 1-D numpy array containing the morgan fingerprint.
    """
    if type(smiles) == str:
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = smiles
    if use_counts:
        fp_vect = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    else:
        fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    fp = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_vect, fp)
    return fp 
#====================================================================================================#


#====================================================================================================#
# Use "phosphatase_smiles.dat" to obtain a dict for substrates names and SMILES
smiles_substrate_dict=dict([])
smiles_list=[]
with open(data_folder / smiles_file) as f:
    lines = f.readlines()
    for one_line in lines:
        one_line=one_line.replace("\n","")
        one_pair_list = one_line.split("\t") # Get [substrate, SMILES]
        smiles_substrate_dict[unis(one_pair_list[1])]=one_pair_list[0]
        smiles_list.append(unis(one_pair_list[1]))
print("number of smiles identified: ", len(smiles_substrate_dict))
#print("smiles_substrate_dict: ", smiles_substrate_dict)

#====================================================================================================#
substrates_properties_list=[]
count_unknown=0


for one_subs_smiles in subs_smiles_list:
    y_prpty_reg=np.array(subs_df.loc[subs_df['SUBSTRATES'] == subs_list[subs_smiles_list.index(one_subs_smiles)]]["Conversion"])
    #print(subs_df.loc[subs_df['SUBSTRATES'] == one_subs_smiles])
    #--------------------------------------------------#
    y_prpty_cls_2=np.array(subs_df_bi.loc[subs_df['SUBSTRATES'] == subs_list[subs_smiles_list.index(one_subs_smiles)]]["Conversion"])
    #--------------------------------------------------#
    y_list_0_1=[]
    for y_value in y_prpty_reg:
        y_list_0_1.append(1 if y_value>y_prpty_cls_threshold else 0)
    y_prpty_cls=np.array(y_list_0_1)
    #--------------------------------------------------#
    if one_subs_smiles in smiles_substrate_dict.keys():
        substrates_properties=[smiles_substrate_dict[one_subs_smiles],y_prpty_reg,y_prpty_cls,y_prpty_cls_2,one_subs_smiles]
    else:
        count_unknown+=1
        substrates_properties=["substrate_"+str(count_unknown),y_prpty_reg,y_prpty_cls,y_prpty_cls_2,one_subs_smiles]
    #print(substrates_properties)
    substrates_properties_list.append(substrates_properties)
#====================================================================================================#
#====================================================================================================#
# substrates_properties
# [ SUBSTRATE_name, Y_Property_value, Y_Property_Class_#1, Y_Property_Class_#2, SMILES_str ]
# Y_Property_Class_#1: threshold 1e-5 (y_prpty_cls_threshold)
# Y_Property_Class_#2: threshold 1e-2 (provided by LEARNING PROTEIN SEQUENCE EMBEDDINGS USING INFORMATION FROM STRUCTURE)
pickle.dump( substrates_properties_list, open( output_folder / output_file_1, "wb" ) )
print(Step_code + " Done!")

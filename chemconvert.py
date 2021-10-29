#!/usr/bin/python
#This file contains a bunch of functions independent of each other
#####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#####
#####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#####
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit import DataStructs
from rdkit.Chem.AtomPairs import Torsions

import os, os.path
from sys import platform
if os.name == 'nt' or platform == 'win32':
    os.chdir(os.path.dirname(__file__))

import os
import sys
from sys import platform

#from chemfuncs import *

global bkgd_cmpd_list; bkgd_cmpd_list=['O','[H]O[H]', 'O=P(O)(O)O', 'O=C=O', 'N']
def bkgd_cmpd_list_func():
    return bkgd_cmpd_list

#####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#####
#####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#####

global CoA_cmpd_list; CoA_cmpd_list=["Acetyl-CoA","Malonyl-CoA", "Succinyl-CoA"] # For test only
def CoA_cmpd_list_func():
    #return CoA_cmpd_list
    return []

global CoA_cmpd_dict; CoA_cmpd_dict={ "CC(=O)SCCNC(=O)CCNC(=O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O": "Acetyl-CoA",
                                      "CC(C)(COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O)C(O)C(=O)NCCC(=O)NCCSC(=O)CC(=O)O" : "Malonyl-CoA",
                                      "CC(C)(COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O)C(O)C(=O)NCCC(=O)NCCSC(=O)CCC(=O)O": "Succinyl-CoA", 
                                     
                                     } # For test only
def CoA_cmpd_list_func():
    #return CoA_cmpd_list
    return []
#####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#####
#####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#####

def unique_input_smiles_zx(smiles_x):
    r_fmt_cmpd_x=Chem.MolFromSmiles(smiles_x)
    try:
        r_fmt_cmpd_x_unique=Chem.MolFromSmarts(Chem.MolToSmarts(r_fmt_cmpd_x))
        unique_smiles=Chem.MolToSmiles(r_fmt_cmpd_x_unique)
    except Exception:
        print("!!!!! Problematic input SMILES string !!!!!")
        unique_smiles=smiles_x
    return unique_smiles

def unique_canonical_smiles_zx(smiles_x):
    molecule=Chem.MolFromSmiles(smiles_x)
    try:
        unique_smiles=Chem.MolToSmiles(molecule)
    except Exception:
        print("problematic")
        unique_smiles=smiles_x
    return unique_smiles

def unique_canonical_smiles_list_zx(list_x):
    new_list=[]
    for one_smiles in list_x:
        new_list.append(unique_canonical_smiles_zx(one_smiles))
    return new_list

def MolFromSmiles_zx(smiles_x, bad_ss_dict):
    r_fmt_cmpd_x=Chem.MolFromSmiles(smiles_x)
    try: 
        Chem.MolToSmiles(r_fmt_cmpd_x)
    except:
        r_fmt_cmpd_x=Chem.MolFromSmarts(bad_ss_dict[smiles_x])
        #r_fmt_cmpd_x=Chem.MolFromSmarts("O")
    return r_fmt_cmpd_x

def MolToSmiles_zx(r_fmt_cmpd_x, bad_ss_dict):
    # !!!!!: Converting to smarts and back to rdkit-format to ensure uniqueness before converting to smiles
    r_fmt_cmpd_x_unique=Chem.MolFromSmarts(Chem.MolToSmarts(r_fmt_cmpd_x))
    smiles_x=Chem.MolToSmiles(r_fmt_cmpd_x_unique)
    try:
        Chem.MolToSmiles(Chem.MolFromSmiles(smiles_x))
    except Exception:
        bad_ss_dict[smiles_x]=Chem.MolToSmarts(r_fmt_cmpd_x_unique)
    return smiles_x

#####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#####
#####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#####
# Similarity
def similarity_metric_select(fp_a,fp_b,parameter_1,parameter=2):
    if (parameter_1=="top"):
        similarity=DataStructs.FingerprintSimilarity(fp_a,fp_b)
    elif (parameter_1=="MACCS"):
        similarity=DataStructs.FingerprintSimilarity(fp_a,fp_b)
    elif (parameter_1=="atom_pairs"):
        similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
    elif (parameter_1=="vec_pairs"):
        similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
    elif (parameter_1=="torsions"):
        similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
    elif (parameter_1=="FCFP"):
        similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
    else: # ECFP
        similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
    return similarity
# ==================================================================================== #
def generate_fingerprint(smiles_a, parameter_1, parameter_2=2):
    try: 
        cmpd_a=Chem.MolFromSmiles(str(smiles_a))
        if (parameter_1=="top"):
            fp_a=FingerprintMols.FingerprintMol(cmpd_a)
        elif (parameter_1=="MACCS"):
            fp_a=MACCSkeys.GenMACCSKeys(cmpd_a)
        elif (parameter_1=="atom_pairs"):
            fp_a=Pairs.GetAtomPairFingerprint(cmpd_a)
        elif (parameter_1=="vec_pairs"):
            fp_a=Pairs.GetAtomPairFingerprintAsBitVect(cmpd_a)
        elif (parameter_1=="torsions"):
            fp_a=Torsions.GetTopologicalTorsionFingerprintAsIntVect(cmpd_a)
        elif (parameter_1=="FCFP"):
            fp_a=AllChem.GetMorganFingerprint(cmpd_a,parameter_2,useFeatures=True)
        else: #ECFP
            fp_a=AllChem.GetMorganFingerprint(cmpd_a,parameter_2)
    except Exception:
        print("ERROR: generate fingerprint")
        cmpd_a=Chem.MolFromSmiles(str('O'))
        if (parameter_1=="top"):
            fp_a=FingerprintMols.FingerprintMol(cmpd_a)
        elif (parameter_1=="MACCS"):
            fp_a=MACCSkeys.GenMACCSKeys(cmpd_a)
        elif (parameter_1=="atom_pairs"):
            fp_a=Pairs.GetAtomPairFingerprint(cmpd_a)
        elif (parameter_1=="vec_pairs"):
            fp_a=Pairs.GetAtomPairFingerprintAsBitVect(cmpd_a)
        elif (parameter_1=="torsions"):
            fp_a=Torsions.GetTopologicalTorsionFingerprintAsIntVect(cmpd_a)
        elif (parameter_1=="FCFP"):
            fp_a=AllChem.GetMorganFingerprint(cmpd_a,parameter_2,useFeatures=True)
        else: #ECFP
            fp_a=AllChem.GetMorganFingerprint(cmpd_a,parameter_2)
    return fp_a
# ==================================================================================== #
def similarity_score(smiles_a, smiles_b, parameter_1="ECFP", parameter_2=2): # Return the similarity of two compounds
    try:
        # parameter_1 is similarity metric selected
        cmpd_a=Chem.MolFromSmiles(str(smiles_a))
        cmpd_b=Chem.MolFromSmiles(str(smiles_b))
        if (parameter_1=="top"):
            fp_a=FingerprintMols.FingerprintMol(cmpd_a)
            fp_b=FingerprintMols.FingerprintMol(cmpd_b)  
            similarity=DataStructs.FingerprintSimilarity(fp_a,fp_b)
        elif (parameter_1=="MACCS"):
            fp_a=MACCSkeys.GenMACCSKeys(cmpd_a)
            fp_b=MACCSkeys.GenMACCSKeys(cmpd_b)
            similarity=DataStructs.FingerprintSimilarity(fp_a,fp_b)
        elif (parameter_1=="atom_pairs"):
            fp_a=Pairs.GetAtomPairFingerprint(cmpd_a)
            fp_b=Pairs.GetAtomPairFingerprint(cmpd_b)
            similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
        elif (parameter_1=="vec_pairs"):
            fp_a=Pairs.GetAtomPairFingerprintAsBitVect(cmpd_a)
            fp_b=Pairs.GetAtomPairFingerprintAsBitVect(cmpd_b)
            similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
        elif (parameter_1=="torsions"):
            fp_a=Torsions.GetTopologicalTorsionFingerprintAsIntVect(cmpd_a)
            fp_b=Torsions.GetTopologicalTorsionFingerprintAsIntVect(cmpd_b)
            similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
        elif (parameter_1=="FCFP"):
            fp_a=AllChem.GetMorganFingerprint(cmpd_a,parameter_2,useFeatures=True)
            fp_b=AllChem.GetMorganFingerprint(cmpd_b,parameter_2,useFeatures=True)
            similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
        else: #ECFP
            fp_a=AllChem.GetMorganFingerprint(cmpd_a,parameter_2)
            fp_b=AllChem.GetMorganFingerprint(cmpd_b,parameter_2)
            similarity=DataStructs.DiceSimilarity(fp_a,fp_b)
    except Exception:
        if smiles_a.find("CoA")==-1 and smiles_b.find("CoA")==-1:
            similarity=0
            print("ERROR: similarity score")
        else:
            similarity=1
    return similarity
# ==================================================================================== #
def similarity_dict(list_tb_elmnt, list_tb_cmp, parameter_1, num_cmpds=10): 
    # Compare "list to be eliminated" with "list of SMILES to be compared" and return top "num cmpds" compounds
    # Inputs: a list of hashes, a list of SMILES, num compounds to return
    # Output: a list of hashes
    # 0. Initialize
    taofactors_list=[] #
    # 1. if using MNA
    if (type(parameter_1)==int):
        for hash_a in list_tb_elmnt:
            taofactor=[]
            for hash_b in list_tb_cmp:
                taofactor.append(similarity_score(hash_a, hash_b, parameter_1))
            taofactors_list.append((hash_a,max(taofactor)))
    # 2. if using SimIndex's fingerprints
    if (type(parameter_1)==str):
        # 2.1. Convert "list to be compared" to "fp2" (fp2 is a list of fingerprints)
        fp2=[]
        for smiles_a in list_tb_cmp:
            # (hash -> SMILES str -> molecule -> fingerprint)
            fp2.append(generate_fingerprint(smiles_a,parameter_1))
        # 2.2. Convert COMPOUNDS in "list to be eliminated" to "fp1" and obtain maximum taofactors for all compounds

        for smiles_a in list_tb_elmnt:
            # (hash -> SMILES str -> molecule -> fingerprint)
            fp1=generate_fingerprint(smiles_a,parameter_1)
            taofactor=[]
            for k in range(len(fp2)):
                taofactor.append(similarity_metric_select(fp1,fp2[k],parameter_1))
            taofactors_list.append((smiles_a,max(taofactor)))
    # 3. Sort the taofactors for all compounds and return top ranked compunds
    taofactors_dict={}
    for (hash,tao) in taofactors_list:
        if hash not in bkgd_cmpd_list:
            taofactors_dict[hash]=tao
    return taofactors_dict


#####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#####
#####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#####
def chemconvert_test1():
    #####
    # All of a,b and c are CoA.
    a="CC(C)(COP(=O)(O)OP(=O)(O)OCC1C(C(C(O1)N2C=NC3=C2N=CN=C3N)O)OP(=O)(O)O)C(C(=O)NCCC(=O)NCCS)O"
    b="O=C(NCCS)CCNC(=O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n2cnc1c(ncnc12)N)[C@H](O)[C@@H]3OP(=O)(O)O"
    c="CC(C)(COP(=O)(O)OP(=O)(O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N2C=NC3=C2N=CN=C3N)O)OP(=O)(O)O)[C@H](C(=O)NCCC(=O)NCCS)O"
    d="[H]N=c1c(c([H])nc(n1[H])C([H])([H])[H])C([H])([H])[n+]1cc(CC(N)C(=O)O)c2ccccc21"
    print(unique_canonical_smiles_zx(a))
    print(unique_canonical_smiles_zx(b))
    print(unique_canonical_smiles_zx(c))
    print(unique_input_smiles_zx(a))
    print(unique_input_smiles_zx(b))
    print(unique_input_smiles_zx(c))
    bad_ss_dict=dict([])
    print(MolToSmiles_zx(MolFromSmiles_zx(a,bad_ss_dict),bad_ss_dict))
    print(MolToSmiles_zx(MolFromSmiles_zx(b,bad_ss_dict),bad_ss_dict))
    print(MolToSmiles_zx(MolFromSmiles_zx(c,bad_ss_dict),bad_ss_dict))
    MolFromSmiles_zx
    print(unique_canonical_smiles_list_zx([a,b,c]))
    
    #####
    bkgd=['O','CC(C)(COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O)C(O)C(=O)NCCC(=O)NCCS','O=P(O)(O)O','O=C=O','N','CCCC(O)CC=O']
    bkgd_cmpd_list=[]
    for i in bkgd:
        bkgd_cmpd_list.append(unique_input_smiles_zx(i))
        print(unique_input_smiles_zx(i))
    print(bkgd_cmpd_list)
    print(unique_canonical_smiles_zx(unique_canonical_smiles_zx(c)))

#####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#####
#####++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#####
if __name__ == '__main__':
    chemconvert_test1()


#!/usr/bin/env python
# coding: utf-8
#########################################################################################################
#########################################################################################################
# Microsoft VS header 
import os, os.path
from sys import platform
if os.name == 'nt' or platform == 'win32':
    try:
        os.chdir(os.path.dirname(__file__))
        print("Running in Microsoft VS!")
    except:
        print("Not Running in Microsoft VS")
#====================================================================================================#
import simpleaudio as sa
def sound(frequency, seconds):
    frequency# Our played note will be 440 Hz
    seconds# Note duration of 3 seconds
    fs = 44100  # 44100 samples per second
    t = np.linspace(0, seconds, seconds * fs, False) # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
    note = np.sin(frequency * t * 2 * np.pi) # Generate a 440 Hz sine wave
    audio = note * (2**15 - 1) / np.max(np.abs(note)) # Ensure that highest value is in 16-bit range
    audio = audio.astype(np.int16) # Convert to 16-bit data
    play_obj = sa.play_buffer(audio, 1, 2, fs) # Start playback
    play_obj.wait_done() # Wait for playback to finish before exiting

#########################################################################################################
#########################################################################################################
import sys
import time
import torch
import numpy as ny
import pandas as pd
import pickle
import argparse
import requests
#--------------------------------------------------#
from torch import nn
from torch.utils import data as data
#--------------------------------------------------#
from tape import datasets
from tape import TAPETokenizer
from tape import ProteinBertForMaskedLM
#--------------------------------------------------#
from ModifiedModels import *
from pathlib import Path
#--------------------------------------------------#
from Bio import SeqIO
from tqdm.auto import tqdm
#====================================================================================================#
from transformers import BertModel, BertTokenizer
from transformers import AlbertModel, AlbertTokenizer
from transformers import ElectraTokenizer, ElectraForPreTraining, ElectraForMaskedLM, ElectraModel
from transformers import T5EncoderModel, T5Tokenizer
from transformers import XLNetModel, XLNetTokenizer
#--------------------------------------------------#
import esm


#########################################################################################################
#########################################################################################################
# Args
Step_code = "X03_"

data_folder = Path("X_DataProcessing/")
input_seq_fasta_file = "X00_phosphatase.fasta"

models_list = ["TAPE_FT", "BERT", "ALBERT", "Electra", "T5", "Xlnet", "ESM_1B", "TAPE"]
model_select = models_list[0]

pretraining_name = "X01_phosphatase_FT_epoch10_trial_training.pt"

batch_size=100
output_file_name_header = Step_code + "embedding_"

#########################################################################################################
#########################################################################################################
class LoaderClass(data.Dataset):
    def __init__(self, input_ids, attention_mask):
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    def __len__(self):
        return self.input_ids.shape[0]
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]
#====================================================================================================#
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x,target = None):
        return (x,)
#########################################################################################################
#########################################################################################################
def embedding_LM(model_select, data_folder ,input_seq_fasta_file, output_file_name_header, pretraining_name=None, batch_size=100, xlnet_mem_len=512):
    #====================================================================================================#
    assert model_select in models_list, "query model is not found, currently support TAPE_FT, bert, albert, electra, T5 and xlnet !!"
    #====================================================================================================#
    input_file = data_folder / input_seq_fasta_file #data path (fasta)
    output_file = data_folder / (output_file_name_header + model_select + ".p") #output path (pickle)
    #########################################################################################################
    #########################################################################################################
    if model_select == "TAPE_FT" or model_select == "TAPE":
        model = ProteinBertForMaskedLM.from_pretrained('bert-base')
        #--------------------------------------------------#
        if model_select == "TAPE":
            pretraining_name == None
        #--------------------------------------------------#
        if pretraining_name is not None:
            checkpoint = torch.load( data_folder / pretraining_name )
            model.load_state_dict(checkpoint['model_state_dict'])
        #--------------------------------------------------#
        model.mlm = Identity()
        model.eval()
        embed = datasets.EmbedDataset(data_file=input_file,tokenizer="iupac")
        loader = data.DataLoader(embed,batch_size,False,collate_fn = embed.collate_fn)
        #--------------------------------------------------#
        count_x = 0
        model.cuda()
        #--------------------------------------------------#
        seq_encodings = []
        seq_all_hiddens = []
        seq_ids = []
        for seq_batch in loader:
            count_x+=1
            ids, input_ids, input_mask = seq_batch["ids"],seq_batch["input_ids"],seq_batch["input_mask"]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = [] 
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                seq_emd = output[seq_num][1:seq_len-1]
                #print(seq_emd)
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print("features.shape: ", features.shape)
            seq_encodings.append(features)
            seq_ids += ids
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")
    #########################################################################################################
    #########################################################################################################
    if model_select == "BERT":
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        model = BertModel.from_pretrained("Rostlab/prot_bert")
        training_set = []
        seq_ids = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            seq = str(seq_record.seq)
            new_string = ""
            for i in range(len(seq)-1):
                new_string += seq[i]
                new_string += " "
            new_string += seq[-1]
            seq_ids.append(str(seq_record.id))
            training_set.append(new_string)
        ids = tokenizer.batch_encode_plus(training_set, add_special_tokens=True, padding=True)
        #--------------------------------------------------#
        loader = data.DataLoader(LoaderClass(np.array(ids["input_ids"]), np.array(ids["attention_mask"])), batch_size, False)
        model.cuda()
        model.eval()
        seq_encodings = []
        seq_all_hiddens = []
        count = 0
        #--------------------------------------------------#
        for seq_batch in loader:
            count+=1
            input_ids, input_mask = seq_batch[0],seq_batch[1]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = []
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                seq_emd = output[seq_num][1:seq_len-1]
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print(features.shape)
            seq_encodings.append(features)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")

    #########################################################################################################
    #########################################################################################################
    if model_select == "ALBERT":
        tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_albert", do_lower_case=False)
        model = AlbertModel.from_pretrained("Rostlab/prot_albert")
        training_set = []
        seq_ids = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            seq = str(seq_record.seq)
            new_string = ""
            for i in range(len(seq)-1):
                new_string += seq[i]
                new_string += " "
            new_string += seq[-1]
            seq_ids.append(str(seq_record.id))
            training_set.append(new_string)
        ids = tokenizer.batch_encode_plus(training_set, add_special_tokens=True, padding=True)
        #--------------------------------------------------#
        loader = data.DataLoader(LoaderClass(np.array(ids["input_ids"]), np.array(ids["attention_mask"])), batch_size, False)
        model.cuda()
        model.eval()
        seq_encodings = []
        seq_all_hiddens = []
        count = 0
        #--------------------------------------------------#
        for seq_batch in loader:
            count+=1
            input_ids, input_mask = seq_batch[0],seq_batch[1]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = [] 
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                seq_emd = output[seq_num][1:seq_len-1]
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print(features.shape)
            seq_encodings.append(features)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")
    #########################################################################################################
    #########################################################################################################
    if model_select == "Electra": ##### !!!!! Deprecated!
        generatorModelUrl = 'https://www.dropbox.com/s/5x5et5q84y3r01m/pytorch_model.bin?dl=1'
        discriminatorModelUrl = 'https://www.dropbox.com/s/9ptrgtc8ranf0pa/pytorch_model.bin?dl=1'
        generatorConfigUrl = 'https://www.dropbox.com/s/9059fvix18i6why/config.json?dl=1'
        discriminatorConfigUrl = 'https://www.dropbox.com/s/jq568evzexyla0p/config.json?dl=1'
        vocabUrl = 'https://www.dropbox.com/s/wck3w1q15bc53s0/vocab.txt?dl=1'
        downloadFolderPath = 'models/electra/'
        discriminatorFolderPath = os.path.join(downloadFolderPath, 'discriminator')
        generatorFolderPath = os.path.join(downloadFolderPath, 'generator')
        discriminatorModelFilePath = os.path.join(discriminatorFolderPath, 'pytorch_model.bin')
        generatorModelFilePath = os.path.join(generatorFolderPath, 'pytorch_model.bin')
        discriminatorConfigFilePath = os.path.join(discriminatorFolderPath, 'config.json')
        generatorConfigFilePath = os.path.join(generatorFolderPath, 'config.json')
        vocabFilePath = os.path.join(downloadFolderPath, 'vocab.txt')

        if not os.path.exists(discriminatorFolderPath):
            os.makedirs(discriminatorFolderPath)
        if not os.path.exists(generatorFolderPath):
            os.makedirs(generatorFolderPath)
        def download_file(url, filename):
            response = requests.get(url, stream=True)
            with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                                total=int(response.headers.get('content-length', 0)),
                                desc=filename) as fout:
                for chunk in response.iter_content(chunk_size=4096):
                    fout.write(chunk)
        if not os.path.exists(generatorModelFilePath):
            download_file(generatorModelUrl, generatorModelFilePath)
        if not os.path.exists(discriminatorModelFilePath):
            download_file(discriminatorModelUrl, discriminatorModelFilePath)
        if not os.path.exists(generatorConfigFilePath):
            download_file(generatorConfigUrl, generatorConfigFilePath)
        if not os.path.exists(discriminatorConfigFilePath):
            download_file(discriminatorConfigUrl, discriminatorConfigFilePath)
        if not os.path.exists(vocabFilePath):
            download_file(vocabUrl, vocabFilePath)

        tokenizer = ElectraTokenizer(vocabFilePath, do_lower_case=False )
        model = ElectraModel.from_pretrained(discriminatorFolderPath)
        model.cuda()
        model.eval()    
        training_set = []
        seq_ids = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            seq = str(seq_record.seq)
            new_string = ""
            for i in range(len(seq)-1):
                new_string += seq[i]
                new_string += " "
            new_string += seq[-1]
            seq_ids.append(str(seq_record.id))
            training_set.append(new_string)
        ids = tokenizer.batch_encode_plus(training_set, add_special_tokens=True, padding=True)
        loader = data.DataLoader(LoaderClass(np.array(ids["input_ids"]), np.array(ids["attention_mask"])), batch_size, False)
        seq_encodings = []
        seq_all_hiddens = []
        count = 0
        for seq_batch in loader:
            count+=1
            input_ids, input_mask = seq_batch[0],seq_batch[1]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = [] 
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                seq_emd = output[seq_num][1:seq_len-1]
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print(features.shape)
            seq_encodings.append(features)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")

    #########################################################################################################
    #########################################################################################################
    if model_select == "T5":
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        model.cuda()
        model.eval()    
        training_set = []
        seq_ids = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            seq = str(seq_record.seq)
            new_string = ""
            for i in range(len(seq)-1):
                new_string += seq[i]
                new_string += " "
            new_string += seq[-1]
            seq_ids.append(str(seq_record.id))
            training_set.append(new_string)
        ids = tokenizer.batch_encode_plus(training_set, add_special_tokens=True, padding=True)
        loader = data.DataLoader(LoaderClass(np.array(ids["input_ids"]), np.array(ids["attention_mask"])), batch_size, False)
        seq_encodings = []
        seq_all_hiddens = []
        count = 0
        for seq_batch in loader:
            count+=1
            input_ids, input_mask = seq_batch[0],seq_batch[1]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = [] 
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                seq_emd = output[seq_num][:seq_len-1]
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print(features.shape)
            seq_encodings.append(features)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")

    #########################################################################################################
    #########################################################################################################
    if model_select == "Xlnet":
        tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)
        model = XLNetModel.from_pretrained("Rostlab/prot_xlnet",mem_len=xlnet_mem_len)
        model.cuda()
        model.eval()    
        training_set = []
        seq_ids = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            seq = str(seq_record.seq)
            new_string = ""
            for i in range(len(seq)-1):
                new_string += seq[i]
                new_string += " "
            new_string += seq[-1]
            seq_ids.append(str(seq_record.id))
            training_set.append(new_string)
        ids = tokenizer.batch_encode_plus(training_set, add_special_tokens=True, padding=True)
        loader = data.DataLoader(LoaderClass(np.array(ids["input_ids"]), np.array(ids["attention_mask"])), batch_size, False)
        seq_encodings = []
        seq_all_hiddens = []
        count = 0
        for seq_batch in loader:
            count+=1
            input_ids, input_mask = seq_batch[0],seq_batch[1]
            input_ids, input_mask = input_ids.cuda(), input_mask.cuda()
            with torch.no_grad():
                output = model(input_ids,input_mask)
            output = output[0].cpu().detach().numpy()
            features = [] 
            for seq_num in range(len(output)):
                seq_len = (input_mask[seq_num] == 1).sum()
                padded_seq_len = len(input_mask[seq_num])
                seq_emd = output[seq_num][padded_seq_len-seq_len:padded_seq_len-2]
                seq_all_hiddens.append(seq_emd)
                features.append(np.mean(seq_emd,axis=0))
            features = np.stack(features)
            print(features.shape)
            seq_encodings.append(features)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")
    #########################################################################################################
    #########################################################################################################
    if model_select == "ESM_1B":
        #model, alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        batch_converter = alphabet.get_batch_converter()
        data_set = []
        for seq_record in SeqIO.parse(input_file, "fasta"):
            data_set.append((str(seq_record.id),str(seq_record.seq)))        
        model.eval()
        model.cuda()
        seq_encodings = []
        seq_all_hiddens = []
        seq_ids = []
        for i in range(0,len(data_set),batch_size):
            if i+batch_size<=len(data_set):
                batch = data_set[i:i+batch_size]
            else:
                batch = data_set[i:]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)
            seq_ids += batch_labels
            print(batch_tokens.size())
            batch_tokens = batch_tokens.cuda()
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33])
            results = results["representations"][33].cpu().detach()
            print(results.size())
            sequence_representations = []
            for i, ( _ , seq ) in enumerate(batch):
                seq_all_hiddens.append(results[i, 1 : len(seq) + 1].numpy())
                sequence_representations.append(results[i, 1 : len(seq) + 1].mean(0))
            sequence_representations = np.stack(sequence_representations)
            seq_encodings.append(sequence_representations)
        seq_embeddings = np.concatenate(seq_encodings)
        print("seq_embeddings.shape: ", seq_embeddings.shape)
        seq_embedding_output = {"seq_embeddings":seq_embeddings, "seq_ids":seq_ids, "seq_all_hiddens":seq_all_hiddens}
        pickle.dump( seq_embedding_output, open( output_file, "wb" ) )
        print("done")

    #########################################################################################################
    #########################################################################################################
    return seq_embedding_output

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
if __name__ == "__main__":
    #--------------------------------------------------#
    data_folder = Path("X_DataProcessing/")
    input_seq_fasta_file = "X00_phosphatase.fasta"
    #--------------------------------------------------#
    models_list = ["TAPE_FT", "BERT", "ALBERT", "Electra", "T5", "Xlnet", "ESM_1B", "TAPE"]
    model_select = models_list[5] ##### !!!!! models_list[3] Electra deprecated !
    #--------------------------------------------------#
    pretraining_name = "X01_phosphatase_FT_epoch10_trial_training.pt"
    #--------------------------------------------------#
    batch_size=15
    output_file_name_header = "X03_embedding_"
    #====================================================================================================#
    embedding_LM(model_select, data_folder ,input_seq_fasta_file, output_file_name_header, pretraining_name, batch_size, xlnet_mem_len=512)
    #sound(440,0.5)
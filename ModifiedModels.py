from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from tape.datasets import *
from tape.models.modeling_utils import *

import typing
from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from pathlib import Path
from copy import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.weight_norm import weight_norm
from torch.utils.data import Dataset

import numpy as np
import random

#########################################################################
#########################################################################
class MLMD(Dataset): # MaskedLanguageModelingDataset_modified
    def __init__(self,
                 data_path: Union[str, Path],
                 tokenizer: Union[str, TAPETokenizer] = 'iupac'):
        super().__init__()
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        self.data = dataset_factory(data_path)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        tokens = self.tokenizer.tokenize(item['primary'])
        tokens = self.tokenizer.add_special_tokens(tokens)
        
        masked_tokens, labels = self._apply_bert_mask(tokens)
        tokens_ids=np.array(self.tokenizer.convert_tokens_to_ids(tokens), np.int64)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        input_mask = np.ones_like(masked_token_ids)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        return masked_token_ids, input_mask, tokens_ids #####!!!!! return tokens instead of labels here!!
    
    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, tokens_ids = tuple(zip(*batch)) #####!!!!! replace tokens_ids instead of lm_label_ids here!!
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))

        # ignore_index is -1
        #lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))
        tokens_ids = torch.from_numpy(pad_sequences(tokens_ids, 0))
        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': tokens_ids}
    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1

        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                pass
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)
                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token
                    token = self.tokenizer.convert_id_to_token(
                        random.randint(0, self.tokenizer.vocab_size - 1))
                else:
                    # 10% chance to keep current token
                    pass
                masked_tokens[i] = token
        return masked_tokens, labels #####!!!!! return tokens instead of labels here!!
    
#########################################################################
#########################################################################   
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
#########################################################################
#########################################################################
class MLMHead_1(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 hidden_act: typing.Union[str, typing.Callable] = 'gelu',
                 layer_norm_eps: float = 1e-12,
                 ignore_index: int = -100):
        super().__init__()
        self.transform = PredictionHeadTransform(hidden_size, hidden_act, layer_norm_eps)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        
        self.bias = nn.Parameter(data=torch.zeros(vocab_size))  # type: ignore
        self.vocab_size = vocab_size
        self._ignore_index = ignore_index

    def forward(self, hidden_states, targets=None):
        hidden_states = self.transform(hidden_states)
        outputs = (hidden_states,)
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            masked_lm_loss = loss_fct(
                hidden_states.view(-1, self.vocab_size), targets.view(-1))
            metrics = {'perplexity': torch.exp(masked_lm_loss)}
            loss_and_metrics = (masked_lm_loss, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs  # (loss), prediction_scores
    
#########################################################################
#########################################################################
class MLMHead_2(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 hidden_act: typing.Union[str, typing.Callable] = 'gelu',
                 layer_norm_eps: float = 1e-12,
                 ignore_index: int = -100):
        super().__init__()
        self.transform = PredictionHeadTransform(hidden_size, hidden_act, layer_norm_eps)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        
        self.bias = nn.Parameter(data=torch.zeros(vocab_size))  # type: ignore
        self.vocab_size = vocab_size
        self._ignore_index = ignore_index

    def forward(self, hidden_states, targets=None):

        outputs = (hidden_states,)
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            masked_lm_loss = loss_fct(
                hidden_states.view(-1, self.vocab_size), targets.view(-1))
            metrics = {'perplexity': torch.exp(masked_lm_loss)}
            loss_and_metrics = (masked_lm_loss, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs  # (loss), prediction_scores
    

#########################################################################
#########################################################################     
class MLMHead_3(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 hidden_act: typing.Union[str, typing.Callable] = 'gelu',
                 layer_norm_eps: float = 1e-12,
                 ignore_index: int = -100):
        super().__init__()
        self.transform = PredictionHeadTransform(hidden_size, hidden_act, layer_norm_eps)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        
        self.bias = nn.Parameter(data=torch.zeros(vocab_size))  # type: ignore
        self.vocab_size = vocab_size
        self._ignore_index = ignore_index

    def forward(self, hidden_states, targets=None):

        outputs = (hidden_states,)
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            masked_lm_loss = loss_fct(
                hidden_states.view(-1, self.vocab_size), targets.view(-1))
            metrics = {'perplexity': torch.exp(masked_lm_loss)}
            loss_and_metrics = (masked_lm_loss, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs  # (loss), prediction_scores   


#########################################################################
#########################################################################

import logging
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tape import models
from tape.models import modeling_utils

from tape import ProteinConfig
from tape import ProteinModel
from tape.models.modeling_utils import prune_linear_layer
from tape.models.modeling_utils import get_activation_fn
from tape.models.modeling_utils import LayerNorm
from tape.models.modeling_utils import MLMHead
from tape.models.modeling_utils import ValuePredictionHead
from tape.models.modeling_utils import SequenceClassificationHead
from tape.models.modeling_utils import SequenceToSequenceClassificationHead
from tape.models.modeling_utils import PairwiseContactPredictionHead
from tape.registry import registry


from tape.models import modeling_bert
from tape.models.modeling_bert import *


logger = logging.getLogger(__name__)

URL_PREFIX = "https://s3.amazonaws.com/proteindata/pytorch-models/"
BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base': URL_PREFIX + "bert-base-pytorch_model.bin",
}
BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base': URL_PREFIX + "bert-base-config.json"
}


class ProteinBertForMaskedLM_ZX(ProteinBertAbstractModel):
    
    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBertModel(config)
        
        self.mlm = MLMHead_3(
            config.hidden_size, config.vocab_size, config.hidden_act, config.layer_norm_eps,
            ignore_index=-1)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.mlm.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self,
                input_ids,
                input_mask=None,
                targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]
        # add hidden states and attention if they are here
        outputs = self.mlm(pooled_output, targets) + outputs[2:]
        # (loss), prediction_scores, (hidden_states), (attentions)
        return outputs





















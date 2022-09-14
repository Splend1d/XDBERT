# coding=utf-8
# Copyleft 2019 project LXRT.

from collections import defaultdict
import json
import random
import os
import pickle5 as pickle
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
from models.clip import tokenize_clip
#from param import args

from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead
)


Split2CorpusPath = {
    'train': 'data/wiki-cased/wiki.train.raw',
    'val': 'data/wiki-cased/en.valid.raw'
}

class InputExample(object):
    """A single training/test example for the language model."""
    def __init__(self, token_ids, vis_input_ids,is_matched=None):
        self.token_ids = token_ids
        self.vis_input_ids = vis_input_ids
        self.is_matched = is_matched  # whether the visual and obj matched

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, vis_input_ids, input_mask, segment_ids,lm_label_ids, is_matched):
        self.input_ids = input_ids
        self.vis_input_ids = vis_input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        # for clip
        #self.sent = sent
        self.is_matched = is_matched

class SSLDataset(Dataset):
    def __init__(self, args,tokenizer,splits, max_seq_len=512, tail = None):

        self.task_matched = args.task_matched

        file_path = Split2CorpusPath[splits]
        #assert os.path.isfile(file_path)
         #if args.tokenizer_name:
        tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            do_lower_case=True
        )
        if max_seq_len < 0:
            max_seq_len = tokenizer.model_max_length - 2
        #tokenizer = args.tokenizer_name
        max_seq_len = max_seq_len - 2#(tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        #print(max_seq_len)
        #s()
        directory, filename = os.path.split(file_path)
        header = tokenizer_name + "_cached_lm_" + str(max_seq_len) + "_" + filename
        print(header)
        # cached_features_file = os.path.join(
        #     directory, header
        # )
        

        # if os.path.exists(cached_features_file) and not args.overwrite_cache:       
        if True:   # we always pretokenize the data because it is too large 
            self.examples = []
            for file in os.listdir(directory):
                if file.startswith(header) and (file[-1] == tail or not tail):
                    cached_features_file = os.path.join(
                        directory, file
                    )
                    #print("Loading features from cached file %s", cached_features_file)
                    with open(cached_features_file, "rb") as handle:
                        self.examples += (pickle.load(handle))
            print("examples length",len(self.examples))
        
        
        else:
            #print("Creating features from dataset file at %s", directory,file_path)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
            #print("Finish reading dataset file at %s", file_path)
            #enter = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("\n"))
            #print(enter)
            #s()
            text = text.split("\n")
            tokenized_text = []
            excluded_samples = 0
            for i in tqdm(range(len(text))):
                tokenized_text += (tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text[i])))
            #print(tokenized_text)
            #print(f"Excluded {excluded_samples} samples because sentence is too long")
            print(len(tokenized_text))
            for i in tqdm(range(0, len(tokenized_text) - max_seq_len + 1, max_seq_len)):  # Truncate in block of block_size
                #print(tokenized_text[i : i + max_seq_len])
                token_ids = tokenizer.bos_token + tokenized_text[i  : i + max_seq_len] + tokenizer.eos_token
                assert len(token_ids) == 512
                #assert text_ids[0] == 101 and text_ids[1] == 102
                text_recon = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokenized_text[i : i + max_seq_len]))
                #print(text_ids)
                #assert len(text_ids) == 510
                vis_input_ids = tokenize_clip([text_recon],context_length = 77, assign_seg = 9)
                #print(i.shape[1])
                new_datum = {
                    'token_ids': token_ids,
                    'sent': text_recon,
                    'vis_input_ids': vis_input_ids
                }
                self.examples.append(new_datum)
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            print("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item) -> InputExample:
        datum = self.examples[item]
        #torch.tensor(self.examples[item][0], dtype=torch.long), self.examples[item][1]
        if self.task_matched:
            if random.random() < 0.5:
                is_matched = 0
                other_item = random.randint(0, len(self.examples)-1) 
                while other_item == item:
                    other_item = random.randint(0, len(self.examples)-1) 
                other_datum = self.examples[other_item]
                #sent = other_datum['sent']
                vis_input_ids = other_datum["vis_input_ids"]
                
                is_matched = 0
            else:
                is_matched = 1
                #sent = datum["sent"]
                vis_input_ids = datum["vis_input_ids"]
        else:
            vis_input_ids = datum["vis_input_ids"]
            is_matched = None
        example = InputExample(
            datum["token_ids"], vis_input_ids,
            is_matched = is_matched
        )

        return example 



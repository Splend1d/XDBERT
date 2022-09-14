# coding=utf-8
# Copyright 2019 project LXRT.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LXRT model."""

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
import random

import collections
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from models.clip import load_clip, tokenize_clip
from .xattn_data import InputExample, InputFeatures
from models.utils_model import BertPreTrainingHeads,LXRTXLayer, BertPooler
from pretrain.xattn_data import InputExample, InputFeatures, SSLDataset
from tqdm import tqdm
from param import args
import transformers
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
DataTuple = collections.namedtuple("DataTuple", 'dataset torchdset loader evaluator')
LOSSES_NAME = ["mlm","vlm","matched"]



def cleanup():
    dist.destroy_process_group()

def random_word(i, tokenizer):
    """
    Masking some random token_ids for Language Model task with probabilities as in the original BERT paper.
    :param token_ids: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked token_ids and related labels for LM prediction
    """
    original_ids = torch.LongTensor(i)
    token_ids = torch.LongTensor(i)
    mask_ids = torch.ones(token_ids.shape, dtype = torch.long)*103
    output_label = torch.ones(token_ids.shape, dtype = torch.long)*-1
    probs = torch.ones(token_ids.shape)*0.85
    # dont mask cls and sep
    probs[0] = 1
    probs[-1] = 1
    probs2 = torch.ones(token_ids.shape)*0.1
    keep = torch.bernoulli(probs)
    corrupt = torch.bernoulli(probs2)
    corrupt_tokens = torch.randint(1001, len(tokenizer.vocab)-1,token_ids.shape)  
    original = torch.bernoulli(probs2)

    token_ids = torch.where(keep == 1,token_ids,mask_ids)
    output_label = torch.where(keep == 1,output_label,original_ids)
    
    #corrupt
    token_ids = torch.where(torch.eq(token_ids ,103) & torch.eq(original ,1),original_ids,token_ids)
    token_ids = torch.where(torch.eq(token_ids ,103) & torch.eq(corrupt ,1),corrupt_tokens,token_ids)
    
    #print(token_ids[:10],output_label[:10])
    #o = o.maximum(,0)
    #print(tokenizer.convert_ids_to_tokens(token_ids[:30]),tokenizer.convert_ids_to_tokens(torch.where(output_label[:30] > 0,output_label[:30],0)))
    #s()
    #random 
    # for i, token in enumerate(token_ids[1:-1],1): # Dont mask start and end
    #     prob = random.random()
    #     # mask token with probability
    #     ratio = args.word_mask_rate
    #     if prob < ratio:
    #         prob /= ratio

    #         # append current token to output (we will predict these later)
    #         output_label[i] = token

    #         # 80% randomly change token to mask token
    #         if prob < 0.8:
    #             token_ids[i] = 103

    #         # 10% randomly change token to random token
    #         elif prob < 0.9:
    #             token_ids[i] = random.randint(1001,len(tokenizer.vocab)-1)#choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            
        
    #print(len(output_label))
    return token_ids.tolist(), output_label.tolist()
def convert_example_to_features(example: InputExample, tokenizer, max_seq_len)->InputFeatures:
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param clip_tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    
    #text stream
    masked_tokens, lm_label_ids = random_word(example.token_ids,tokenizer)
    #input_ids = example.token_ids # pad CLS and SEP
    #tokens = clip_tokenizer.tokenize(example.sent.strip())
    #lm_label_ids =  masked_label 
    input_mask = [1] * len(masked_tokens) + max(max_seq_len - len(masked_tokens),0) * [0]
    segment_ids = max_seq_len * [0]
    masked_tokens = masked_tokens + max(max_seq_len - len(masked_tokens),0) * [0]
    #print(len(input_ids))
    assert len(masked_tokens) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len  
    features = InputFeatures(
        input_ids=masked_tokens,
        vis_input_ids = example.vis_input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        lm_label_ids=lm_label_ids,
        #sent = example.sent,
        is_matched=example.is_matched,
    )
    return features

class XATTNBERTModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.config = transformers.AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
        model = transformers.AutoModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=self.config,
            cache_dir=args.cache_dir,
        )
        self.txtmodel = model

        # IMAGE model
        self.vismodel, _ = load_clip('ViT-B/32', "cuda", jit = False)
        vis_config = copy.deepcopy(self.config)
        vis_config.hidden_size = 512 # clip hidden size
        vis_config.embedding_size = 512
        vis_config.num_attention_heads = 8 # clip number of heads
        print(vis_config)
        
        # CROSS model
        self.xmodel = nn.ModuleList(
            [LXRTXLayer(self.config,vis_config) for _ in range(args.xlayers)]
        )

        # POOLER
        self.pooler = BertPooler(self.config)
        #classifier
        self.cls = BertPreTrainingHeads(self.config, self.txtmodel.embeddings.word_embeddings.weight)
        self.cls_vis = BertPreTrainingHeads(vis_config, self.vismodel.token_embedding.weight)

        #tasks
        self.task_matched = args.task_matched 
        self.task_mask_lm = args.task_mask_lm

    def forward(self, input_ids,vis_input_ids , token_type_ids=None, input_mask=None,
        visn_input_mask = None, lm_labels = None, matched_label=None, ans=None):
        #print(input_ids, token_type_ids,input_mask)
        #s()
        lang_feats = self.txtmodel(
            input_ids, token_type_ids = token_type_ids, attention_mask = input_mask
            #visual_feats=(visual_feats, pos),
        ).last_hidden_state
        #print("l",lang_feats.shape)
        #s()
        #lang_feats_diff = lang_feats - torch.mean(lang_feats, dim = 1, keepdim = True)
        #diversity_txt = torch.mean(torch.norm(lang_feats_diff, dim = -1))  
        #print("div txt BERT",diversity_txt,lang_feats[0,:3,:3])
        #print(vis_input_ids.shape)
        bs, nseg, context_length = vis_input_ids.shape
        vis_input_ids = vis_input_ids.reshape(-1,context_length)
        #bs * nseg, context_length 
        #print(vis_input_ids.shape,vis_input_ids)
        visn_feats_verbose,visn_input_mask, vlm_labels, _ = self.vismodel.encode_text(vis_input_ids)
        visn_feats_verbose = visn_feats_verbose.reshape(bs,-1,visn_feats_verbose.shape[-1])
        #visn_feats = visn_feats.reshape(bs, nseg, visn_feats.shape[-1])
        # visn_feats_diff = visn_feats - torch.mean(visn_feats, dim = 1, keepdim = True)
        # diversity_vis = torch.mean(torch.norm(visn_feats_diff, dim = -1))  
        #print("div vis clip",diversity_vis,visn_feats[0,:3,:3])
        #print(visn_input_mask.shape)
        #print(visn_feats.shape,visn_feats)
        
        #bs * nseg, context_length, hidden_dim
        #bs * nseg, hidden_dim
        #visn_feats_verbose = visn_feats_verbose.reshape(bs, nseg, context_length, visn_feats_verbose.shape[-1])
        #print(visn_feats_verbose[0])
        
        visn_input_mask = visn_input_mask.reshape(bs,nseg * visn_input_mask.shape[-1])
        #vlm_labels = vlm_labels.reshape(bs,nseg,-1)
        #print(visn_feats.shape,visn_feats)
        
        #bs, nseg
        #visn_feats=visn_feats.unsqueeze(1).float() # sequence of 1
        #print("v", visn_feats.shape)

        extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)
        


        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if visn_input_mask is not None:
            extended_vision_attention_mask = visn_input_mask.unsqueeze(1).unsqueeze(2)
        


            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            extended_vision_attention_mask = extended_vision_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_vision_attention_mask = (1.0 - extended_vision_attention_mask) * -10000.0
            #print(extended_vision_attention_mask.shape,extended_vision_attention_mask)
        #print(lang_feats)
        #print(extended_attention_mask,extended_attention_mask.shape)
        #print(extended_vision_attention_mask,extended_vision_attention_mask.shape)
        #print(visn_feats_verbose,visn_feats_verbose.shape)
        #print(visn_input_mask)
        for layer_module in self.xmodel:
            lang_feats, visn_feats_verbose = layer_module(lang_feats, extended_attention_mask,
                                                  visn_feats_verbose, extended_vision_attention_mask)
        #print("after x l",lang_feats)
        #print("after x v",visn_feats)
        #lang_feats_diff = lang_feats - torch.mean(lang_feats, dim = 1, keepdim = True)
        #diversity = torch.mean(torch.norm(lang_feats_diff, dim = -1))
        #print("x div txt",diversity,lang_feats[0,:3,:3])

        #visn_feats_diff = visn_feats - torch.mean(visn_feats, dim = 1, keepdim = True)
        #diversity_vis = torch.mean(torch.norm(visn_feats_diff, dim = -1))  
        #print("x div vis",diversity_vis,visn_feats[0,:3,:3])

        lang_pooled_output = self.pooler(lang_feats)
        #print("pooled",lang_pooled_output)
        #print(lang_feats.shape,lang_pooled_output.shape, visn_feats_verbose.shape,vlm_labels.shape)
        #print(self.cls,self.cls_vis)
        lang_prediction_scores, cross_relationship_score = self.cls(lang_feats, lang_pooled_output)
        vis_prediction_scores, _ = self.cls_vis(visn_feats_verbose, None)
        #print(lang_prediction_scores.shape,vis_prediction_scores.shape, torch.min(lm_labels),torch.max(lm_labels), \
        #    torch.min(vlm_labels),torch.max(vlm_labels))
        #print(self.vismodel.token_embedding.weight.shape)
        total_loss = 0.
        loss_fct = CrossEntropyLoss(ignore_index = -1)
        losses = ()
        if lm_labels is not None and self.task_mask_lm:
            masked_lm_loss = loss_fct(
                lang_prediction_scores.view(-1, self.config.vocab_size),
                lm_labels.view(-1)
            )
            total_loss += masked_lm_loss
            losses += (masked_lm_loss.detach(),)
            #VLM
            masked_vlm_loss = loss_fct(
                vis_prediction_scores.view(-1, self.vismodel.token_embedding.weight.shape[0]),
                vlm_labels.view(-1)
            )
            total_loss += masked_vlm_loss
            losses += (masked_vlm_loss.detach(),)
        #print(matched_label,self.task_matched)
        if matched_label is not None and self.task_matched:
            matched_loss = loss_fct(
                cross_relationship_score.view(-1, 2),
                matched_label.view(-1)
            )
            total_loss += matched_loss
            losses += (matched_loss.detach(),)
        #print("losses",losses)
        return total_loss, torch.stack(losses).unsqueeze(0), None#answer_score.detach()

class XATTNBERTTrainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.grad_accumulation = args.grad_accumulation
        self.gpu = args.gpu
        # tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_type
        # for model_class in MODEL_CLASSES:
        #     if tokenizer_name.startswith(model_class):
        #         config_class, model_class, tokenizer_class = MODEL_CLASSES[model_class]
         
        # self.tokenizer = tokenizer_class.from_pretrained(
        #     tokenizer_name,
        #     do_lower_case=True
        # )
        # Build model
        # TEXT model
        self.model = XATTNBERTModel(args)

        if args.from_scratch:
            print("Train from Scratch: re-initialize all BERT weights.")
            self.model.apply(self.model.init_bert_weights)
        if args.load is not None:
            self.load(args.load)
            epoch = int(args.load[-2:])
            assert type(epoch) == int
            self.startfrom = int(epoch)
            print(self.startfrom)
        else:
            self.startfrom = -1
        if args.load_lxmert is not None:
            # Load lxmert would not load the answer head.
            self.load_lxmert(args.load_lxmert)

        # GPU Options
        
        # if args.multiGPU:
        #     print(args.gpu)
        #     self.model = nn.parallel.DistributedDataParallel(self.model, device_ids = [args.gpu], find_unused_parameters=True)
        # else:
        #     assert False==True
            #self.model = self.model.cuda()


    def forward(self, examples):
        
        #s()
        
        #print(type(examples))
        train_features = [convert_example_to_features(example, self.model.tokenizer, self.max_seq_len) for example in examples]

        # language Inputs
        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        #print(input_ids[:,:10])
        #s()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Inputs
        vis_input_ids = torch.cat([f.vis_input_ids for f in train_features], dim = 0).cuda()
        #vis_input_ids = vis_input_ids.cuda()
        #visn_input_mask = visn_input_mask.cuda()
        # bs, nseg, 77

        #feats = torch.from_numpy(np.stack([f.visual_feats[0] for f in train_features])).cuda()
        #pos = torch.from_numpy(np.stack([f.visual_feats[1] for f in train_features])).cuda()

        # Language Prediction
        lm_labels = torch.tensor([f.lm_label_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Prediction
        # obj_labels = {}
        # for key in ('obj', 'attr', 'feat'):
        #     visn_labels = torch.from_numpy(np.stack([f.obj_labels[key][0] for f in train_features])).cuda()
        #     visn_mask = torch.from_numpy(np.stack([f.obj_labels[key][1] for f in train_features])).cuda()
        #     assert visn_labels.size(0) == visn_mask.size(0) and visn_labels.size(1) == visn_mask.size(1)
        #     obj_labels[key] = (visn_labels, visn_mask)

        # Joint Prediction
        matched_labels = torch.tensor([f.is_matched for f in train_features], dtype=torch.long).cuda()
        #ans = torch.from_numpy(np.stack([f.ans for f in train_features])).cuda()

        """
        forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                visual_feats=None, pos=None, obj_labels=None, matched_label=None, ans=None):
        """
        #print(input_ids,segment_ids,matched_labels)
        loss, losses, ans_logit = self.model(
            input_ids, vis_input_ids, segment_ids, input_mask,lm_labels = lm_labels,matched_label = matched_labels
        )
        # loss, losses, ans_logit = self.model(
        #     input_ids, segment_ids, input_mask, lm_labels,
        #     feats, pos, obj_labels, matched_labels, ans
        # )
        return loss, losses.detach().cpu(), ans_logit

    def train_batch(self, batch,n, epoch):
        if epoch > self.startfrom:
            loss, losses, ans_logit = self.forward(batch)
            loss = loss / self.grad_accumulation
            if args.multiGPU:
                loss = loss.mean() 
                losses = losses.mean(0) 
            loss.backward()
        else:
            loss = torch.Tensor([0])
            losses = None
            ans_logit = None
        if (n+1) % self.grad_accumulation == 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optim.step()
            self.optim.zero_grad()
            #torch.distributed.barrier()
            #loss = loss.item()

        return loss.item(), losses, ans_logit

    def valid_batch(self, batch):
        with torch.no_grad():
            loss, losses, ans_logit = self.forward(batch)
            if args.multiGPU:
                loss = loss.mean()
                losses = losses.mean(0)
        return loss.item(), losses, ans_logit

    def train(self, train_dataset, valid_dataset, args):
        
        if args.gpu == 0:
            print("Finished creating distributed parellel")
        # Optimizer
        from models.optimization import BertAdam
        batch_per_epoch = 13960*10  #len(train_dataloader)
        t_total = int(batch_per_epoch * args.epochs)
        warmup_ratio = 0.05
        warmup_iters = int(t_total * warmup_ratio)
        if args.gpu == 0:
            print("Batch per epoch: %d" % batch_per_epoch)
            print("Total Iters: %d" % t_total)
            print("Warm up Iters: %d" % warmup_iters)
        self.optim = BertAdam(self.model.parameters(), lr=args.lr, warmup=warmup_ratio, t_total=t_total)

        # Train
        best_eval_loss = 9595.
        for epoch in range(args.epochs):
            # Train
            self.model.train()
            if epoch < args.freeze_pretrained:
                for param in self.model.txtmodel.parameters():
                    param.requires_grad = False
                for param in self.model.vismodel.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.txtmodel.parameters():
                    param.requires_grad = True
                for param in self.model.vismodel.parameters():
                    param.requires_grad = True
            dataset_folds = list("bcdefghijklmnopqrstuvwx") +["extra"]
            #random.shuffle(dataset_folds)
            dataset_folds = ["a"] + dataset_folds 
            for subepoch,dataset_fold in enumerate(dataset_folds):
                if args.gpu != 0:
                    torch.distributed.barrier()

                train_dataset = SSLDataset(args, args.model_name_or_path, args.train, max_seq_len=args.max_seq_len,tail = None) #args.model_name_or_path = tokenizer
                valid_dataset = SSLDataset(args, args.model_name_or_path, args.valid, max_seq_len=args.max_seq_len,tail =None)
                if args.gpu == 0:
                    torch.distributed.barrier()
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=args.world_size,
                    rank=args.rank,
                    shuffle=True,
                )
                train_dataloader = DataLoader(
                    train_dataset, sampler=train_sampler, shuffle=False, num_workers=0,collate_fn=lambda x: x,
                    batch_size=args.batch_size, pin_memory=True, drop_last = True
                )
                valid_sampler = DistributedSampler(
                    valid_dataset,
                    num_replicas=args.world_size,
                    rank=args.rank,
                    shuffle=True,
                )
                valid_dataloader = DataLoader(
                    valid_dataset, sampler=valid_sampler, shuffle=False, num_workers=0,collate_fn=lambda x: x,
                    batch_size=args.batch_size, pin_memory=True, drop_last = True
                )
                total_loss = 0.
                total_losses = 0.
                uid2ans = {}
                if args.gpu == 0:
                    titer = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
                else:
                    titer = enumerate(train_dataloader)
                for n,batch in titer:
                    loss, losses, logit = self.train_batch(batch, n, epoch)
                    if epoch > self.startfrom:
                        total_loss += loss
                        total_losses += losses
                        if n % 100 == 0 and n and args.gpu == 0:
                            #print(total_losses)

                            #titer.set_description("The average training loss for Epoch %d is %0.4f" % (epoch, total_loss / (n+1)*train_dataloader.batch_size))

                            losses_str = "The losses are "
                            for name, loss in zip(LOSSES_NAME, total_losses): #.squeeze()

                                losses_str += "%s: %0.4f " % (name, loss / ((n+1)*train_dataloader.batch_size))
                            if args.gpu == 0:
                                titer.set_description(losses_str)
                                titer.refresh()
                #avg_eval_loss = self.evaluate_epoch(valid_dataloader)
                if args.gpu == 0:
#                     if avg_eval_loss < best_eval_loss:
#                         best_eval_loss = avg_eval_loss
#                         self.save("BEST_EVAL_LOSS")
                    self.save_intermediate("Epoch%02d" % (epoch+1),subepoch)
                torch.distributed.barrier()
            if epoch <= self.startfrom:
                continue
            if args.gpu == 0:
                print("The final training loss for Epoch %d is %0.4f" % (epoch, total_loss / (batch_per_epoch*train_dataloader.batch_size)))
            #losses_str = "The losses are "
            # for name, loss in zip(LOSSES_NAME, total_losses):
            #     losses_str += "%s: %0.4f " % (name, loss / batch_per_epoch)
            #print(losses_str)
            # if args.task_qa:
            #     train_tuple.evaluator.evaluate(uid2ans, pprint=True)

            # Eval
            #avg_eval_loss = self.evaluate_epoch(valid_dataloader)

            # Save
            if args.gpu == 0:
#                 if avg_eval_loss < best_eval_loss:
#                     best_eval_loss = avg_eval_loss
#                     self.save("BEST_EVAL_LOSS")
                self.save("Epoch%02d" % (epoch+1))

    def evaluate_epoch(self, valid_dataloader, iters: int=-1):
        self.model.eval()
        
        total_loss = 0.
        total_losses = 0.
        uid2ans = {}
        for i, batch in enumerate(valid_dataloader):
            loss, losses, logit = self.valid_batch(batch)
            total_loss += loss
            total_losses += losses
            # if args.task_qa:
            #     score, label = logit.max(1)
            #     for datum, l in zip(batch, label.cpu().numpy()):
            #         uid = datum.uid
            #         ans = train_tuple.dataset.answer_table.id2ans(l)
            #         uid2ans[uid] = ans
            
        if self.gpu == 0:
            print("The valid loss is %0.4f" % (total_loss / (len(valid_dataloader)*valid_dataloader.batch_size)))
        # losses_str = "The losses are "
        # for name, loss in zip(LOSSES_NAME, total_losses / len(eval_ld)):
        #     losses_str += "%s: %0.4f " % (name, loss)
        # print(losses_str)

        # if args.task_qa:
        #     eval_tuple.evaluator.evaluate(uid2ans, pprint=True)

        return total_loss / len(valid_dataloader)

    def save_intermediate(self, name,subepoch):
        os.makedirs(os.path.join(args.output,name), exist_ok=True)
        #assert os.path.join(args.output,name, "%s_XATTNBERT.pth" % name) not in os.listdir(os.path.join(args.output,name))
        torch.save(self.model.state_dict(),
                   os.path.join(args.output,name, f"{name}_XATTNBERT_{subepoch}.pth"))
        torch.save(self.model.txtmodel.state_dict(),
                   os.path.join(args.output,name, f"{name}_txtmodel_{subepoch}.pth"))
#         try:
#             torch.save(self.optim.state_dict(), os.path.join(os.path.join(args.output,name), "optimizer.pt"))
#         except:
#             pass
        
    def save(self, name):
        os.makedirs(os.path.join(args.output,name), exist_ok=True)
        #assert os.path.join(args.output,name, "%s_XATTNBERT.pth" % name) not in os.listdir(os.path.join(args.output,name))
        torch.save(self.model.state_dict(),
                   os.path.join(args.output,name, "%s_XATTNBERT.pth" % name))
        torch.save(self.model.txtmodel.state_dict(),
                   os.path.join(args.output,name, "%s_txtmodel.pth" % name))
        try:
            torch.save(self.optim.state_dict(), os.path.join(os.path.join(args.output,name), "optimizer.pt"))
        except:
            pass
        #torch.save(scheduler.state_dict(), os.path.join(oos.path.join(args.output,name), "scheduler.pt"))

    def load(self, path):
        print("Load BERT extractor from %s" % path)
        state_dict = torch.load("%s_XATTNBERT.pth" % path)
        self.model.load_state_dict(state_dict)

    def load_lxmert(self, path):
        print("Load LXMERT model from %s" % path)
        state_dict = torch.load("%s_XATTNBERT.pth" % path)

        # Do not load any answer head
        for key in list(state_dict.keys()):
            if 'answer' in key:
                state_dict.pop(key)

        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        state_dict = new_state_dict

        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Keys in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Keys in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        self.model.load_state_dict(state_dict, strict=False)

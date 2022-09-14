# coding=utf-8
# Copyleft 2019 project LXRT.

import collections
import os
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pretrain.param import args
from pretrain.xattn_data import InputExample, InputFeatures, SSLDataset
from pretrain.model_main import XATTNBERTTrainer
from models.clip import tokenize_clip
import torch.multiprocessing as mp
DataTuple = collections.namedtuple("DataTuple", 'dataset torchdset loader evaluator')


# def get_tuple(splits: str,tokenizer, bs: int, shuffle=False, drop_last=False, topk=-1) -> DataTuple:


#     # Build dataset, data loader, and evaluator.

#     dset = SSLDataset(args, tokenizer,splits, max_seq_len=args.max_seq_len)

#     tset = None#LXMERTTorchDataset(dset, topk)
#     data_loader = DataLoader(
#         dset, batch_size=bs,
#         shuffle=shuffle, num_workers=args.num_workers,
#         collate_fn=lambda x: x,
#         drop_last=drop_last, pin_memory=True
#     )
#     evaluator = None#LXMERTEvaluator(dset)
#     #print()

#     return DataTuple(dataset=dset, torchdset=tset, loader=data_loader, evaluator=evaluator)

# train_tuple = get_tuple(args.train,args.model_name_or_path, args.batch_size, shuffle=True, drop_last=True)
# valid_batch_size = 2048 if args.multiGPU else 32
# valid_tuple = get_tuple(args.valid,args.model_name_or_path, valid_batch_size, shuffle=False, drop_last=False, topk=5000)


# +
#LOSSES_NAME = ('MLM','Matched')
# -

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

# +
def setup(gpu, args):
	
	torch.cuda.set_device(gpu)
	device = torch.device("cuda", gpu)
	args.gpu = gpu                                  # Local device id.
	args.device = device                            # Local device object.
	args.rank = args.nr * args.gpus + gpu           # The gpu id in the world.
	torch.distributed.init_process_group(
		backend="nccl",
		init_method='env://',
		world_size=args.world_size,
		rank=args.rank
	)
	set_seed(args)
	lxmert = XATTNBERTTrainer(args)
	
	lxmert.to(args.device)
	lxmert = torch.nn.parallel.DistributedDataParallel(
		lxmert, device_ids=[args.gpu], find_unused_parameters=True
	)
	if gpu == 0:
		
		lxmert.module.save("INIT")
	torch.distributed.barrier()
	if args.train:
# 		if gpu != 0:
# 			torch.distributed.barrier()
		
# 		train_dataset = SSLDataset(args, args.model_name_or_path, args.train, max_seq_len=args.max_seq_len) #args.model_name_or_path = tokenizer
# 		valid_dataset = SSLDataset(args, args.model_name_or_path, args.valid, max_seq_len=args.max_seq_len)
# 		if gpu == 0:
# 			torch.distributed.barrier()
		lxmert.module.train(None, None, args)


# -

def is_port_in_use(port):
	import socket
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		return s.connect_ex(('localhost', port)) == 0

if __name__ == "__main__":
	#random.seed(997)
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	port = 9595
	while is_port_in_use(port):
		port += 1
	print("Use port", port)
	os.environ['MASTER_PORT'] = str(port)
	args.gpus = torch.cuda.device_count()
	print("Use gpus ", list(range(args.gpus)),"Total Nodes ",args.nodes)
	args.world_size = args.gpus * args.nodes
	mp.spawn(setup, nprocs=args.gpus, args=(args,))





def random_feat(feats):
	mask_feats = feats.copy()
	feat_mask = np.zeros(len(feats), dtype=np.float32)
	for i in range(len(feats)):
		prob = random.random()
		# mask token with probability
		if prob < args.obj_mask_rate:
			prob /= args.obj_mask_rate

			# 80% randomly change token to zero feat
			if prob < 0.8:
				mask_feats[i, :] = 0.

			# 10% randomly change token to random feat
			elif prob < 0.9:
				mask_feats[i, :] = train_tuple.torchdset.random_feat()
			# -> rest 10% randomly keep current feat

			# Need to predict this feat
			feat_mask[i] = 1.

	return mask_feats, feat_mask

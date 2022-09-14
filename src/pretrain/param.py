# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    parser_pretrain = argparse.ArgumentParser()

    # Data Splits
    parser_pretrain.add_argument("--train", default='train')
    parser_pretrain.add_argument("--valid", default='valid')
    parser_pretrain.add_argument("--test", default=None)

    # Training Hyper-parameters
    parser_pretrain.add_argument('--batchSize', dest='batch_size', type=int, default=256)
    parser_pretrain.add_argument('--optim', default='bert')
    parser_pretrain.add_argument('--lr', type=float, default=1e-4)
    parser_pretrain.add_argument('--epochs', type=int, default=10)
    parser_pretrain.add_argument('--dropout', type=float, default=0.1)
    parser_pretrain.add_argument('--seed', type=int, default=9595, help='random seed')

    # Debugging
    parser_pretrain.add_argument('--output', type=str, default='snap/test')
    parser_pretrain.add_argument("--fast", action='store_const', default=False, const=True)
    parser_pretrain.add_argument("--tiny", action='store_const', default=False, const=True)
    parser_pretrain.add_argument("--tqdm", action='store_const', default=False, const=True)

    # Model Loading
    parser_pretrain.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser_pretrain.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
                        help='Load the pre-trained LXMERT model.')
    parser_pretrain.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default=None,
                        help='Load the pre-trained LXMERT model with QA answer head.')
    parser_pretrain.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                             'the model would be trained from scratch. If --fromScratch is'
                             ' not specified, the model would load BERT-pre-trained weights by'
                             ' default. ')

    # Optimization
    parser_pretrain.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    # parser_pretrain.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
    parser_pretrain.add_argument("--xlayers", default=5, type=int, help='Number of CROSS-modality layers.')
    # parser_pretrain.add_argument("--rlayers", default=5, type=int, help='Number of object Relationship layers.')

    # LXMERT Pre-training Config
    parser_pretrain.add_argument("--taskMatched", dest='task_matched', action='store_const', default=False, const=True)
    parser_pretrain.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const', default=False, const=True)
    parser_pretrain.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser_pretrain.add_argument("--freezePretrained", dest = 'freeze_pretrained',default = 0, type = int)
    parser_pretrain.add_argument("--gradAccumulation", dest = 'grad_accumulation',default = 0, type = int)

    #From vokenization
    parser_pretrain.add_argument(
        "--model_name_or_path", type=str, help="The model architecture to be trained or fine-tuned.",)
    parser_pretrain.add_argument(
        "--config_name", default=None, type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path.",)
    parser_pretrain.add_argument(
        "--tokenizer_name", default=None, type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",)
    
    parser_pretrain.add_argument(
        "--max_seq_len", default=-1, type=int,
        help="Optional input sequence length after tokenization."
             "The training dataset will be truncated in block of this size for training."
             "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser_pretrain.add_argument(
        "--overwrite_cache", action="store_true",
        help="Overwrite the cached training and evaluation sets")

    parser_pretrain.add_argument(
        "--cache_dir", default=None, type=str,
        help="Overwrite the cached training and evaluation sets")
    
    # Training configuration
    parser_pretrain.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser_pretrain.add_argument("--numWorkers", dest='num_workers', default=0)
    parser_pretrain.add_argument("--nodes", type=int, default=1)
    parser_pretrain.add_argument("--nr", type=int, default=0)
    #parser_pretrain.add_argument("--gpus",default = "0", type = str)

    # Parse the arguments.
    args = parser_pretrain.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()

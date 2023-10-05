# %%
from datasets import load_dataset
import os
os.getcwd() 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
from test import *
warnings.filterwarnings("ignore")
from datasets import load_dataset,load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch_optimizer as optim
from transformers.optimization import Adafactor, AdafactorSchedule
import torch.backends.cudnn as cudnn
from utils import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
import logging
import sys
import transformers
import torch
import time
import argparse
from tqdm import tqdm
import string
from RoBERTa import *
from transformers import RobertaTokenizer
from transformers import AutoTokenizer

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser("main")

parser.add_argument('--valid_num_points', type=int,             default = 8,              help='validation data number')
parser.add_argument('--train_num_points', type=int,             default = 80,              help='train data number')
parser.add_argument('--model_name',       type=str,             default = 'roberta-base',   help='model name')
parser.add_argument('--max_length',       type=int,             default=128,                help='max_length')
parser.add_argument('--batch_size',       type=int,             default=8,                  help='Batch size')
parser.add_argument('--num_workers',      type=int,             default=0,                  help='num_workers')
parser.add_argument('--epochs',           type=int,             default=5000,                  help='num of epochs')
parser.add_argument('--lr',               type=float,           default=1e-5,               help='lr')
parser.add_argument('--gamma',            type=float,           default=1,                  help='lr*gamma after each test')

args = parser.parse_args(args=[])#(args=['--batch_size', '8',  '--no_cuda'])#used in ipynb
print(args)

dataset = load_dataset('glue', 'mnli')

# %%
train = dataset['train'][:args.train_num_points]
valid = dataset['train'][-args.valid_num_points:]

tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
#mnli
#The Multi-Genre Natural Language Inference Corpus is a crowdsourced collection of sentence pairs with textual entailment annotations. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral). The premise sentences are gathered from ten different sources, including transcribed speech, fiction, and government reports. The authors of the benchmark use the standard test set, for which they obtained private labels from the RTE authors, and evaluate on both the matched (in-domain) and mismatched (cross-domain) section. They also uses and recommend the SNLI corpus as 550k examples of auxiliary training data.

# %%
train_data = get_Dataset(train, tokenizer,max_length=args.max_length)
train_dataloader = DataLoader(train_data, sampler= SequentialSampler(train_data), 
                        batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)
valid_data = get_Dataset(valid, tokenizer,max_length=args.max_length)
valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), 
                        batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)

# %%
model = TextClassifier(args).to(device)
model.train(train_dataloader,train_dataloader,device)
    

# %%



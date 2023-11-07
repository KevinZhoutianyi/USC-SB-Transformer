# %%
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
import logging    # first of all import the module
from datetime import datetime

foldername = datetime.now().strftime('./logs/%Y_%m_%d_%H_%M_%S')
# Create the folder if it doesn't exist
if not os.path.exists(foldername):
    os.makedirs(foldername)
# Define the log filename inside the newly created folder
logfilename = os.path.join(foldername, 'logfile.log')

logging.getLogger().setLevel(logging.INFO)

logging.basicConfig(filename=logfilename, filemode='w', format='%(asctime)s %(levelname)s - %(funcName)s: %(message)s')
handle = "root"
logger = logging.getLogger(handle)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser("main")

parser.add_argument('--train_num_points', type=int,             default = 10000,            help='train data number')
parser.add_argument('--valid_num_points', type=int,             default = 1000,               help='validation data number')
parser.add_argument('--report_num_points',type=int,             default = 500,              help='report number')
parser.add_argument('--model_name',       type=str,             default = 'roberta-base',   help='model name')
parser.add_argument('--max_length',       type=int,             default=64,                 help='max_length')
parser.add_argument('--num_labels',       type=int,             default=3,                 help='num_labels')
parser.add_argument('--batch_size',       type=int,             default=32,                help='Batch size')
parser.add_argument('--num_workers',      type=int,             default=0,                  help='num_workers')
parser.add_argument('--replace_size',     type=int,             default=3,                  help='to test sensitivity, we need to replance each word by x random words from vocab, here we specify the x')
parser.add_argument('--epochs',           type=int,             default=5,                  help='num of epochs')
parser.add_argument('--lr',               type=float,           default=1e-5,               help='lr')
parser.add_argument('--gamma',            type=float,           default=1,                  help='lr*gamma after each test')

args = parser.parse_args()#(args=['--batch_size', '8',  '--no_cuda'])#used in ipynb
logger.info(f'args:{args}')

dataset = load_dataset('glue', 'mnli')

logger.info('\n Property of dataset:')
logger.info(f'train set size: {len(dataset["train"])}')
logger.info(f'validation_mismatched set size: {len(dataset["validation_matched"])}')
logger.info(f'test_matched set size: {len(dataset["test_matched"])}')
logger.info(f'test_mismatched set size: {len(dataset["test_mismatched"])}')
# %%
# %%



train = dataset['train'][:args.train_num_points]
valid = dataset['validation_matched'][-args.valid_num_points:]
replaced = replaced_data(valid, args.replace_size) 


if args.model_name=='roberta-scratch':
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
#mnli
#The Multi-Genre Natural Language Inference Corpus is a crowdsourced collection of sentence pairs with textual entailment annotations. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral). The premise sentences are gathered from ten different sources, including transcribed speech, fiction, and government reports. The authors of the benchmark use the standard test set, for which they obtained private labels from the RTE authors, and evaluate on both the matched (in-domain) and mismatched (cross-domain) section. They also uses and recommend the SNLI corpus as 550k examples of auxiliary training data.

# %%
train_data = get_Dataset(train, tokenizer,max_length=args.max_length)
train_dataloader = DataLoader(train_data, sampler= SequentialSampler(train_data), 
                        batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)
valid_data = get_Dataset(valid, tokenizer,max_length=args.max_length)
valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), 
                        batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)
replaced_data = get_Replaced_Dataset(replaced, tokenizer, max_length = args.max_length)
replaced_dataloader = DataLoader(replaced_data, sampler=SequentialSampler(replaced_data), 
                        batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)

# %%

model = TextClassifier(args,foldername).to(device)
model.train(train_dataloader,valid_dataloader,replaced_dataloader,device)



# %%




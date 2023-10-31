# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import random
import os
import numpy as np

from torch.autograd import Variable
from utils import *

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import logging    # first of all import the module
from datetime import datetime
from transformers import RobertaConfig, RobertaForSequenceClassification

# %%
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(1)
def get_savefilename(foldername,filename):
    logfilename = os.path.join(foldername, filename)
    return logfilename
# %%
# EXAMPLE
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# model = RobertaModel.from_pretrained('roberta-base')
# text_batch = ["I love Pixar.", "I don't care for Pixar."]
# encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
# input_ids = encoding['input_ids']
# attention_mask = encoding['attention_mask']
# labels = torch.tensor([1,0]).unsqueeze(0)
# outputs = model(input_ids, attention_mask=attention_mask)
# logging.info(outputs.last_hidden_state.shape) #number of data, sequence length, hidden state size
# logging.info(outputs.last_hidden_state[:,0,:].shape) # connect this to FC layer (768,2) to do binary classfication)

# %%
class TextClassifier(nn.Module):
    def __init__(self, args, foldername ):
        super(TextClassifier, self).__init__()
        if args.model_name=='roberta-scratch':
            configuration = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=3).config
            self.model = RobertaForSequenceClassification(configuration)
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            self.model =AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels) 

        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.fc = nn.Linear(self.model.config.hidden_size, 3)  # FC layer
        self.criterion = torch.nn.CrossEntropyLoss()#ignore_index=0
        self.softmax = torch.nn.Softmax(dim=1)
        self.optimizer = torch.optim.Adam(self.model.parameters(),  lr= args.lr ,  betas=(0, 0.9)  )
        self.scheduler =torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=args.gamma)
        self.epochs = args.epochs
        self.report_number = args.report_num_points
        self.replace_size = args.replace_size
        self.sensitivity_2dlist_report = []
        self.train_loss_report = []
        self.validation_loss_report = []
        self.log_foldername = foldername
        self.validation_acc_report = []

    def forward(self,x,x_att):
        output = self.model(x,attention_mask=x_att)
        logits = output.logits
        logits = self.softmax(logits) #not sure whether we need this line to compute loss
        return logits
    
    def loss(self,logits,labels):
        loss = self.criterion(logits, labels)
        return loss
    def validation(self,valid_dataloader, replaced_dataloader, epoch,device,save=False):
        logger =  logging.getLogger('training')
        self.model.eval()
        all_acc = []
        all_loss = []
        with torch.no_grad():
            for step,batch in enumerate(valid_dataloader):
                input_ids = Variable(batch[0], requires_grad=False).to(device, non_blocking=False)
                input_attn = Variable(batch[1], requires_grad=False).to(device, non_blocking=False)
                labels = Variable(batch[2], requires_grad=False).to(device, non_blocking=False)   
                logits = self.forward(input_ids,input_attn)
                loss = self.loss(logits,labels)
                predict = torch.argmax(logits,dim=1)
                accuracy = accuracy_score(labels, predict)
                all_loss.append(loss)
                all_acc.append(accuracy)
        sensitivity = word_label_sensitivity(replaced_dataloader, valid_dataloader, self, device, self.replace_size)
        self.sensitivity_2dlist_report.append(sensitivity)
        self.validation_loss_report.append(sum(all_loss)    / len(all_loss) )
        self.validation_acc_report.append(sum(all_acc) / len(all_acc))
        logger.info(f"sensitivity: {sensitivity}")
        logger.info(f"sensitivity mean: {sum(sensitivity) / len(sensitivity) :.4f}")
        logger.info(f"Validation Loss:  {sum(all_loss)    / len(all_loss)    :.4f}")
        logger.info(f"Validation Accuracy: {sum(all_acc) / len(all_acc):.4f}")
        if save:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'acc': sum(all_acc) / len(all_acc),}, './checkpoints/'+str(epoch)+'_model.pt')

    def train(self,train_dataloader,valid_dataloader,replaced_dataloader, device):
        logger =  logging.getLogger('training')
        self.validation(valid_dataloader,replaced_dataloader, 0,device,False)
        self.model.train()
        report_counter  = 0
        total_counter = 0
        epoch_train_loss = 0.0
        batch_train_loss = 0.0
        batch_loss_counter = 0
        for epoch in range(self.epochs):
            #train
            for step,batch in enumerate(train_dataloader):
                report_counter += len(batch[0])
                total_counter += len(batch[0])
                input_ids = Variable(batch[0], requires_grad=False).to(device, non_blocking=False)
                input_attn = Variable(batch[1], requires_grad=False).to(device, non_blocking=False)
                labels = Variable(batch[2], requires_grad=False).to(device, non_blocking=False)    
                self.optimizer.zero_grad()
                logits = self.forward(input_ids,input_attn)
                loss = self.loss(logits,labels)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
                batch_train_loss += loss.item()
                loss_counter += 1


                if report_counter > self.report_number:
                    logger.info(f"Average training loss for each batch:{batch_train_loss/(batch_loss_counter)} ")
                    self.train_loss_report.append(batch_train_loss/(batch_loss_counter))
                    report_counter  = 0
                    batch_loss_counter = 0
                    batch_train_loss = 0
                    logger.info(f'======total trained data counter: {total_counter}======')
                    logger.info(f'report_counter hit { self.report_number}, will do validation')
                    # Validation
                    self.validation(valid_dataloader, replaced_dataloader,epoch,device,False)


                    temp1 = torch.tensor(self.sensitivity_2dlist_report, device = 'cpu')
                    temp2= torch.tensor(self.train_loss_report, device = 'cpu')
                    temp3 = torch.tensor(self.validation_loss_report, device = 'cpu')
                    temp4 = torch.tensor(self.validation_acc_report, device = 'cpu')
                    np.save(get_savefilename(self.log_foldername,'sensitivity'), temp1)
                    np.save(get_savefilename(self.log_foldername,'trainloss'),temp2)
                    np.save(get_savefilename(self.log_foldername,'validationloss'), temp3)
                    np.save(get_savefilename(self.log_foldername,'validationacc'), temp4)


            logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss/(step+1):.4f}")

            
   




# %%

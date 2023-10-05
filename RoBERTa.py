# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import random
import os
import numpy as np

from torch.autograd import Variable
from utils import accuracy_score

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

# %%
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(1)

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
# print(outputs.last_hidden_state.shape) #number of data, sequence length, hidden state size
# print(outputs.last_hidden_state[:,0,:].shape) # connect this to FC layer (768,2) to do binary classfication)

# %%
class TextClassifier(nn.Module):
    def __init__(self, args, model_name='roberta-base', num_labels=3 ):
        super(TextClassifier, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model =AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels) 
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.fc = nn.Linear(self.model.config.hidden_size, 3)  # FC layer
        self.criterion = torch.nn.CrossEntropyLoss()#ignore_index=0
        self.softmax = torch.nn.Softmax(dim=1)
        self.optimizer = torch.optim.Adam(self.model.parameters(),  lr= args.lr ,  betas=(0, 0.9)  )
        self.scheduler =torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=args.gamma)
        self.epochs = args.epochs

    def forward(self,x,x_att):
        output = self.model(x,attention_mask=x_att)
        logits = output.logits
        logits = self.softmax(logits) #not sure whether we need this line to compute loss
        return logits
    
    def loss(self,logits,labels):
        loss = self.criterion(logits, labels)
        return loss

    def train(self,train_dataloader,valid_dataloader,device):
        self.model.train()
        for epoch in range(self.epochs):
            train_loss = 0.0
            for step,batch in enumerate(train_dataloader):
                input_ids = Variable(batch[0], requires_grad=False).to(device, non_blocking=False)
                input_attn = Variable(batch[1], requires_grad=False).to(device, non_blocking=False)
                labels = Variable(batch[2], requires_grad=False).to(device, non_blocking=False)    
                self.optimizer.zero_grad()
                logits = self.forward(input_ids,input_attn)
                loss = self.loss(logits,labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss/(step+1):.4f}")

            # Validation
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


            print(f"Validation Loss: {sum(all_loss) / len(all_loss) :.4f}")
            print(f"Validation Accuracy: {sum(all_acc) / len(all_acc):.4f}")
            self.model.train()

   

# classifier = RoBERTaTextClassifier("roberta-base", num_labels=3)
# train_texts, train_labels, validation_texts, validation_labels = ... multiNLP
# classifier.train(train_texts, train_labels, validation_texts, validation_labels)
# test_texts = ...
# predictions = classifier.predict(test_texts)


# %%




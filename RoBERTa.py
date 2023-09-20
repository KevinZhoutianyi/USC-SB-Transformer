# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import random
import os
import numpy as np

from utils import accuracy_score

from transformers import RobertaTokenizer, RobertaModel

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
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
text_batch = ["I love Pixar.", "I don't care for Pixar."]
encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
labels = torch.tensor([1,0]).unsqueeze(0)
outputs = model(input_ids, attention_mask=attention_mask)
print(outputs.last_hidden_state.shape) #number of data, sequence length, hidden state size
print(outputs.last_hidden_state[:,0,:].shape) # connect this to FC layer (768,2) to do binary classfication)

# %%
class TextClassifier(nn.Module):
    def __init__(self, model_name='roberta-base', num_groups=3, batch_size=32, max_length=128):
        super(TextClassifier, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc = nn.Linear(self.model.config.hidden_size, 3)  # FC layer


    def forward(self,x,x_att):
        output = self.model(x,attention_mask=x_att)
        hidden_states = output.last_hidden_state[:,0,:] #[cls] tokens
        logits = self.fc(hidden_states)
        return logits
    
    def loss(self,logits,labels):
        loss = F.cross_entropy(logits, labels)
        return loss

    def train(self, train_texts, train_labels, validation_texts, validation_labels, epochs=3):
        train_inputs, train_labels = self.preprocess_text(train_texts, train_labels)
        validation_inputs, validation_labels = self.preprocess_text(validation_texts, validation_labels)

        train_dataset = torch.utils.data.TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
        validation_dataset = torch.utils.data.TensorDataset(validation_inputs['input_ids'], validation_inputs['attention_mask'], validation_labels)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size)

        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        for epoch in range(epochs):
            train_loss = 0.0
            for batch in train_loader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, label_ids = batch

                optimizer.zero_grad()
                outputs = self.forward(input_ids, attention_mask=attention_mask)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

            # Validation
            self.model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in validation_loader:
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, attention_mask, label_ids = batch
                    outputs = self.forward(input_ids, attention_mask=attention_mask)

                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(label_ids.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            print(f"Validation Accuracy: {accuracy:.4f}")
            self.model.train()

    def predict(self, text_list):
        inputs, _ = self.preprocess_text(text_list, labels=None)
        inputs = inputs.to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.forward(inputs['input_ids'],inputs['attention_mask'])
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        return preds

# classifier = RoBERTaTextClassifier("roberta-base", num_labels=3)
# train_texts, train_labels, validation_texts, validation_labels = ... multiNLP
# classifier.train(train_texts, train_labels, validation_texts, validation_labels)
# test_texts = ...
# predictions = classifier.predict(test_texts)


# %%




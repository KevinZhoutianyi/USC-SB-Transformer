
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch

def accuracy_score(all_labels, all_preds):
    correct_count = 0
    all_labels = all_labels.squeeze().tolist()
    all_preds = all_preds.squeeze().tolist()
    for label, pred in zip(all_labels, all_preds):
        if label == pred:
            correct_count += 1
    
    return correct_count/len(all_labels)



def tokenize(text_data, tokenizer, max_length, padding = True):
    
    encoding = tokenizer(text_data, return_tensors='pt', padding=padding, truncation = True, max_length = max_length)
    # encoding = tokenizer(premise, hypothesis, ...)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    return input_ids, attention_mask


def get_Dataset(dataset, tokenizer,max_length):
    premise_hypothesis = ['<\s>'+premise+'<\s>'+hypothesis for premise,hypothesis in zip(dataset['premise'],dataset['hypothesis'])]
    label  = torch.Tensor(dataset['label'])
    token_ids, token_attn = tokenize(premise_hypothesis, tokenizer, max_length = max_length)
    train_data = TensorDataset(token_ids, token_attn, label)
    return train_data 



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



def tokenize(premise, hypothesis,tokenizer, max_length, padding = True):
    # tokenizer.build_inputs_with_special_tokens(premise,hyp)
    # print('premise',tokenizer(text=premise, return_tensors='pt', add_special_tokens=True, padding=padding, truncation = True, max_length = max_length)[0])
    # print('hyper',tokenizer(text=hypothesis, return_tensors='pt', add_special_tokens=True, padding=padding, truncation = True, max_length = max_length)[0])
    encoding = tokenizer(text=premise,text_pair = hypothesis, return_tensors='pt', add_special_tokens=True, padding=padding, truncation = True, max_length = max_length)
    # encoding = tokenizer(premise, hypothesis, ...)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    # print('input',input_ids[0])
    # print('att',attention_mask[0])
    return input_ids, attention_mask


def get_Dataset(dataset, tokenizer,max_length):
    premise , hypothesis = dataset['premise'],dataset['hypothesis']
    
    label  = torch.Tensor(dataset['label'])
    label = label.type(torch.LongTensor)  
    token_ids, token_attn = tokenize(premise,hypothesis, tokenizer, max_length = max_length)
    train_data = TensorDataset(token_ids, token_attn, label)
    return train_data 


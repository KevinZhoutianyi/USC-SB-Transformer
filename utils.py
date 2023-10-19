from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
import re
from transformers import RobertaTokenizer
import logging

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
    # logger.info('premise',tokenizer(text=premise, return_tensors='pt', add_special_tokens=True, padding=padding, truncation = True, max_length = max_length)[0])
    # logger.info('hyper',tokenizer(text=hypothesis, return_tensors='pt', add_special_tokens=True, padding=padding, truncation = True, max_length = max_length)[0])
    encoding = tokenizer(text=premise,text_pair = hypothesis, return_tensors='pt', add_special_tokens=True, padding=padding, truncation = True, max_length = max_length)
    # encoding = tokenizer(premise, hypothesis, ...)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    # logger.info('input',input_ids[0])
    # logger.info('att',attention_mask[0])
    return input_ids, attention_mask


def get_Dataset(dataset, tokenizer,max_length):
    premise , hypothesis = dataset['premise'],dataset['hypothesis']
    
    label  = torch.Tensor(dataset['label'])
    label = label.type(torch.LongTensor)  
    
    token_ids, token_attn = tokenize(premise,hypothesis, tokenizer, max_length = max_length)
    train_data = TensorDataset(token_ids, token_attn, label)
    return train_data 

#unsure if we should considering casing, punctuation when replacing — original: Product and geography are what make cream skimming work., modified: parks and geography are what make cream skimming work.

def get_vocab(dataset):
    vocab = []
    for premise in dataset["premise"]:
        for word in premise.replace(".", " ").split():
            if word not in vocab:
                vocab.append(word)
    for hypothesis in dataset["hypothesis"]:
        for word in hypothesis.replace(".", " ").split():
            if word not in vocab:
                vocab.append(word)
    return vocab

def word_label_sensitivity(dataset, n, model, device): #Tianyi: this function runs slow, we may need to implement the batch version to speed it up
    #can be called for matched, mismatched validation sets — use untokenized data

    logger =  logging.getLogger('sensitivity')
    vocab = get_vocab(dataset)
    labels = dataset["label"]
    premise = dataset["premise"]
    index = 0
    sensitivity = []
    for hypothesis in dataset["hypothesis"]:
        # logger.info(f"-----hypothesis:{hypothesis}")

        input_ids, att_mask = tokenize(premise[index], hypothesis, RobertaTokenizer.from_pretrained('roberta-base'), 512)
        input_ids, att_mask = input_ids.to(device), att_mask.to(device)
        with torch.no_grad():
            output = model.forward(input_ids, att_mask)
            original_label =torch.argmax(output, dim=1)  

        # logger.info("original label:", original_label)
        temp_hypothesis = hypothesis[:]
        label_change_per_word = 0
        len = 0
        for word in hypothesis.replace(".", " ").split():
            len += 1
            label_change = 0
            for i in range(n):
                #may need to seed this so its truly random 
                replacement = random.choice(vocab)
                hypothesis_replaced = re.sub(r'\b' + re.escape(word) + r'\b', replacement, temp_hypothesis)
                # logger.info("one-word-replaced hypothesis:", hypothesis_replaced)
                #i can parameterize the model and max length arguments as well if we think it is necessary 
                # Tianyi: Yes, we need to parameterize the max length, so that this function only replace the i-th word where i < max_length
                # logger.info("premise:", premise[index])
                input_ids, att_mask = tokenize(premise[index], hypothesis_replaced, RobertaTokenizer.from_pretrained('roberta-base'), 512)
                input_ids, att_mask = input_ids.to(device), att_mask.to(device)
                # data = torch.tensor(data)
                with torch.no_grad():
                    output = model.forward(input_ids, att_mask)
                    pred_label =torch.argmax(output, dim=1)  
                    # logger.info("pred:", pred_label)
                    if original_label != pred_label:
                        label_change += 1
            # logger.info("Label changed counter:", label_change)
            label_change /= n
            label_change_per_word += label_change
            # logger.info('-------------------')
        # logger.info(f"length:{len}")
        # logger.info(f"label_change_per_word:{label_change_per_word}")  
        sensitivity.append(label_change_per_word/len)
        index += 1
    return sensitivity 


# if __name__ == "__main__":
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
import re
from transformers import RobertaTokenizer
import logging
from torch.utils.data import SequentialSampler

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

def get_Replaced_Dataset(dataset, tokenizer,max_length):
    logger =  logging.getLogger('replaced_dataloader')
    token_id_arr = []
    token_attn_arr = []
    seq_len  = torch.Tensor(dataset['sequence_length'])
    seq_len = seq_len.type(torch.LongTensor) 
    for i in range(len(dataset['premise'])): 
        token_ids, token_attn = tokenize(dataset["premise"][i],dataset["hypothesis"][i], tokenizer, max_length = max_length, padding = "max_length")
        token_id_arr.append(token_ids)
        token_attn_arr.append(token_attn)
    #pad each tensor in tensor_id, tensor_attn to the max seq_len*n tensor
    logger.debug(f"token_id_arr length:{len(token_id_arr)}")
    logger.debug(f"token_id_arr[0] length:{len(token_id_arr[0])}")
    logger.debug(f"token_id_arr[0][0] length:{len(token_id_arr[0][0])}")
    logger.debug(f"token_attn_arr length:{len(token_attn_arr)}")
    logger.debug(f"seq_len[0]:{seq_len[0]}")

    max_id = max(tensor.shape[0] for tensor in token_id_arr) #find the max replacesize*seq_length
    logger.info(f"max_id:{max_id},max_length:{max_length}")
    padded_id = [torch.cat((tensor, torch.zeros(max_id - tensor.shape[0], tensor.shape[1], dtype=torch.int32))) for tensor in token_id_arr]
    padded_attn = [torch.cat((tensor, torch.zeros(max_id - tensor.shape[0], tensor.shape[1], dtype=torch.int32))) for tensor in token_attn_arr]
    
    tensor_id = torch.stack(padded_id, dim = 0)
    tensor_attn = torch.stack(padded_attn, dim = 0)
    logger.debug(f"tensor_id.shape:{tensor_id.shape},tensor_attn.shape:{tensor_attn.shape}.seq_len:{seq_len.shape}")# get_Replaced_Dataset: tensor_id.shape:torch.Size([32, 66, 64]),tensor_attn.shape:torch.Size([32, 66, 64]).seq_len:torch.Size([32])
    replaced_data = TensorDataset(tensor_id, tensor_attn, seq_len)
    return replaced_data

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

def replaced_data(dataset, n):
    vocab = get_vocab(dataset)
    labels = dataset["label"]
    premise = dataset["premise"]
    index = 0
    replaced_hypothesis = []
    replaced_premise = []
    sequence_length = []
    for hypothesis in dataset["hypothesis"]:
        len = 0
        temp_hypothesis = hypothesis[:]
        replaced_per_sentence = []
        replaced_labels_per_sentence = []
        replaced_premise_per_sentence = []
        for word in hypothesis.replace(".", " ").split():
            len += 1
            for i in range(n):
                replacement = random.choice(vocab)
                hypothesis_replaced = re.sub(r'\b' + re.escape(word) + r'\b', replacement, temp_hypothesis)
                replaced_per_sentence.append(hypothesis_replaced)
                replaced_premise_per_sentence.append(premise[index])
        index += 1
        sequence_length.append(len)
        replaced_premise.append(replaced_premise_per_sentence)
        replaced_hypothesis.append(replaced_per_sentence)
    data_dict = {
        "premise": replaced_premise,
        "hypothesis": replaced_hypothesis,
        "sequence_length": sequence_length
    }
    return data_dict

"""
def word_label_sensitivity(replaced_dataloader, original_dataloader, n, model, device): #Tianyi: this function runs slow, we may need to implement the batch version to speed it up
    #can be called for matched, mismatched validation sets — use untokenized data

    logger =  logging.getLogger('sensitivity')
    index = 0
    sensitivity = []
    label_change_per_word = 0
    len = 0
    for replaced_batch, original_batch in zip(replaced_dataloader, original_dataloader):
        # logger.info(f"-----hypothesis:{hypothesis}")
        input_ids, att_masks = original_batch[0].to(device), original_batch[1].to(device)
        with torch.no_grad():
            output = model.forward(input_ids, att_mask)
            original_label = torch.argmax(output, dim=1)  

        # logger.info("original label:", original_label)

        #batch_size = n
        input_ids, att_mask = replaced_batch[0].to(device), replaced_batch[1].to(device)
        with torch.no_grad():
            output = model.forward(input_ids, att_mask)
            pred_label = torch.argmax(output, dim=1)
            if original_label != pred_label:
                label_change += 1
                
        label_change /= n
        label_change_per_word += label_change 
        
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
"""


def word_label_sensitivity(replaced_dataloader, original_dataloader, model, device, n):
    #datasets are now different sizes, replaced dataset is n*word*original_size
    logger = logging.getLogger('compute_sensitivity')
    sensitivity = []
    label_change = 0
    label_change_per_word = 0
    for replaced_batch, original_batch in zip(replaced_dataloader, original_dataloader):
        '''
        remove padding for replaced_batch
        def contains_only_zeros(tensor):
            return not torch.any(tensor != 0)

        
        rep_input_ids = [[tensor for tensor in row if not contains_only_zeros(tensor)] for row in replaced_batch[0]]
        rep_input_ids = torch.stack([torch.stack(row) for row in rep_input_ids])
        
        # Filter out tensors that contain only zeros in padded_attn
        rep_att_masks = [[tensor for tensor in row if not contains_only_zeros(tensor)] for row in replaced_batch[1]]
        rep_att_masks = torch.stack([torch.stack(row) for row in rep_att_masks])
        '''
        
        seq_len = replaced_batch[2] 
            
        logger.debug(f"replaced_batch.shape:{replaced_batch[0].shape}, replaced_batch.shape:{replaced_batch[1].shape}")
        batchsize,word_len_times_replace_size,_ = replaced_batch[0].shape
        replaced_input_ids =    replaced_batch[0]#.reshape(batchsize*word_len_times_replace_size,-1)
        replaced_input_att =    replaced_batch[1]#.reshape(batchsize*word_len_times_replace_size,-1)
        input_ids, att_masks =  replaced_input_ids.to(device), replaced_input_att.to(device) #
        temp_list = []
        mask_list = []
        with torch.no_grad():
            for i in range(batchsize):#the shape of the input to the model is (word_len_times_replace_size,seq_length), if we don't iterate with the BS the input size is too large that will casue oom error
                output = model.forward(input_ids[i], att_masks[i])
                pred = torch.argmax(output, dim=1)
                temp_list.append(pred)
                mask = torch.zeros(pred.shape[0], dtype=torch.uint8)
                mask[:seq_len[i]*n] = 1
                mask_list.append(mask)
        #finish calculation
        mask = torch.cat(mask_list).to(device) 
        replaced_pred = torch.cat(temp_list, dim=0)
        logger.debug(f"replaced_pred.shape:{replaced_pred.shape}")
        logger.debug(f"mask.shape:{mask.shape}")


        ori_input_ids =    original_batch[0]
        ori_input_att =    original_batch[1]
        input_ids, att_masks =  ori_input_ids.to(device), ori_input_att.to(device)
        with torch.no_grad():
            output = model.forward(input_ids, att_masks)
            ori_pred = torch.argmax(output, dim=1)
        #finish calculation
        logger.debug(f"ori_pred.shape:{ori_pred.shape}")
        repeat_ori_pred = ori_pred.repeat_interleave(word_len_times_replace_size)
        logger.debug(f"repeat_ori_pred.shape:{repeat_ori_pred.shape}")


        wrong = torch.sum(  (replaced_pred != repeat_ori_pred)*mask ).item()
        total = torch.sum(mask)
        sens = wrong / total
        sensitivity.append(sens)
    logger.debug(f"sensitivity:{sensitivity}")

    
    return sensitivity






    
# if __name__ == "__main__":
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
import re
from transformers import RobertaTokenizer
import logging
from torch.utils.data import SequentialSampler
import string
from collections import Counter

def accuracy_score(all_labels, all_preds):
    correct_count = 0
    all_labels = all_labels.squeeze().tolist()
    all_preds = all_preds.squeeze().tolist()
    for label, pred in zip(all_labels, all_preds):
        if label == pred:
            correct_count += 1
    
    return correct_count/len(all_labels)



def tokenize(premise, hypothesis,tokenizer, model, max_length, padding = True):
    # tokenizer.build_inputs_with_special_tokens(premise,hyp)
    # logger.info('premise',tokenizer(text=premise, return_tensors='pt', add_special_tokens=True, padding=padding, truncation = True, max_length = max_length)[0])
    # logger.info('hyper',tokenizer(text=hypothesis, return_tensors='pt', add_special_tokens=True, padding=padding, truncation = True, max_length = max_length)[0])
    encoding = []
    input_ids = []
    attention_mask = []
    if (False):#we only use the roberta tokenizer for now
        preprocessed_sentences = [
        re.sub(r"[^\x00-\x7F]+", " ", re.sub('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]', ' ', f"{p} {h}").lower())
        for p, h in zip(premise, hypothesis)
        ]
        counts = Counter()
        for sentence in preprocessed_sentences:
            words = sentence.split()  # Split the sentence into words
            counts.update(words)
            
        vocab2index = {word: index+2 for index, (word, _) in enumerate(counts.items())}
        vocab2index[""] = 0
        vocab2index["UNK"] = 1

        encoding = [tokenizer(sentence) for sentence in preprocessed_sentences]
        enc1 = [torch.tensor([vocab2index.get(word, vocab2index["UNK"]) for word in sentence]) for sentence in encoding]
        enc2d = torch.nn.utils.rnn.pad_sequence(enc1, batch_first=True, padding_value=vocab2index["UNK"])
        input_ids = enc2d
        attention_mask = torch.zeros_like(enc2d)
    else:
        encoding = tokenizer(text=premise,text_pair = hypothesis, return_tensors='pt', add_special_tokens=True, padding=padding, truncation = True, max_length = max_length)
    # encoding = tokenizer(premise, hypothesis, ...)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
    # logger.info('input',input_ids[0])
    # logger.info('att',attention_mask[0])
    return input_ids, attention_mask


def get_Dataset(dataset, tokenizer, model, max_length):
    premise , hypothesis = dataset['premise'],dataset['hypothesis']
    label  = torch.Tensor(dataset['label'])
    label = label.type(torch.LongTensor)  
    token_ids, token_attn = tokenize(premise,hypothesis, tokenizer, model, max_length = max_length)
    train_data = TensorDataset(token_ids, token_attn, label)
    return train_data 

def get_Replaced_Dataset(dataset, tokenizer,model, max_length):
    logger =  logging.getLogger('replaced_dataloader')
    token_id_arr = []
    token_attn_arr = []
    seq_len  = torch.Tensor(dataset['sequence_length'])
    seq_len = seq_len.type(torch.LongTensor) 
    for i in range(len(dataset['premise'])): 
        token_ids, token_attn = tokenize(dataset["premise"][i],dataset["hypothesis"][i], tokenizer, model, max_length = max_length, padding = "max_length")
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

def embedding_label_sensitivity(validation_dataloader,model,embedding,device,n,variance):
    #1. we need to add noise to each token embedding 
    #2. for each sentence, we first get the origin prediction, and repeat by seqlen*n
    #TODO: is max_len to short? for this dataset

    
    logger = logging.getLogger('sensitivity')
    sensitivity = []
    for batch in validation_dataloader:
        input_ids =    batch[0]
        input_att =    batch[1]
        batchsize =    batch[0].shape[0]
        seqlen =    torch.sum(input_att,dim=1).to('cpu')
        maxlen =    input_ids.shape[1]
        
        logger.debug(f"input_ids.shape:{input_ids.shape}, input_att.shape:{input_att.shape}")
        logger.debug(f"input_ids[0]:{input_ids[0]}, input_att[0]:{input_att[0]}")
        logger.debug(f"batchsize:{batchsize}, seqlen:{seqlen}, maxlen:{maxlen}")
        input_ids, input_att =  input_ids.to(device), input_att.to(device)
        x_emb = embedding(input_ids) #batch size, seq len, embedding size
        logger.debug(f"x_emb.shape:{x_emb.shape}")
        origin__pred_list = []
        noised_pred_list = []
        with torch.no_grad():
            for i in range(batchsize): #for each sentence
                origin_output = model.forward_withembedding(x_emb[i].unsqueeze(dim=0),input_att[i].unsqueeze(dim=0))
                logger.debug(f"origin_output.shape:{origin_output.shape}")
                pred = torch.argmax(origin_output, dim=1)
                logger.debug(f"pred.shape:{pred.shape}")
                logger.debug(f"n*seqlen[i]:{n*seqlen[i]}")
                logger.debug(f"pred.repeat(n*seqlen[i]):{pred.repeat(n*seqlen[i])}")
                origin__pred_list.append(pred.repeat(n*seqlen[i]))
                logger.debug(f"origin__pred_list:{origin__pred_list}") 


                repeated_matrices = x_emb[i].repeat(n*seqlen[i],1,1)       #TODO: add noise 
                # #the dim of the embedding is max_len, embedding size
                # we first repeat the embedding n*seqlen[i] times
                # we generate a guassian matrix and add to it
                # do we add noise n times to each token emb?
                logger.debug(f"repeated_matrices:{repeated_matrices}") 
                mean = 0
                #https://arxiv.org/pdf/2211.12316v1.pdf set it to 15, is not it too large? as the embedding is range from -1 to 1 when initialization
                std_dev = variance ** 0.5
                guass =  torch.randn(n*seqlen[i],x_emb[i].shape[-1]) * std_dev + mean
                for k in range(seqlen[i]):
                    for j in range(n):  # n different noises for each token
                        noise = guass[k * n + j]
                        repeated_matrices[k * n + j, k, :] += noise.to(device)
                noised_embedding =repeated_matrices
                logger.debug(f"noised_embedding:{noised_embedding}") 
                logger.debug(f"noised_embedding.shape:{noised_embedding.shape}") 
                noised_output = model.forward_withembedding(noised_embedding,input_att[i].repeat(n*seqlen[i],1))
                logger.debug(f"noised_output.shape:{noised_output.shape}") 
                noised_pred = torch.argmax(noised_output, dim=1)
                logger.debug(f"noised_pred.shape:{noised_pred.shape}") 
                noised_pred_list.append(noised_pred) 
                logger.debug(f"noised_pred_list.shape:{noised_pred_list}") 
        
        origin__pred_list = torch.cat(origin__pred_list, dim=0)
        noised_pred_list = torch.cat(noised_pred_list, dim=0)
    
        logger.debug(f"origin__pred_list:{origin__pred_list}") 
        logger.debug(f"noised_pred_list:{noised_pred_list}") 

        wrong = torch.sum(  (origin__pred_list != noised_pred_list) ).item()
        logger.debug(f"wrong:{wrong}") 
        sens = wrong / len(origin__pred_list)
        logger.debug(f"sens:{sens}") 
        sensitivity.append(sens)
        logger.debug(f"sensitivity:{sensitivity}") 
    logger.debug(f"sensitivity:{sensitivity}")

    
    return sensitivity



def word_label_sensitivity(replaced_dataloader, original_dataloader, model, device, n):
    #datasets are now different sizes, replaced dataset is n*word*original_size
    logger = logging.getLogger('sensitivity')
    sensitivity = []
    label_change = 0
    label_change_per_word = 0
    for replaced_batch, original_batch in zip(replaced_dataloader, original_dataloader):
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
                # att_masks[i][:,1] = 1 #make the lenght of seq > 0, otherwise raise error for pack_padded_sequence in LSTM
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
        sensitivity.append(sens.item())
    logger.debug(f"sensitivity:{sensitivity}")

    
    return sensitivity



def get_model_size(model):#return model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


    

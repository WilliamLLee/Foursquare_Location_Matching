from copyreg import pickle
import sys
sys.path.append('F:/Foursquare_Location_Matching/')

import torch
from models.config.defaults import cfg
from models.LM import LM
from models.data import DATA
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import pickle as pkl


def validate(model, valid_data, threshold = None):
    """
    Validate the model and caculate the accuracy
    @param:
        valid_data: list of valid data
        threshold: threshold to filter the predictions
    """
    if threshold is None:
        threshold = cfg.MODEL.THRESHOLD
    # caculate the accuracy
    accuracy = 0
    # set batch generator
    batch_generator = batch_generator(valid_data, cfg.MODEL.BATCH_SIZE, shuffle=False)
    # validate
    with torch.no_grad():
        for idx, (text, match) in enumerate(batch_generator):
            # set input
            input = text
            # set target
            target = match
            # set output
            output = model(input)
            # set accuracy
            accuracy += accuracy(output, target, threshold)
    # set accuracy
    accuracy = accuracy / len(valid_data)
    # print accuracy
    print('Validation accuracy: {}'.format(accuracy))

def batch_generator(data, batch_size, shuffle=False):
    """
    Generate batch data
    @param:
        data: list of data
        batch_size: batch size
        shuffle: whether to shuffle
    @return:
        batch: batch data
    """
    
    data_t = TensorDataset(data[0], data[1])
    # set batch size
    batch_size = min(batch_size, len(data_t))
    # set batch generator
    batch_generator = DataLoader(   data_t, 
                                    batch_size=batch_size, 
                                    shuffle=shuffle) 
    # return batch generator
    return batch_generator

def train(cfg, model, train_data, device = 'cpu',  max_epochs = None, batch_size = None, learning_rate = None, weight_decay = None, model_path = None, save_every = None, valid_size = None):
    """
    Train the model
    @param:
        train_data: list of train data
        max_epoches: max number of epoches
        batch_size: batch size
        learning_rate: learning rate
        weight_decay: weight decay
        model_path: path to save the model
        save_every: save the model every x epoches
        valid_size: size of validation set
    """
    # set the device
    device = torch.device(device)
    # set the model to device
    model.to(device)
    # set the parameters
    if learning_rate is  None:
        learning_rate = cfg.MODEL.LR
    if batch_size is None:
        batch_size = cfg.MODEL.BATCH_SIZE
    if weight_decay is  None:
        weight_decay = cfg.MODEL.WEIGHT_DECAY
    if max_epochs is  None:
        max_epochs = cfg.MODEL.MAX_EPOCHS
    if save_every is  None:
        save_every = cfg.MODEL.SAVE_EVERY
    if model_path is  None:
        model_path = cfg.MODEL.MODEL_PATH
    if valid_size is  None:
        valid_size = cfg.MODEL.VALID_SIZE

    print('Training model...')
    # set optimizer
    print(model.parameters(), learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # set loss function
    loss_fn = torch.nn.BCELoss()
    # set scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=save_every, gamma=0.1)

    # validate the model
    total = len(train_data)
    valid_size = int(total * valid_size)
    print("valid_size :" , valid_size)
    valid_data = train_data[-valid_size:]
    train_data = train_data[:-valid_size]
    # train
    for epoch in range(max_epochs):
        # set train mode
        model.train()
        # set loss
        loss = 0
        # set batch generator
        bg = batch_generator(train_data, batch_size, shuffle=True)
        # train
        with tqdm(total = len(train_data)/batch_size) as pbar:
            for idx, (text, match) in enumerate(bg):
                # print(text, match)
                # set optimizer
                optimizer.zero_grad()
                # set input
                input = text
                # set target
                target = match
                # set output
                output = model(input)
                # set loss
                loss += loss_fn(output, target)
                # backward
                loss.backward()
                # optimize
                optimizer.step()
                # update pbar
                pbar.update(1)
        # set scheduler
        scheduler.step()
        # set loss
        loss = loss / len(train_data[0])
        # print loss
        print('Epoch: {}/{}, Loss: {}'.format(epoch + 1, max_epochs, loss))
        # set scheduler
        scheduler.step()
        # save model
        if (epoch + 1) % save_every == 0:
            # validate the model
            validate(model, valid_data)
            model.save_model(model_path+'_'+str(epoch+1))
        
    # save model
    model.save_model(model_path+'_final') 





def main(cfg):
    # load data
    # D = DATA(cfg)
    # load model
    model = LM(cfg)
    
    # data = D.get_pair_train_data_dict(auto_gen=True, rounds = 2, n_neighbors = 5, features = cfg.DATA.FEATURES)
    
    # tokenizer = AutoTokenizer.from_pretrained(cfg.DATA.TOKENIZER_PATH)

    # text = []
    # match = []
    # for  i in tqdm(range(len(data)//32)):
    #     text.append(tokenizer(
    #             data[i]['text'],
    #             add_special_tokens=True,
    #             padding = 'max_length',
    #             truncation = True,
    #             return_offsets_mapping = False,
    #             max_length = cfg.DATA.PREPROCESS_MAX_LEN,
    #             return_token_type_ids = False,
    #             return_attention_mask = False,
    #             return_tensors = 'pt',
    #         ))
    #     match.append(data[i]['match'])
    
    # print(data[0]['text'], data[0]['match'])
    # print(text[0], match[0])

    # train_dataset = {'text': text, 'match': match}
    # file= open('F:/Foursquare_Location_Matching/dataset/train_dataset_1_32.pkl', 'wb')
    # pkl.dump(train_dataset, file)
    
    train_dataset = pkl.load(open('F:/Foursquare_Location_Matching/dataset/train_dataset_4000.pkl', 'rb'))
    
    text, match = train_dataset['text'][:200], train_dataset['match'][:200]
    
    text_t = []
    match_t = []
    for i in tqdm(range(len(text))):
        text_t.append(text[i]['input_ids'].tolist())
        match_t.append(float(match[i]))
    print(len(text_t), len(match_t))
    text_t = torch.tensor(text_t).reshape(len(text_t), -1)
    match_t = torch.tensor(match_t).reshape(len(match_t), -1)

    data_t = TensorDataset(text_t, match_t)
    # train model
    train(cfg, model, data_t)

if __name__ == "__main__":
    main(cfg)
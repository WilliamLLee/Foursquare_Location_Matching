import sys
from tabnanny import verbose
sys.path.append('../')
import torch
from models.config.defaults import cfg
from models.LM import LM
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForMaskedLM
import numpy as np
import os
import pickle as pkl

def cal_accuracy(output, target, threshold):
    """
    Calculate the accuracy
    @param:
        output: model output
        target: target
        threshold: threshold to filter the predictions
    @return:
        accuracy: accuracy
    """
    # set accuracy
    accuracy = 0
    # set output
    output = output.detach().cpu().numpy()
    # set target
    target = target.detach().cpu().numpy()
    # filter the predictions
    output[output < threshold] = 0
    output[output >= threshold] = 1
    # set accuracy
    accuracy = np.sum(output == target)
    # return accuracy
    return accuracy

def validate(model, transformer, device, valid_data, threshold = None):
    """
    Validate the model and caculate the accuracy
    @param:
        model: model
        device: device
        valid_data: list of valid data
        threshold: threshold to filter the predictions
    """
    if threshold is None:
        threshold = cfg.MODEL.THRESHOLD
    # caculate the accuracy
    accuracy = 0
    # set batch generator
    bg = batch_generator(valid_data, cfg.MODEL.BATCH_SIZE, device,shuffle=False)
    # validate
    with torch.no_grad():
        with tqdm(total=len(valid_data[0])) as pbar:
            for idx, (text, match) in enumerate(bg):
                # set input
                input = text
                # set target
                target = match
                # set output
                output = transformer(input_ids = input)['hidden_states'][-1]
                output = torch.mean(output, dim=1)
                output = output.reshape(output.shape[0], -1)
                output = model(output)
                # set accuracy
                accuracy += cal_accuracy(output, target, threshold)
                # update progress bar
                pbar.update(cfg.MODEL.BATCH_SIZE)
    # set accuracy
    accuracy = accuracy / len(valid_data[0])
    # print accuracy
    print('Validation accuracy: {}'.format(accuracy))

def batch_generator(data, batch_size, device,shuffle=False):
    """
    Generate batch data
    @param:
        data: list of data
        batch_size: batch size
        device: device
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

def train(cfg, model, train_data, device = 'cpu',  max_epochs = None, batch_size = None, learning_rate = None, weight_decay = None, model_path = None, model_name = None, save_every = None, valid_size = None):
    """
    Train the model
    @param:
        train_data: list of train data
        max_epoches: max number of epoches
        batch_size: batch size
        learning_rate: learning rate
        weight_decay: weight decay
        model_path: path to save the model
        model_name: name of the model
        save_every: save the model every x epoches
        valid_size: size of validation set
    """
    # set the device
    if device.startwith("cuda"):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    # set the model to device
    model.to(device)
    transformer = AutoModelForMaskedLM.from_pretrained(
        cfg.MODEL.PRETRAINED_MODEL_PATH, 
        output_hidden_states=True, 
        output_attentions=True).to(device)

    for param in transformer.parameters():
        param.requires_grad = False
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
    if model_name is  None:
        model_name = cfg.MODEL.MODEL_NAME
    if valid_size is  None:
        valid_size = cfg.MODEL.VALID_SIZE

    print('Training model...')
    # set optimizer
    print(model.parameters(), learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # set loss function
    loss_fn = torch.nn.BCELoss()
    # set scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=save_every, gamma=0.1, verbose=True)
    # validate the model
    train_data = TensorDataset(train_data[0].to(device), train_data[1].to(device))
    total = len(train_data)
    valid_size = int(total * valid_size)
    print("valid_size :" , valid_size)
    valid_data = train_data[-valid_size:]
    train_data = train_data[:-valid_size]
    # set batch generator
    bg = batch_generator(train_data, batch_size, device, shuffle=True)
    # train
    for epoch in range(max_epochs):
        # set train mode
        model.train()
        # set loss
        total_loss = 0  
        # train
        with tqdm(total = len(train_data[0])) as pbar:
            for idx, (text, match) in enumerate(bg):
                # set optimizer
                optimizer.zero_grad()
                # set input
                input = text
                # set target
                target = match
                # set output
                output = transformer(input_ids = input)['hidden_states'][-1]
                output = torch.mean(output, dim=1)
                output = output.reshape(output.shape[0], -1)
                output = model(output)
                # set loss
                loss = loss_fn(output, target)
                # backward
                loss.backward()
                total_loss += loss.item()
                # optimize
                optimizer.step()
                # update pbar
                pbar.update(batch_size)
        # set loss
        total_loss = total_loss / len(train_data[0])
        # print loss
        print('Epoch: {}/{}, Loss: {}'.format(epoch + 1, max_epochs, loss))
        # set scheduler
        scheduler.step()
        # save model
        if (epoch + 1) % save_every == 0:
            # validate the model
            validate(model, transformer, device, valid_data, threshold = 0.5)
            model.save_model(os.path.join(model_path, model_name+'_'+str(epoch+1)+'.pth'))
        
    # save model
    model.save_model(os.path.join(model_path, model_name+'_final.pth')) 


def main(cfg):
    # train codes
    # load model
    model = LM(cfg)
    train_dataset = pkl.load(open('../dataset/train_dataset_4000.pkl', 'rb'))
    
    text, match = train_dataset['text'], train_dataset['match']
    
    text_t = []
    match_t = []
    for i in tqdm(range(len(text))):
        text_t.append(text[i]['input_ids'].tolist())
        match_t.append(float(match[i]))
    print(len(text_t), len(match_t))
    text_t = torch.tensor(text_t).reshape(len(text_t), -1)
    match_t = torch.tensor(match_t).reshape(len(match_t), -1)

    data_t = (text_t, match_t)
    # train model
    train(cfg, model, data_t, device = cfg.MODEL.DEVICE)

if __name__ == "__main__":
    main(cfg)
import sys
sys.path.append('../')
import torch
from models.config.defaults import cfg
from models.LM import LM
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import pickle as pkl
import random
import gc
from transformers import AutoModelForMaskedLM, XLMRobertaModel
import logging


#if is a number
def is_number(s):
    """
    Determine if it is a number
    @param:
        s: string
    @return:
        bool
    """
    if str(s) == 'nan':
        return False
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False


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

def validate(logger, model, transformer, device, valid_data, threshold = None):
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
    # count the pred_label and target_label
    pred_label = []
    target_label = []
    # set batch generator
    bg = batch_generator(valid_data, cfg.MODEL.BATCH_SIZE, shuffle=False)
    # validate
    with torch.no_grad():
        with tqdm(total=len(valid_data[0])) as pbar:
            for idx, (text1, text2, numerical,match) in enumerate(bg):
                
                target = match.to(device)

                numerical = numerical.to(device)
                
                input1 = text1.to(device)
                input2 = text2.to(device)

                output1 = transformer(input_ids = input1)
                output1 = output1.pooler_output
                output2 = transformer(input_ids = input2)
                output2 = output2.pooler_output

                output = torch.cat((output1,output2),axis = 1)
                output = model(output,numerical)
                
                # get the pred and set them to 0 or 1 based on the threshold
                pred = torch.sigmoid(output).detach().cpu().numpy()
                pred[pred < threshold] = 0
                pred[pred >= threshold] = 1
                # set pred_label
                pred_label.extend(pred)
                # set target_label
                target_label.extend(target.detach().cpu().numpy())
                # update progress bar
                pbar.update(cfg.MODEL.BATCH_SIZE)
    # print accuracy
    print('Validation accuracy: {}'.format(accuracy_score(target_label, pred_label)))
    # print auc
    print('Validation AUC: {}'.format(roc_auc_score(target_label, pred_label)))
    # print f1  
    print('Validation F1: {}'.format(f1_score(target_label, pred_label)))
    # print precision
    print('Validation precision: {}'.format(precision_score(target_label, pred_label)))
    # print recall
    print('Validation recall: {}'.format(recall_score(target_label, pred_label)))

    logger.info('************************************************')
    # print accuracy
    logger.info('Validation accuracy: {}'.format(accuracy_score(target_label, pred_label)))
    # print auc
    logger.info('Validation AUC: {}'.format(roc_auc_score(target_label, pred_label)))
    # print f1  
    logger.info('Validation F1: {}'.format(f1_score(target_label, pred_label)))
    # print precision
    logger.info('Validation precision: {}'.format(precision_score(target_label, pred_label)))
    # print recall
    logger.info('Validation recall: {}'.format(recall_score(target_label, pred_label)))
    logger.info('************************************************')

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
    data_t = TensorDataset(data[0], data[1], data[2], data[3])
    # set batch size
    batch_size = min(batch_size, len(data_t))
    # set batch generator
    batch_generator = DataLoader(   data_t, 
                                    batch_size=batch_size, 
                                    shuffle=shuffle) 
    # return batch generator
    return batch_generator

def train(cfg, model, train_data, device = 'cpu', 
     max_epochs = None, batch_size = None, 
     learning_rate = None, weight_decay = None, 
     model_path = None, model_name = None, 
     save_every = None, valid_size = None):
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

    # set the device
    if device.startswith("cuda"):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # if the model_path not exit, then make it and save the config setting to it
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        # save the config to the model_path as text fomat
        with open(os.path.join(model_path, 'config.yaml'), 'w') as f:
            f.write(cfg.dump())
        f.close()

    # set logger
    logging.basicConfig(filename = os.path.join(cfg.MODEL.MODEL_PATH, 'log.txt'),
                        format = '%(asctime)s-%(name)s-%(levelname)s-%(message)s-%(funcName)s:%(lineno)d',
                        datefmt = '%d %b %Y, %a %H:%M:%S',
                        level=logging.DEBUG,
                        encoding = 'utf-8',
                        filemode = 'w')

    logger = logging.getLogger()
    logger.info(cfg.dump())
    logger.info('begin logging...')

    # set the model to device
    model.to(device)

    transformer = XLMRobertaModel.from_pretrained("xlm-roberta-base").to(device)
    for idx, param in enumerate(transformer.parameters()):
        param.requires_grad = False
    
    print('Training model...')
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # set loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # set scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.MODEL.SCHEDULER_STEP, gamma=0.1, verbose=False)
    # validate the model
    train_data = TensorDataset(train_data[0], train_data[1],train_data[2],train_data[3])
    
    total = len(train_data)
    valid_size = int(total * valid_size)
    print("valid_size: " , valid_size)
    valid_data = train_data[-valid_size:]
    train_data = train_data[:-valid_size]

    match_count = 0
    non_match_count = 0
    for i in range(len(train_data[3])):
        if train_data[3][i] == 1:
            match_count += 1
        else:
            non_match_count += 1
    print("match_count: ", match_count/len(train_data[2]))
    print("non_match_count: ", non_match_count/len(train_data[2]))
    

    max_iter = (max_epochs + 1)*(len(train_data[0])//batch_size)
    iter_count = 0
    with tqdm(total = max_iter) as pbar:
        # train
        for epoch in range(max_epochs):
            # set train mode
            model.train()
            # set loss
            total_loss = 0  
            # set batch generator
            bg = batch_generator(train_data, batch_size, shuffle=True)
            # train
            for idx, (text1,text2, numerical,match) in enumerate(bg):
                # set optimizer, clear the grad
                optimizer.zero_grad()
                # set input
                input1 = text1.to(device)
                input2 = text2.to(device)

                #numerical feature
                numerical = numerical.to(device)

                # set target
                output1 = transformer(input_ids = input1)
                output1 = output1.pooler_output
                output2 = transformer(input_ids = input2)
                output2 = output2.pooler_output  

                #merge text   
                output = torch.cat((output1,output2),axis = 1)
                target = match.to(device)
                # set output
                output = model(output,numerical)

                # set loss
                loss = loss_fn(output, target)              
                # backward
                loss.backward()
                # update the parameters
                optimizer.step()
                # set total loss
                total_loss += loss.item()
                # set scheduler
                scheduler.step()   
                # update pbar
                pbar.update(1)
                iter_count += 1
                # save model
                if (iter_count) % save_every == 0:
                    # validate the model
                    print("current learning rate: ",scheduler.get_last_lr())
                    logger.info('{} epoch, {} iteration\'s loss: {}'.format(epoch ,iter_count, loss.item()))
                    logger.info("current learning rate: {}".format(scheduler.get_last_lr()))
                    validate(logger, model,transformer,device, valid_data, threshold = 0.5)
                    model.save_model(os.path.join(model_path, model_name+'_'+str(iter_count)+'.pth'))

        # set loss
        total_loss = total_loss / len(train_data[0])
        # print loss
        print('Epoch: {}/{}, Loss: {}'.format(epoch + 1, max_epochs, total_loss))
        logger.info('Epoch: {}/{}, Loss: {}'.format(epoch + 1, max_epochs, total_loss))
    # save model
    model.save_model(os.path.join(model_path, model_name+'_final.pth')) 

def negative_sampling(data, negative_size):
    """
    negative sampling
    @param:
        data: list of data
        negative_size: size of negative sampling
    @return:
        data: list of data
    """
    # set negative_size
    negative_size = min(negative_size, len(data[0]))
    


def main(cfg):
    # train codes
    # load model
    model = LM(cfg)

    #load model
    #model.load_model('/home/yushenglong/ML/Foursquare_Location_Matching/checkpoints/test_1597274/model_65000.pth')

    # load data from multiple files
    file_path_list = [
        '../dataset/numerical_datas/pair_train_dataset_last_397274.pkl',
    ]
    for i in range(400000, 1600000, 400000):
        file_path_list.append('../dataset/numerical_datas/pair_train_dataset_'+str(i)+'.pkl')
    print(file_path_list)

    train_data = {
        'text': [],
        'match': [],
        'numerical' : [], 
    }
    for path in file_path_list:
        print(path)
        cur_data = pkl.load(open(path, 'rb'))
        train_data['text'].extend(cur_data['text'])
        train_data['match'].extend(cur_data['match'])
        train_data['numerical'].extend(cur_data['numerical'])

    print("load over")

    # get text and match field from train_data
    text, match,numerical= train_data['text'], train_data['match'],train_data['numerical']
    
    
    #reduced memory usage
    del train_data
    gc.collect()

    ## count the match and non-match
    match_count = 0
    non_match_count = 0
    for i in tqdm(range(len(match))):
        if match[i] == 1:
            match_count += 1
        else:
            non_match_count += 1
    print("match_count: ", match_count/len(match))
    print("non_match_count: ", non_match_count/len(match))

    text_t1 = []
    text_t2 = []
    numerical_t = []
    match_t = []
    
    for i in tqdm(range(len(text))):
        text_t1.append(text[i][0]['input_ids'].tolist())
        text_t2.append(text[i][1]['input_ids'].tolist())
        temp = []
        for item in numerical[i]:
            #preprocess & padding
            if is_number(str(numerical[i][item])) == False:
                temp.append(-1)
            else:
                temp.append(float(numerical[i][item]))
        numerical_t.append(temp)
        match_t.append(float(match[i]))


    print("text size: ", len(text_t1), "match_size:" ,len(match_t))
    text_t1 = torch.tensor(text_t1).reshape(len(text_t1), -1)
    text_t2 = torch.tensor(text_t2).reshape(len(text_t2), -1)
    numerical_t = torch.tensor(numerical_t).reshape(len(numerical_t), -1)
    match_t = torch.tensor(match_t).reshape(len(match_t), -1)
    
    # shuffle the train and validate data
    random.seed(0)
    indices = list(range(len(text_t1)))
    random.shuffle(indices)
    text_t1 = text_t1[indices]
    text_t2 = text_t2[indices]
    match_t = match_t[indices]

    # train model
    data_t = (text_t1, text_t2,numerical_t,match_t)
    train(cfg, model, data_t, device = cfg.MODEL.DEVICE)

if __name__ == "__main__":
    main(cfg)

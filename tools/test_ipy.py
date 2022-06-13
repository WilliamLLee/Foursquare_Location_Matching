#.py for test.ipynb
import torch
import os
import sys
import gc
sys.path.append('../')

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import  XLMRobertaModel
from transformers import AutoTokenizer
from models.config.defaults import cfg
from models.LM import LM
from models.data import DATA

def is_number(s):
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

def inference(model, tokenizer, trasformer, device, test_data, threshold = 0.5, result_path = './result.csv', batch_size = 32):
    # process test data
    model.eval()

    text, id_1s, id_2s,numerical = test_data['text'], test_data['id_1'], test_data['id_2'],test_data['numerical']
    del test_data
    gc.collect()

    text_t1 = []
    text_t2 = []
    numerical_t = []
    for i in tqdm(range(len(text))):
        text_t1.append(tokenizer(
                text[i][0],
                add_special_tokens=True,
                padding = 'max_length',
                truncation = True,
                return_offsets_mapping = False,
                max_length = cfg.DATA.PREPROCESS_MAX_LEN,
                return_token_type_ids = False,
                return_attention_mask = False,
                return_tensors = 'pt',
            )['input_ids'].tolist())
        text_t2.append(tokenizer(
                text[i][1],
                add_special_tokens=True,
                padding = 'max_length',
                truncation = True,
                return_offsets_mapping = False,
                max_length = cfg.DATA.PREPROCESS_MAX_LEN,
                return_token_type_ids = False,
                return_attention_mask = False,
                return_tensors = 'pt',
            )['input_ids'].tolist())
        temp = []
        for item in numerical[i]:
            if is_number(str(numerical[i][item])) == False:
                temp.append(-1)
            else:
                temp.append(float(numerical[i][item]))
        print(temp)
        numerical_t.append(temp)

    text_t1 = torch.tensor(text_t1).reshape(len(text_t1), -1)
    text_t2 = torch.tensor(text_t2).reshape(len(text_t2), -1)
    numerical_t = torch.tensor(numerical_t).reshape(len(numerical_t), -1)

    # id2int : conver id to integer 
    id2int = {}
    int2id = list(set(id_1s+id_2s))
    for i, id_1 in enumerate(int2id):
        id2int[id_1] = i

    # convert id_1s and id_2s to integer
    id_1s = torch.tensor([id2int[id_1] for id_1 in id_1s]).reshape(-1, 1)
    id_2s = torch.tensor([id2int[id_2] for id_2 in id_2s]).reshape(-1, 1)

    for id1, id2 in zip(id_1s, id_2s):
        print(id1, id2)


    batch_size = min(batch_size, len(id_1s))
    bg = DataLoader(TensorDataset(text_t1, text_t2, id_1s, id_2s,numerical_t), batch_size = batch_size, shuffle = False)
    
    dict_ans = {ent:set([ent]) for ent in int2id}
    with torch.no_grad():
        with tqdm(total=len(text_t1)) as pbar:
            for idx, (text_1, text_2, id_1, id_2,numerical) in enumerate(bg):
                text_1 = text_1.to(device)
                text_2 = text_2.to(device)
                id_1 = id_1.to(device)
                id_2 = id_2.to(device)

                numerical = numerical.to(device)

                output1 = transformer(input_ids = text_1)
                output1 = output1.pooler_output

                output2 = transformer(input_ids = text_2)
                output2 = output2.pooler_output
                
                output = torch.cat((output1,output2),axis = 1)

                predict_res = model.predict(output,numerical, threshold = threshold)
                
                # filter the id_1s and id_2s which get predict_res as 1
                id_1_t = id_1[predict_res == 1]
                id_2_t = id_2[predict_res == 1]

                print(id_1_t.tolist(), id_2_t.tolist(), predict_res)
                pbar.update(batch_size)

                # record the result 
                for i in range(len(id_1_t)):
                    dict_ans[int2id[id_1_t[i]]].add(int2id[id_2_t[i]])
                    # dict_ans[int2id[id_2_t[i]]].add(int2id[id_1_t[i]])

    for key in dict_ans.keys():
        print(key, dict_ans[key])
    print('result saved in {}'.format(result_path))          
    with open(result_path, 'w') as f:
        for ent in dict_ans:
            f.write(ent + ',' + ' '.join(dict_ans[ent]) + '\n')
    
cfg.MODEL.IS_TRAIN = False
cfg.DATA.DATA_SAVED = False
cfg.MODEL.DEVICE = 'cuda'
cfg.TEST.MODEL_PATH = '/home/yushenglong/ML/Foursquare_Location_Matching/checkpoints/test_1597274/model_65000.pth'
cfg.TEST.BEST_THRESHOLD = 0.5
# set the device
if cfg.MODEL.DEVICE == "cpu":
    device = torch.device(cfg.MODEL.DEVICE)
else:
    device = torch.device(cfg.MODEL.DEVICE  if torch.cuda.is_available() else 'cpu')

# load model
model = LM(cfg)
model.load_model(cfg.TEST.MODEL_PATH)
transformer = XLMRobertaModel.from_pretrained("xlm-roberta-base").to(device)

for param in transformer.parameters():
    param.requires_grad = False

# set the model to device
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.PRETRAINED_MODEL_PATH)

# load data 
D = DATA(cfg)
test_data = D.get_test_data_dict(auto_gen=True, features = cfg.DATA.FEATURES[:-1], rounds = cfg.TEST.ROUNDS, n_neighbors=cfg.TEST.N_NEIGHBORS)

# inference on test dataset

inference(model, tokenizer, transformer, device, test_data, threshold = cfg.TEST.BEST_THRESHOLD, result_path = cfg.TEST.RESULT_PATH, batch_size = cfg.TEST.BATCH_SIZE)
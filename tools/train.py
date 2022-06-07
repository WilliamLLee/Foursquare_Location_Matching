from copyreg import pickle
import sys
sys.path.append('F:/Foursquare_Location_Matching/')

from models.config.defaults import cfg
from models.LM import LM
from models.data import DATA
import torch.utils.data as Data
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import pickle as pkl

class MyDataset(Data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def train(cfg):
    # load data
    data = DATA(cfg)
    # load model
    model = LM(cfg)
    
    data = data.get_pair_train_data_dict(auto_gen=True, rounds = 2, n_neighbors = 5, features = cfg.DATA.FEATURES)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.DATA.TOKENIZER_PATH)

    text = []
    match = []
    for  i in tqdm(range(len(data))):
        text.append(tokenizer(
                data[i]['text'],
                add_special_tokens=True,
                padding = 'max_length',
                truncation = True,
                return_offsets_mapping = False,
                max_length = cfg.DATA.PREPROCESS_MAX_LEN,
                return_token_type_ids = False,
                return_attention_mask = False,
                return_tensors = 'pt',
            ))
        match.append(data[i]['match'])
    
    print(data[0]['text'], data[0]['match'])
    print(text[0], match[0])

    train_dataset = dict({'text': text, 'match': match})
    pkl.dump(train_dataset, open('F:/Foursquare_Location_Matching/dataset/train_dataset.pkl', 'wb'))
    
    train_data = MyDataset(text, match)

    # train model
    model.train(train_data)

if __name__ == "__main__":
    train(cfg)
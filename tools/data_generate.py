import sys
sys.path.append('../')
from models.config.defaults import cfg
from models.data import DATA
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle as pkl
import gc

def main(cfg):
    # #  reseting the config

    # load data class 
    D = DATA(cfg)
    data = D.get_pair_train_data_dict(auto_gen=True, rounds = 4, n_neighbors = 10, features = cfg.DATA.FEATURES)
    tokenizer = AutoTokenizer.from_pretrained(cfg.DATA.TOKENIZER_PATH)

    #pos-neg
    pos_neg = [0,0]
    for i in tqdm(range(len(data))):
        pos_neg[int(data[i]['match'])] += 1
    print("neg:pos",pos_neg)

    text = []
    match = []
    numerical = []
    save_every = 400000
    for  i in tqdm(range(len(data))):
        text.append([tokenizer(
                data[i]['text'][0],
                add_special_tokens=True,
                padding = 'max_length',
                truncation = True,
                return_offsets_mapping = False,
                max_length = cfg.DATA.PREPROCESS_MAX_LEN,
                return_token_type_ids = False,
                return_attention_mask = False,
                return_tensors = 'pt',
            ),tokenizer(
                data[i]['text'][1],
                add_special_tokens=True,
                padding = 'max_length',
                truncation = True,
                return_offsets_mapping = False,
                max_length = cfg.DATA.PREPROCESS_MAX_LEN,
                return_token_type_ids = False,
                return_attention_mask = False,
                return_tensors = 'pt',
            )])

        #separate & save
        numerical.append(data[i]['num_entities'])
        match.append(data[i]['match'])
        if (i + 1) % save_every == 0:
            path = '../dataset/numerical_datas/pair_train_dataset_'+str(i+1)+'.pkl'
            pkl.dump({'text':text, 'match':match,'numerical':numerical}, open(path, 'wb'))
            text = []
            match = []
            gc.collect()
    
    path = '../dataset/numerical_datas/pair_train_dataset_last_'+str(len(text))+'.pkl'
    pkl.dump({'text':text, 'match':match,'numerical':numerical}, open(path, 'wb'))

if __name__ == "__main__":
    main(cfg)
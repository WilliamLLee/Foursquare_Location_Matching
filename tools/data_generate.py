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
    cfg.DATA.DATA_SAVED = False
    cfg.DATA.PAIRS_DATA_DICT = ""
    cfg.DATA.PAIRS_DATA_LIST = ""
    # load data class 
    D = DATA(cfg)

    data = D.get_pair_train_data_dict(auto_gen=False, rounds = 2, n_neighbors = 2, features = cfg.DATA.FEATURES)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.DATA.TOKENIZER_PATH)

    text = []
    match = []
    save_every = 200000
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
        if (i + 1) % save_every == 0:
            path = '../dataset/pair_train_dataset_'+str(i+1)+'.pkl'
            pkl.dump({'text':text, 'match':match}, open(path, 'wb'))
            text = []
            match = []
            gc.collect()
    
    path = '../dataset/pair_train_dataset_last_'+str(len(text))+'.pkl'
    pkl.dump({'text':text, 'match':match}, open(path, 'wb'))
    # finished in about 18 minutes

if __name__ == "__main__":
    main(cfg)
from yacs.config import CfgNode as CN

_C = CN()

# data path
_C.DATA = CN()
_C.DATA.DATA_PATH = 'F:\Foursquare_Location_Matching/dataset'
_C.DATA.FEATURES = [
    'id',
    'name',
    'latitude',
    'longitude',
    'address',
    'city',
    'state',
    'zip',
    'country',
    'url',
    'phone',
    'categories',
    'point_of_interest',
]
_C.DATA.FEATURE_INDEX = [{   # data index for each feature in train data
    'id': 0,
    'name': 1,
    'latitude': 2,
    'longitude': 3,
    'address': 4,
    'city': 5,
    'state': 6,
    'zip': 7,
    'country': 8,
    'url': 9,
    'phone': 10,
    'categories': 11,
    'point_of_interest': 12,
}]
_C.DATA.TEXT_FEATURE_TYPE = ['name', 'address', 'city', 'state', 'country','categories']  # drop the 'url' feature
_C.DATA.NUMERICAL_FEATURE_TYPE = ['latitude', 'longitude', 'zip', 'phone']

_C.DATA.PREPROCESS_MAX_LEN = 150
_C.DATA.TOKENIZER_PATH = 'F:\Foursquare_Location_Matching/models/xlm-roberta-base'

# files names  
_C.DATA.TEST_FILE = 'test.csv'
_C.DATA.TRAIN_FILE = 'train.csv'
_C.DATA.PAIRS_FILE = 'pairs.csv'


# saved data 
_C.DATA.DATA_SAVED = True
_C.DATA.PAIRS_DATA_DICT = 'F:\Foursquare_Location_Matching/dataset/pairs_data_dict.npy'
_C.DATA.PAIRS_DATA_LIST = 'F:\Foursquare_Location_Matching/dataset/pairs_data_list.npy'
# _C.DATA.PAIRS_DATA_DICT = ''
# _C.DATA.PAIRS_DATA_LIST = ''


# model config
_C.MODEL = CN()
_C.MODEL.IS_TRAIN = True
_C.MODEL.PRETRAINED_MODEL_PATH = 'F:\Foursquare_Location_Matching/models/xlm-roberta-base'
_C.MODEL.PRETRAINED_MODEL_NAME = 'xlm-roberta-base'
_C.MODEL.DROPOUT_RATE1 = 0.15
_C.MODEL.DROPOUT_RATE2 = 0.20
_C.MODEL.DROPOUT_RATE3 = 0.25
_C.MODEL.OUTPUT_DIM = 1
_C.MODEL.INPUT_DIM = 768*150

_C.MODEL.LR = 0.001
_C.MODEL.MAX_EPOCHS = 10
_C.MODEL.BATCH_SIZE = 64
_C.MODEL.NUM_WORKERS = 4
_C.MODEL.WEIGHT_DECAY = 0.1
_C.MODEL.SAVE_EVERY = 2
_C.MODEL.MODEL_PATH = 'checkpoints'
_C.MODEL.VALID_SIZE = 0.1

_C.MODEL.THRESHOLD = 0.5

cfg = _C
from yacs.config import CfgNode as CN

_C = CN()

# data path
_C.DATA = CN()
_C.DATA.DATA_PATH = './dataset'
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
_C.DATA.FEATURE_INDEX = {   # data index for each feature in train data
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
}
_C.DATA.TEXT_FEATURE_TYPE = ['name', 'address', 'city', 'state', 'country','categories']  # drop the 'url' feature
_C.DATA.NUMERICAL_FEATURE_TYPE = ['latitude', 'longitude', 'zip', 'phone']

_C.DATA.PREPROCESS_MAX_LEN = 150
_C.DATA.TOKENIZER_PATH = './models/xlm-roberta-base'

# files names  
_C.DATA.TEST_FILE = 'test.csv'
_C.DATA.TRAIN_FILE = 'train.csv'
_C.DATA.PAIRS_FILE = 'pairs.csv'

# model config
_C.MODEL = CN()
_C.MODEL.IS_TRAINING = True
_C.MODEL.PRETRAINED_MODEL_PATH = './models/xlm-roberta-base'
_C.MODEL.PRETRAINED_MODEL_NAME = 'xlm-roberta-base'
_C.MODEL.DROPOUT_RATE1 = 0.15
_C.MODEL.DROPOUT_RATE2 = 0.20
_C.MODEL.DROPOUT_RATE3 = 0.25
_C.MODEL.TARGET_SIZE = 1 

cfg = _C
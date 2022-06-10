from yacs.config import CfgNode as CN

_C = CN()

# data path
_C.DATA = CN()
_C.DATA.DATA_PATH = '../dataset'
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
_C.DATA.TOKENIZER_PATH = '../models/xlm-roberta-base'

# files names  
_C.DATA.TEST_FILE = 'test.csv'
_C.DATA.TRAIN_FILE = 'train.csv'
_C.DATA.PAIRS_FILE = 'pairs.csv'


# saved data 
_C.DATA.DATA_SAVED = True
_C.DATA.PAIRS_DATA_DICT = '../dataset/pairs_data_dict.npy'
_C.DATA.PAIRS_DATA_LIST = '../dataset/pairs_data_list.npy'
# _C.DATA.PAIRS_DATA_DICT = ''
# _C.DATA.PAIRS_DATA_LIST = ''


# model config
_C.MODEL = CN()
_C.MODEL.IS_TRAIN = True
_C.MODEL.PRETRAINED_MODEL_PATH = '../models/xlm-roberta-base'
_C.MODEL.PRETRAINED_MODEL_NAME = 'xlm-roberta-base'
_C.MODEL.DROPOUT_RATE1 = 0.15
_C.MODEL.DROPOUT_RATE2 = 0.15
_C.MODEL.DROPOUT_RATE3 = 0.20
_C.MODEL.OUTPUT_DIM = 1
_C.MODEL.INPUT_DIM = 768


_C.MODEL.DEVICE = 'cuda:1'
_C.MODEL.LR = 0.01
_C.MODEL.MAX_EPOCHS = 10
_C.MODEL.BATCH_SIZE = 4
_C.MODEL.NUM_WORKERS = 4
_C.MODEL.WEIGHT_DECAY = 0.1
_C.MODEL.SAVE_EVERY = 1
_C.MODEL.SCHEDULER_STEP = 1000
_C.MODEL.MODEL_PATH = '../checkpoints/20220610_xlm-roberta-base_rate1_0.15_rate2_0.15_rate3_0.20_lr_0.001_epochs_10_batch_64_weight_decay_0.1'
_C.MODEL.MODEL_NAME = 'model'
_C.MODEL.VALID_SIZE = 0.1

_C.MODEL.THRESHOLD = 0.5



# test config
_C.TEST = CN()
_C.TEST.RESULT_PATH = '../dataset/test_result.csv'
_C.TEST.MODEL_PATH = '../checkpoints/20220610_xlm-roberta-base_rate1_0.15_rate2_0.15_rate3_0.20_lr_0.001_epochs_10_batch_64_weight_decay_0.1/model_1.pth'
_C.TEST.BEST_THRESHOLD = 0.45
_C.TEST.BATCH_SIZE = 3
_C.TEST.ROUNDS = 4
_C.TEST.N_NEIGHBORS = 5

cfg = _C

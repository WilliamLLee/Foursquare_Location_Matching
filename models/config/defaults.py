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

_C.DATA.PREPROCESS_MAX_LEN = 100
_C.DATA.TOKENIZER_PATH = '../models/xlm-roberta-base'

# files names  
_C.DATA.TEST_FILE = 'test.csv'
_C.DATA.TRAIN_FILE = 'train.csv'
_C.DATA.PAIRS_FILE = 'pairs.csv'


# saved data 
_C.DATA.DATA_SAVED = True
_C.DATA.PAIRS_DATA_DICT = '../dataset/pairs_data_dict_s.npy'
_C.DATA.PAIRS_DATA_LIST = '../dataset/pairs_data_list_s.npy'


# model config
_C.MODEL = CN()
_C.MODEL.IS_TRAIN = True
_C.MODEL.DEVICE = 'cuda:3'
_C.MODEL.LR = 0.001
_C.MODEL.MAX_EPOCHS = 30


# if the batch size is set to small number, 
# then the model could be trained under less GPU memory
# for example: whem batch_size=12, the GPU memory is just 3986mb,
# which can be trained on a laptop with only 4GB of GPU memory 
_C.MODEL.BATCH_SIZE = 512
_C.MODEL.WEIGHT_DECAY = 0.01
_C.MODEL.SAVE_EVERY = 1000
_C.MODEL.SCHEDULER_STEP = 1000000000
_C.MODEL.MODEL_PATH = '../checkpoints/'
_C.MODEL.MODEL_NAME = 'model'
_C.MODEL.VALID_SIZE = 0.01
_C.MODEL.THRESHOLD = 0.5

_C.MODEL.PRETRAINED_MODEL_PATH = '../models/xlm-roberta-base'
_C.MODEL.PRETRAINED_MODEL_NAME = 'xlm-roberta-base'
_C.MODEL.PRETRAINED_LAYER_NUM = 101    # 202 layers
_C.MODEL.DROPOUT_RATE1 = 0.15
_C.MODEL.DROPOUT_RATE2 = 0.15
_C.MODEL.DROPOUT_RATE3 = 0.20
_C.MODEL.OUTPUT_DIM = 1
_C.MODEL.INPUT_DIM = 768*2

# test config
_C.TEST = CN()
_C.TEST.RESULT_PATH = '../dataset/test_result.csv'
_C.TEST.MODEL_PATH = ''
_C.TEST.BEST_THRESHOLD = 0.5
_C.TEST.BATCH_SIZE = 1
_C.TEST.ROUNDS = 5
_C.TEST.N_NEIGHBORS = 5

cfg = _C

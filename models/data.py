import os
import numpy as np
import pandas as pd
from sklearn import neighbors
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer 
from sklearn.preprocessing import StandardScaler

from .config import cfg

class DATA:
    def __init__(self, cfg, data_path):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.DATA.TOKENIZER_PATH)
        self.data_path = data_path
        if self.cfg.MODEL.IS_TRAIN:
            self.train_data = self.load_data(self.cfg.DATA.TRAIN_FILE)
            self.pairs_data = self.load_data(self.cfg.DATA.PAIRS_FILE)
        else:
            self.test_data = self.load_data(self.cfg.DATA.TEST_FILE)

    def load_data(self, filename):
        """
        Load data from the data path
        """
        if os.path.join(self.cfg.DATA.DATA_PATH, filename) is not None:
            return pd.read_csv(os.path.join(self.cfg.DATA.DATA_PATH, filename))

    def genereate_data(self, rounds = 2, n_neighbors = 10, features = ['id', 'latitude', 'longitude'], knn_features = ['latitude', 'longitude']):
        """
        Generate data pairs for the model
        using the KNN to find the nearest neighbors that have the nearest latitude and longitude.
        @param:
            rounds: number of rounds to generate test data
            n_neighbors: number of neighbors to find
            features: list of features to use
            knn_features: list of features to use for KNN
        @return:
            test_data: list of test data
        """
        assert rounds > 0, "rounds must be greater than 0"
        assert n_neighbors > 0, "n_neighbors must be greater than 0"
        assert self.cfg.MODEL.IS_TRAIN, "This function can only be used in training mode"

        # scale data for KNN 
        scaler = StandardScaler
        scaled_data = scaler.fit_transform(self.train_data[knn_features])
        # fit KNN and predict indices
        knn_model = NearestNeighbors( 
            n_neighbors=n_neighbors, 
            algorithm='kd_tree',
            radius = 1.0,
            leaf_size = 30,
            metric = 'minkowski',
            p = 2,
            n_jobs = -1,
        )
        knn_model.fit(scaled_data)
        indices = knn_model.kneighbors(scaled_data, return_distance=False)

        # generate data
        data_features = self.train_data[features]
        dataset = []

        for j in range(rounds):
            tmp_dataset = []
            for k in tqdm(range(len(data_features))):
                neighbors = indices[k]
                try:
                    neighbors.remove(k)
                except:
                    pass
                    
                ind1 = k
                ind2 = neighbors[j]
                if ind1 == ind2:
                    print("indices are the same!")
                
                tmp_dataset.append(np.concatenate([ data_features.iloc[ind1],
                                                    data_features.iloc[ind2]], 
                                                    axis = 0))
            dataset.append(
                pd.DataFrame(tmp_dataset, columns = [i+'_1' for i in features] + [i+'_2' for i in features])
            )
        dataset = pd.concat(dataset, axis = 0)
        dataset.drop_duplicates(inplace = True)
        dataset.reset_index(drop = True, inplace = True)
        col_64 = list(dataset.dtypes[dataset.dtypes==np.float64].index)
        for col in col_64:
            dataset[col] = dataset[col].astype(np.float32)
        return data_features, dataset

    @staticmethod
    def preprocess_data(self, data):
        '''
        preprocess the data input, for example, convert text to index with a tokenizer
        @param:
            data: data to preprocess
        @return:
            data: preprocessed data
        '''
        input_ids = []
        for text in tqdm(data, total=len(data)):
            inputs = self.tokenizer(
                text, 
                add_special_tokens=True,
                padding = 'max_length',
                truncation = True,
                return_offsets_mapping = False,
                max_length = self.cfg.DATA.PREPROCESS_MAX_LEN,
                return_token_type_ids = False,
                return_attention_mask = False,
            )
        return np.array(input_ids)


    def get_train_data(self):
        '''
        get the train data
        '''
        return self.train_data
    
    def get_pair_train_data(self, auto_gen = False, rounds = 2, n_neighbors = 10, features = ['id', 'latitude', 'longitude'], knn_features = ['latitude', 'longitude']):
        '''
        Get the data pairs for the model, and preprocess the data, if auto_gen is set to True, generate the data pairs from the train_data
        @param:
            auto_gen: whether to generate the data pairs from the train_data
            rounds: number of rounds to generate test data
            n_neighbors: number of neighbors to find
            features: list of features to use
            knn_features: list of features to use for KNN
        @return:
            pairs_data_dict: dictionary of data pairs
        '''
        pairs_data = self.pairs_data
        print("orgigin pairs_data: ", pairs_data.shape)
        if auto_gen:
            _, gen_pairs_data = self.genereate_data(rounds, n_neighbors, features, knn_features)
            gen_pairs_data['match'] = gen_pairs_data.apply(lambda x: x['point_of_interest_1'] == x['point_of_interest_2'], axis = 1)  # generate match column
            gen_pairs_data.drop(['point_of_interest_1', 'point_of_interest_2'], axis = 1, inplace = True) # drop the point of interest columns
            pairs_data = pd.concat([pairs_data, gen_pairs_data], axis = 0)
            print("generated pairs_data: ", pairs_data.shape)
        pairs_data_list = []
        for ind in tqdm(range(len(pairs_data)), total = len(pairs_data)):
            temp_dict = {}
            for col in pairs_data.columns:
                temp_dict[col] = pairs_data[col].iloc[ind]
            pairs_data_list.append(temp_dict)
        
        print("pairs_data_list: ", len(pairs_data_list))
        return pairs_data_list
    
    def get_test_data(self):
        '''
        get the test data, and organize the test data into pairs
        '''
        test_data = self.test_data
        test_data_list = []
        # organize the test data into pairs one by one

        for ind1 in tqdm(range(len(test_data))):
            for ind2 in range(ind1, len(test_data)):
                tmp_dict = {}
                for col in self.DATA.FEATURES[:-1]:
                    tmp_dict[col+'_1'] = test_data[col].iloc[ind1]
                    tmp_dict[col+'_2'] = test_data[col].iloc[ind2]
                test_data_list.append(tmp_dict)

        print("test_data_list: ", len(test_data_list))
        return self.test_data_list

    
    
# TODO:
# 1. 决定好用什么特征
# 2. 对数据进行预处理，比如：
#    a. 对数据进行分词
#    b. 对数据进行编码
# 3. 使用数据进行训练，训练一个文本相似度模型（可作为baseline提交，将这个任务看做是短文本匹配的任务）
# 4. 使用数据进行预测，预测一个文本相似度，并且提交结果
# 5. 使用决策树等模型进一步进行匹配度计算，并且提交结果，可以将文本相似匹配度看做是一个增强特征
    

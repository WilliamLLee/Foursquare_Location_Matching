import os
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class DATA:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_path = self.cfg.DATA.DATA_PATH
        if not self.cfg.DATA.DATA_SAVED:
            print("Loading data..., data path: ", self.data_path)
            if self.cfg.MODEL.IS_TRAIN:
                self.train_data = self.load_data(self.cfg.DATA.TRAIN_FILE)
                self.pairs_data = self.load_data(self.cfg.DATA.PAIRS_FILE)
                print("train_data: ", len(self.train_data))
                print("pairs_data: ", len(self.pairs_data))
            else:
                self.test_data = self.load_data(self.cfg.DATA.TEST_FILE)
                print("test_data: ", len(self.test_data))

    def load_data(self, filename):
        """
        Load data from the data path
        """
        if os.path.join(self.cfg.DATA.DATA_PATH, filename) is not None:
            return pd.read_csv(os.path.join(self.cfg.DATA.DATA_PATH, filename))

    def genereate_data(self, input_data, rounds = 2, n_neighbors = 10, features = ['id', 'latitude', 'longitude'], knn_features = ['latitude', 'longitude']):
        """
        Generate data pairs for the model
        using the KNN to find the nearest neighbors that have the nearest latitude and longitude.
        @param:
            input_data : data to generate data pairs for
            rounds: number of rounds to generate test data
            n_neighbors: number of neighbors to find
            features: list of features to use
            knn_features: list of features to use for KNN
        @return:
            test_data: list of test data
        """
        assert rounds > 0, "rounds must be greater than 0"
        assert n_neighbors > 0, "n_neighbors must be greater than 0"

        # scale data for KNN 
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(input_data[knn_features])
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
        data_features = input_data[features]
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
                    # print("indices are the same!"), only for train data, skip the same indices
                    continue                
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
            _, gen_pairs_data = self.genereate_data(self.train_data, rounds, n_neighbors, features, knn_features)
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
    
    def get_pair_train_data_dict(self, auto_gen = False, rounds = 2, n_neighbors = 10, features = ['id', 'latitude', 'longitude'], knn_features = ['latitude', 'longitude']):
        '''
        get the paired training data as directory format
        @param:
            auto_gen: whether to generate the data pairs from the train_data
            rounds: number of rounds to generate test data
            n_neighbors: number of neighbors to find
            features: list of features to use
            knn_features: list of features to use for KNN
        @return:
            pairs_data_list: list of data pairs
            pairs_data_dict: dictionary of data pairs
        '''

        if self.cfg.DATA.DATA_SAVED and self.cfg.DATA.PAIRS_DATA_LIST != '' and self.cfg.DATA.PAIRS_DATA_DICT != '':
            print("pairs_data_list and pairs_data_dict are already generated,load data...")
            dict_load = np.load(self.cfg.DATA.PAIRS_DATA_DICT, allow_pickle = True)
            list_load = np.load(self.cfg.DATA.PAIRS_DATA_LIST, allow_pickle = True)
            print("pairs_data_list: ", len(list_load))
            print(list_load[-10:])
            print("only return the 'text' and 'match' columns")
            dict_return = []
            for i in tqdm(range(len(dict_load))):
                dict_return.append({'num_entities':dict_load[i]['num_entities'],'text': dict_load[i]['text'], 'match': dict_load[i]['match']})   
            return dict_return

        pairs_data_list = self.get_pair_train_data(auto_gen, rounds, n_neighbors, features, knn_features)
        print("organizing pairs_data_list as dictionary format: {'text': text, 'num_entities': num_entities, 'match': match}")
        pairs_data_dict = []
        for i in tqdm(range(len(pairs_data_list)), total = len(pairs_data_list)):
            temp_dict = {} 

            str1 = ""
            str2 = ""
            for col in [ i + '_1' for i in self.cfg.DATA.TEXT_FEATURE_TYPE]:
                str1 += str(pairs_data_list[i][col]) + " "
            for col in [ i + '_2' for i in self.cfg.DATA.TEXT_FEATURE_TYPE]:
                str2 += str(pairs_data_list[i][col]) + " "
            temp_dict['text'] = [str1,str2]
            
            temp_dict['num_entities'] = {}
            for col in [i + '_1' for i in self.cfg.DATA.NUMERICAL_FEATURE_TYPE] +   [i + '_2' for i in self.cfg.DATA.NUMERICAL_FEATURE_TYPE]:
                temp_dict['num_entities'][col] = pairs_data_list[i][col]

            temp_dict['match'] = pairs_data_list[i]['match']
            
            pairs_data_dict.append(temp_dict)

        np.save(os.path.join(self.cfg.DATA.DATA_PATH, 
                            'pairs_data_dict_s.npy'), 
                pairs_data_dict)
        np.save(os.path.join(self.cfg.DATA.DATA_PATH,
                            'pairs_data_list_s.npy'),
                pairs_data_list)
        print("pairs_data_list and pairs_data_dict are generated, saved to {} and {}".format(os.path.join(self.cfg.DATA.DATA_PATH,
                            'pairs_data_list_s.npy'), os.path.join(self.cfg.DATA.DATA_PATH, 
                            'pairs_data_dict_s.npy')))

        
        return pairs_data_dict
        
    def get_test_data_list(self, full_match = False, auto_gen = False, rounds = 9, n_neighbors = 10, features = ['id', 'latitude', 'longitude'], knn_features = ['latitude', 'longitude']):
        '''
        get the test data, and organize the test data into pairs
        @param:
            full_match: whether to use the full match method
            auto_gen: whether to generate the data pairs from the train_data
            rounds: number of rounds to generate test data
            n_neighbors: number of neighbors to find
            features: list of features to use
            knn_features: list of features to use for KNN
        @return:
            test_data_list: list of test data
        '''
        test_data = self.test_data
        test_data_list = []
        # organize the test data into pairs one by one
        if full_match:
            print("organizing test_data into pairs using full match method...")
            for ind1 in tqdm(range(len(test_data))):
                for ind2 in range(ind1, len(test_data)):
                    tmp_dict = {}
                    for col in self.DATA.FEATURES[:-1]:
                        tmp_dict[col+'_1'] = test_data[col].iloc[ind1]
                        tmp_dict[col+'_2'] = test_data[col].iloc[ind2]
                    test_data_list.append(tmp_dict)
        else:
            if auto_gen:
                print("generating test data...")
                _, pairs_data = self.genereate_data(test_data, rounds, n_neighbors, features, knn_features)
                for ind in tqdm(range(len(pairs_data)), total = len(pairs_data)):
                    temp_dict = {}
                    for col in pairs_data.columns:
                        temp_dict[col] = pairs_data[col].iloc[ind]
                    test_data_list.append(temp_dict)
            else:
                print("there is no test data could be organized as pairs, please check the config file")
        print("test_data_list: ", len(test_data_list))
        return test_data_list

    def get_test_data_dict(self, full_match = False, auto_gen = True, rounds = 2, n_neighbors = 3, features = ['id', 'latitude', 'longitude'], knn_features = ['latitude', 'longitude']):
        '''
        get the test data in dictionary format
        '''
        test_data_list = self.get_test_data_list(auto_gen=auto_gen, full_match=full_match, rounds=rounds, n_neighbors=n_neighbors, features=features, knn_features=knn_features)
        
        print("organizing test_data_list as dictionary format: {'text': text, 'num_entities': num_entities}")
        test_data_dict = {
            'text': [],
            'numerical': [],
            'id_1': [],
            'id_2': [],
        }
        for i in tqdm(range(len(test_data_list)), total = len(test_data_list)):
            test_data_dict['id_1'].append(test_data_list[i]['id_1'])
            test_data_dict['id_2'].append(test_data_list[i]['id_2'])
            text1, text2 = '', ''
            for col in [ i + '_1' for i in self.cfg.DATA.TEXT_FEATURE_TYPE]:
                text1 += str(test_data_list[i][col]) + " "
        
            for col in [ i + '_2' for i in self.cfg.DATA.TEXT_FEATURE_TYPE]:
                text2 += str(test_data_list[i][col]) + " "
            num_entities = {}
            for col in [i + '_1' for i in self.cfg.DATA.NUMERICAL_FEATURE_TYPE] +   [i + '_2' for i in self.cfg.DATA.NUMERICAL_FEATURE_TYPE]:
                num_entities[col] = test_data_list[i][col]
            test_data_dict['text'].append([text1, text2])
            test_data_dict['numerical'].append(num_entities)
        return test_data_dict    

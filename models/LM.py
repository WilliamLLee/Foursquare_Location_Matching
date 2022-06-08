import os
import torch 
import sys
sys.path.append('../')
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class LM(torch.nn.Module):
    def __init__(self, cfg):
        super(LM, self).__init__()
        self.cfg = cfg
        self.transformer = AutoModelForMaskedLM.from_pretrained(self.cfg.MODEL.PRETRAINED_MODEL_PATH, output_hidden_states=True, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.DATA.TOKENIZER_PATH)
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(self.cfg.MODEL.INPUT_DIM, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.cfg.MODEL.DROPOUT_RATE1),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.cfg.MODEL.DROPOUT_RATE2),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.cfg.MODEL.DROPOUT_RATE3),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, self.cfg.MODEL.OUTPUT_DIM),
        )
        self.sigmoid = torch.nn.Sigmoid()
    
    @staticmethod
    def tokenizer(self, text):
        '''
        tokenize the text
        @param:
            text: text to be tokenized
        @return:
            tokenized_text: tokenized text
        '''
        tokenized_text = self.tokenizer(
                text, 
                add_special_tokens=True,
                padding = 'max_length',
                truncation = True,
                return_offsets_mapping = False,
                max_length = self.cfg.DATA.PREPROCESS_MAX_LEN,
                return_token_type_ids = False,
                return_attention_mask = False,
            )
        return tokenized_text

    def forward(self, input):
        """
        Predict the test data
        @param:
            input_data: list of test data
        @return:
            predictions: list of predictions
        """
        # predict
        # output = self.tokenizer(input)
        output = self.transformer(input_ids = input)['hidden_states'][-1]
        print(output.shape)
        output = output.reshape(output.shape[0], -1)
        print(output.shape)
        # linear layer to get the predictions
        output = self.linear_relu_stack(output)
        print(output.shape)
        output = self.sigmoid(output)
        return output

    def predict(self, input_data, threshold = None):
        """
        Predict the test data
        @param:
            input_data: list of test data
            threshold: threshold to filter the predictions
        @return:
            predictions: list of predictions
        """
        if threshold is None:
            threshold = self.cfg.MODEL.THRESHOLD
        # predict
        with torch.no_grad():
            output = self(input_data)
        # filter the predictions by threshold
        predictions = self.filter_predictions(output, threshold)
        return predictions

    def filter_predictions(self, predictions, threshold):
        """
        Filter the predictions by threshold
        @param:
            predictions: list of predictions
            threshold: threshold to filter the predictions
        @return:
            filtered predictions: list of filtered predictions
        """
        filtered_predictions = []
        for prediction in predictions:
            if prediction >= threshold:
                filtered_predictions.append(1)
            else:
                filtered_predictions.append(0)
        return filtered_predictions
    
    def save_model(self, model_path):
        """
        Save the model
        @param:
            model_path: path to save the model
        """
        torch.save(self.state_dict(), model_path)
    
    def load_model(self, model_path):
        """
        Load the model
        @param:
            model_path: path to load the model
        """
        self.load_state_dict(torch.load(model_path))



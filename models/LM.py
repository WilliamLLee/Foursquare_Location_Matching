import torch 
import sys
sys.path.append('../')
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

class LM(torch.nn.Module):
    def __init__(self, cfg):
        super(LM, self).__init__()
        self.cfg = cfg
        # self.linear_1 = nn.Linear(self.cfg.MODEL.INPUT_DIM, 1024)
        # self.batch_norm_1 = nn.BatchNorm1d(1024)
        # self.linear_2 = nn.Linear(1024, 512)
        # self.batch_norm_2 = nn.BatchNorm1d(512)
        self.linear_3 = nn.Linear(512, 256)
        self.batch_norm_3 = nn.BatchNorm1d(256)
        self.linear_4 = nn.Linear(256, cfg.MODEL.OUTPUT_DIM)
        
        self.transformer = AutoModelForMaskedLM.from_pretrained(
                        cfg.MODEL.PRETRAINED_MODEL_PATH, 
                        output_hidden_states=True, 
                        output_attentions=True)
        # freeze the former n-1 layers of transformer, and finetune the last layer
        for idx, param in enumerate(self.transformer.parameters(), 1):
            if idx < cfg.MODEL.PRETRAINED_LAYER_NUM:
                param.requires_grad = False

    def forward(self, input):
        """
        Predict the test data
        @param:
            input_data: list of test data
        @return:
            predictions: list of predictions
        """
        # linear layer to get the predictions
        output = self.transformer(input_ids = input)['hidden_states'][-1]
        output = output[:, 0, :].reshape(output.shape[0], -1)
        output = F.adaptive_avg_pool1d(output.unsqueeze(1), self.cfg.MODEL.INPUT_DIM).squeeze(1)
        # output = F.relu(self.linear_1(output))
        # output = F.dropout(output, p=self.cfg.MODEL.DROPOUT_RATE1)
        # output = self.batch_norm_1(output)
        # output = F.relu(self.linear_2(output))
        # output = F.dropout(output, p=self.cfg.MODEL.DROPOUT_RATE2)
        # output = self.batch_norm_2(output)
        output = F.relu(self.linear_3(output))
        output = F.dropout(output, p=self.cfg.MODEL.DROPOUT_RATE3)
        output = self.batch_norm_3(output)
        output = self.linear_4(output)
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
            output = self.forward(input_data)
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
        filtered_predictions = predictions.clone()
        filtered_predictions[filtered_predictions < threshold] = 0
        filtered_predictions[filtered_predictions >= threshold] = 1
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



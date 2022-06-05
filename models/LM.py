import os
import torch 

from .config import cfg
from transformers import AutoTokenizer, AutoModelForMaskedLM


class LM(torch.nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        self.transformer = AutoModelForMaskedLM.from_pretrained(self.cfg.MODEL.PRETRAINED_MODEL_PATH)
    
    def forward(self, input):
        """
        Predict the test data
        @param:
            input_data: list of test data
        @return:
            predictions: list of predictions
        """
        # predict
        output = self.transformer(**input.text)
        # linear layer to get the predictions
        output = torch.nn.Linear(output.shape[-1], 1024)(output)
        output = torch.nn.ReLU()(output)
        output = torch.nn.Dropout(p=self.cfg.MODEL.DROPOUT_RATE1)(output)
        output = torch.nn.BatchNorm1d(output.shape[-1])(output)
        
        output = torch.nn.Linear(output.shape[-1], 512)(output)
        output = torch.nn.ReLU()(output)
        output = torch.nn.Dropout(p=self.cfg.MODEL.DROPOUT_RATE2)(output)
        output = torch.nn.BatchNorm1d(output.shape[-1])(output)

        output = torch.nn.Linear(output.shape[-1], 256)(output)
        output = torch.nn.ReLU()(output)
        output = torch.nn.Dropout(p=self.cfg.MODEL.DROPOUT_RATE3)(output)
        output = torch.nn.BatchNorm1d(output.shape[-1])(output)

        output = torch.nn.Linear(output.shape[-1], self.cfg.MODEL.TARGET_SIZE)(output)
        output = torch.nn.Sigmoid()(output)
        return output



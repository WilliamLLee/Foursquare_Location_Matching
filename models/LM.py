import os
import torch 
import sys
sys.path.append('../')
from transformers import AutoTokenizer, AutoModelForMaskedLM


class LM(torch.nn.Module):
    def __init__(self, cfg):
        super(LM, self).__init__()
        self.cfg = cfg
        self.transformer = AutoModelForMaskedLM.from_pretrained(self.cfg.MODEL.PRETRAINED_MODEL_PATH)
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
        output = self.transformer(**input)
        print(output.shape[-1])
        # linear layer to get the predictions
        output = self.linear_relu_stack(output)
        output = self.sigmoid(output)
        return output

    def train(self, train_data, max_epochs = None, batch_size = None, learning_rate = None, weight_decay = None, model_path = None, save_every = None, valid_size = None):
        """
        Train the model
        @param:
            train_data: list of train data
            max_epoches: max number of epoches
            batch_size: batch size
            learning_rate: learning rate
            weight_decay: weight decay
            model_path: path to save the model
            save_every: save the model every x epoches
            valid_size: size of validation set
        """
        # set the parameters
        if learning_rate is  None:
            learning_rate = self.cfg.MODEL.LR
        if batch_size is None:
            batch_size = self.cfg.MODEL.BATCH_SIZE
        if weight_decay is  None:
            weight_decay = self.cfg.MODEL.WEIGHT_DECAY
        if max_epochs is  None:
            max_epochs = self.cfg.MODEL.MAX_EPOCHS
        if save_every is  None:
            save_every = self.cfg.MODEL.SAVE_EVERY
        if model_path is  None:
            model_path = self.cfg.MODEL.MODEL_PATH
        if valid_size is  None:
            valid_size = self.cfg.MODEL.VALID_SIZE

        print('Training model...')
        # set optimizer
        print(self.parameters(), learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # set loss function
        loss_fn = torch.nn.BCELoss()
        # set scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=save_every, gamma=0.1)
    
        # validate the model
        total = len(train_data)
        valid_size = int(total * valid_size)
        valid_data = train_data[-valid_size:]
        train_data = train_data[:-valid_size]
        # train
        for epoch in range(max_epochs):
            # set train mode
            super(LM, self).train()
            # set loss
            loss = 0
            # set batch generator
            batch_generator = self.batch_generator(train_data, batch_size, shuffle=True)
            # train
            for text, match in enumerate(batch_generator):
                print(text, match)
                # set optimizer
                optimizer.zero_grad()
                # set input
                input = text
                # set target
                target = match
                # set output
                output = self(input)
                # set loss
                loss += loss_fn(output, target)
                # backward
                loss.backward()
                # optimize
                optimizer.step()
            # set scheduler
            scheduler.step()
            # set loss
            loss = loss / len(train_data)
            # print loss
            print('Epoch: {}/{}, Loss: {}'.format(epoch + 1, max_epochs, loss))
            # set scheduler
            scheduler.step()
            # save model
            if (epoch + 1) % save_every == 0:
                # validate the model
                self.validate(valid_data)
                self.save_model(model_path+'_'+str(epoch+1))
            
        # save model
        self.save_model(model_path+'_final') 

    def validate(self, valid_data, threshold = None):
        """
        Validate the model and caculate the accuracy
        @param:
            valid_data: list of valid data
            threshold: threshold to filter the predictions
        """
        if threshold is None:
            threshold = self.cfg.MODEL.THRESHOLD
        # caculate the accuracy
        accuracy = 0
        # set batch generator
        batch_generator = self.batch_generator(valid_data, self.cfg.MODEL.BATCH_SIZE, shuffle=False)
        # validate
        for text, match in enumerate(batch_generator):
            # set input
            input = text
            # set target
            target = match
            # set output
            output = self(input)
            # set accuracy
            accuracy += self.accuracy(output, target, threshold)
        # set accuracy
        accuracy = accuracy / len(valid_data)
        # print accuracy
        print('Validation accuracy: {}'.format(accuracy))

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
        output = self(input_data)
        # filter the predictions by threshold
        predictions = self.filter_predictions(output, threshold)
        return output

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

    def accuracy(self, output, target, threshold = None):
        '''
            caculate the accuracy 
            @param:
                output: list of output
                target: list of target
                threshold: threshold to filter the predictions
            @return:
                accuracy: accuracy
        '''
        if threshold is None:
            threshold = self.cfg.MODEL.THRESHOLD
        # filter the predictions by threshold
        predictions = self.filter_predictions(output, threshold)
        # caculate the accuracy
        accuracy = 0
        for i in range(len(predictions)):
            if predictions[i] == target[i]:
                accuracy += 1
        return accuracy / len(predictions)

    def batch_generator(self, data, batch_size, shuffle=False):
        """
        Generate batch data
        @param:
            data: list of data
            batch_size: batch size
            shuffle: whether to shuffle
        @return:
            batch: batch data
        """
        # set batch size
        batch_size = min(batch_size, len(data))
        # set batch generator
        batch_generator = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        # return batch generator
        return batch_generator
    
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



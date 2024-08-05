'''Feedforward Neural Network'''
# References
# https://saturncloud.io/blog/how-to-replace-multiple-values-in-one-column-using-pandas/
# https://stackoverflow.com/questions/58217005/how-to-reverse-label-encoder-from-sklearn-for-multiple-columns
# CMPUT 328 Assignment 2
# CMPUT 466 Assignment 2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

torch.manual_seed(466)

class FNN(nn.Module):
    '''The actual network'''
    def __init__(self, in_size, num_classes, dataset):
        super(FNN, self).__init__()

        self.num_classes = num_classes
        # all
        if dataset == 'all':
            self.fc1 = nn.Linear(in_size, 64)
            self.fc15 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128,256)
            self.fc3 = nn.Linear(256, self.num_classes) # 115 categories

        # shortened
        if dataset == 'short':
            self.fc1 = nn.Linear(in_size, 32)
            self.fc15 = nn.Linear(32, 100)  
            self.fc2 = nn.Linear(100,64)
            self.fc3 = nn.Linear(64, self.num_classes) # 34 categories

    def forward(self, x):
        x = x.view(x.size(0), -1)       
        output = F.softmax(self.fc3(torch.sigmoid(self.fc2(torch.relu(self.fc15(torch.sigmoid(self.fc1(x))))))))
        return output


class fnn(object):
    '''Trains a network'''
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None

    def solve(self, X_train, X_val, y_train, num_epochs=2):
        if self.dataset == 'short':
            lr = 0.1
        else:
            lr = 0.01 #all
        in_size = X_train.shape[1]
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        num_classes = len(np.unique(y_train_encoded))
        model = FNN(in_size, num_classes, self.dataset)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay= 0.0001)
        losstype = nn.CrossEntropyLoss()

        for epoch in range(1, num_epochs + 1):
            loss = self.train(epoch, X_train, y_train_encoded, model, optimizer, losstype)
            #print(f"Epoch {epoch}/{num_epochs}, Loss: {loss}")
        
        self.model = model

        return (self, label_encoder)

    def train(self, epoch, features, obs, model, optimizer, losstype):
        X = torch.from_numpy(features.values).float()  # Convert DataFrame to numpy array
        y = torch.from_numpy(obs).long()  # Convert to LongTensor for CrossEntropyLoss

        optimizer.zero_grad()
        outputs = model(X)
        loss = losstype(outputs, y)
        loss.backward()
        optimizer.step()

        return loss.item()
    
    def eval(self, features, label_encoder):
        '''Evaluates a model's accuracy using a validation set'''
        X = torch.from_numpy(features.values).float()  # Convert DataFrame to numpy array
        with torch.no_grad():
            output = self.model(X)
            _, predicted = torch.max(output, 1)
        #print(type(predicted))

        return label_encoder.inverse_transform(predicted.numpy())
    
    def eval_top_k_probs(self, features, label_encoder, top_k=3):
        '''Returns top-K predictions based on probabilities'''
        X = torch.from_numpy(features.values).float()  # Convert DataFrame to numpy array
        with torch.no_grad():
            output = self.model(X)
        
        # Sort probabilities for each instance in descending order
        top_k_indices = np.argsort(output.numpy(), axis=1)[:, -top_k:]
        # Map indices to class labels using the label_encoder
        top_k_preds = [label_encoder.inverse_transform(indices) for indices in top_k_indices]

        return top_k_preds

# Usage
def main(X_train, X_test, y_train, fnn_instance):
    return fnn_instance.solve(X_train, X_test, y_train, num_epochs=2)

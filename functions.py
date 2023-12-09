import pandas as pd
import numpy as np

import torch
import torch.nn as nn


# Класс нейронной сети
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_sizes, output_size):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.layer4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.layer5 = nn.Linear(hidden_sizes[4], output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        x = nn.Sigmoid()(x)
        x = self.layer3(x)
        x = nn.Sigmoid()(x)
        x = self.layer4(x)
        x = nn.Sigmoid()(x)
        x = self.layer5(x)
        x = nn.Sigmoid()(x)
        return x


# Функция тренировки нейронной сети
def train_net(epochs, loader, optimizer, model, criterion):
    for epoch in range(epochs):
        for X, y in loader:
            optimizer.zero_grad()
            prediction = model(X)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
    return model


def get_result(test, user_features, friend_features, input_size, result, model, result_size):
    h = 0
    for user in test['user_id'].values:
        current = pd.DataFrame()
        current['friend_id'] = friend_features['friend_id']
        prediction_list = []
        for friend in friend_features['friend_id'].values:
            difference = list(user_features.loc[user_features['user_id'] == user][f'{i}'].values[0] -
                              friend_features.loc[friend_features['friend_id'] == friend][f'{i}'].values[0]
                              for i in range(input_size))
            prediction = model(torch.tensor(difference, dtype=torch.float32))
            prediction_list.append(float(prediction))
        current['predictions'] = pd.Series(np.array(prediction_list))
        current = current.sort_values(by='predictions')
        print(h)
        h += 1
        current_list = [int(user)]
        for i in range(result_size):
            current_list.append(int(current.iloc[i]['friend_id']))
        result.loc[len(result.index)] = current_list
    return result

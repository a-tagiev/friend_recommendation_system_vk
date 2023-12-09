from settings import *
from functions import *

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Собирает данные
test = pd.read_csv('data/test.csv')
train = pd.read_csv('data/train.csv')
user_features = pd.read_csv('data/user_features.csv')
friend_features = pd.read_csv('data/friend_features.csv')

# Группирует в одной переменной данные для тренировки по user_id и friend_id
all_data = pd.merge(train, user_features, on='user_id')
all_data = pd.merge(all_data, friend_features, on='friend_id')

# Считает разницу между параметрами друга и юзера
for i in range(INPUT_SIZE):
    all_data[str(i)] = all_data[f'{i}_x'] - all_data[f'{i}_y']

# Фильтрует данные, дает итоговые данные
final_data = all_data[['user_id', 'friend_id'] + list(f'{i}' for i in range(INPUT_SIZE)) + ['friendship']]

# Разделяет данные для обучения
X = final_data.iloc[:, 2:-1].values
y = final_data['friendship'].values.reshape(-1, 1)

# Преобразовывает данные в тензоры
X_tensor = torch.tensor(X.astype(np.float32))
y_tensor = torch.tensor(y.astype(np.float32))

# Создает объект данных для загрузки
train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)


# Создает модель нейронной сети, оптимизатор и функцию потери
model = Net(INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
criterion = nn.MSELoss()

# Тренирует модель
model = train_net(EPOCH_NUMBER, train_loader, optimizer, model, criterion)


# Получаем требуемый результат
result = pd.DataFrame(columns=['user_id'] + [f'{i}' for i in range(RESULT_NUMBER)])
result = get_result(test, user_features, friend_features, INPUT_SIZE, result, model, RESULT_NUMBER)

# Записываем результат в файл
result.to_csv('data/result.csv', index=False)

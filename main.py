import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Net(nn.Module):
    def __init__(self, input, hidden, output):
        super(Net, self).__init__()

        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)

        x = self.layer2(x)
        x = nn.Sigmoid()(x)

        return x


INPUT_SIZE = 32
HIDDEN_SIZE = 10
OUTPUT_SIZE = 1
LEARNING_RATE = 0.01
EPOCH_NUMBER = 1000

test = pd.read_csv('data/test.csv')
train = pd.read_csv('data/train.csv')

user_features = pd.read_csv('data/user_features.csv')
friend_features = pd.read_csv('data/friend_features.csv')

all_data = pd.merge(train, user_features, on='user_id')
all_data = pd.merge(all_data, friend_features, on='friend_id')
grouped_data = all_data.groupby('user_id')

for i in range(INPUT_SIZE):
    all_data[str(i)] = all_data[f'{i}_x'] - all_data[f'{i}_y']

final_data = all_data[['user_id', 'friend_id'] + list(f'{i}' for i in range(INPUT_SIZE)) + ['friendship']]

x = final_data.iloc[:, 2:-1].values
y = final_data['friendship'].values.reshape(-1, 1)

x_tensor = torch.tensor(x.astype(np.float32))
y_tensor = torch.tensor(y.astype(np.float32))

train_dataset = TensorDataset(x_tensor, y_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)

model = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
device = torch.device("cuda")
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

loss = nn.BCELoss()

for epoch in range(EPOCH_NUMBER):
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        prediction = model(x)

        ls = loss(prediction, y)

        ls.backward()
        optimizer.step()

result_df = pd.DataFrame(columns=['user_id'] + [f'{i}' for i in range(0, 20)])
'''
grouped_data = all_data.groupby('user_id')
similarity_values = []
for user_id, user_data in grouped_data:
    for index, row in user_data.iterrows():
        friend_id = row['friend_id']
        usr_id = row['user_id']
        differences = [row[f'{i}_y'] - row[f'{i}_x'] for i in range(32)]
        preds = model(torch.tensor(differences, dtype=torch.float32))
        similarity_values.append((preds.item(), user_id, int(friend_id)))
        top = sorted(similarity_values, key=lambda k: k[0], reverse=True)

        result_df.loc[len(result_df.index)] = [[pd.Series(usr_id), pd.Series(top[0][2])]]
        # result_df.iloc[index][]
        print(result_df)

        # result_df = result_df.append({'user_id': user_id, 'friend_id': friend_id,
        # **{f'difference{i}': diff for i, diff in enumerate(differences)}},
        # ignore_index=True)
# print(result_df)
'''
h = 0
for user in test['user_id'].values:
    c_result = pd.DataFrame()
    c_result['friend_id'] = friend_features['friend_id']
    c_list = []
    for friend in friend_features['friend_id'].values:
        dif = list(user_features.loc[user_features['user_id'] == user][f'{i}'].values[0] -
                   friend_features.loc[friend_features['friend_id'] == friend][f'{i}'].values[0] for i in
                   range(INPUT_SIZE))
        pred = model(torch.tensor(dif, dtype=torch.float32))
        c_list.append(float(pred))
    c_result['predictions'] = pd.Series(np.array(c_list))
    c_result = c_result.sort_values(by='predictions', ascending=True)
    print(h)
    h += 1

    l = [int(user)]
    for i in range(20):
        l.append(int(c_result.iloc[i]['friend_id']))
    result_df.loc[len(result_df.index)] = l

result_df.to_csv('output.csv')

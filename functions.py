from cmath import sqrt
import pandas as pd
from math import exp
import numpy as np


users_for_main_user = []


def expon():
    expons = []
    for i in range(50, 0, -1):
        expons.append(exp(i))
    return expons

def search_parameter_two_user(coefficients, main_user, other_user):
    parameter_user = 0
    for i in range(len(coefficients)):
        parameter_user += (coefficients[i] * (main_user - other_user)) ** 2
    return sqrt(parameter_user)


def search_users_for_main_user(coefficients, main_user, users):
    global users_for_main_user
    users_for_main_user = []
    for user in users:
        users_for_main_user.append((user[0], search_parameter_two_user(coefficients, main_user, user)))

    users_for_main_user.sort()


def search_b_array(test_data):
    global users_for_main_user

    b_array = []
    for user in users_for_main_user:
        for i in range(len(test_data)):
            if user[0] == test_data[i][0]:
                b_array.append(i)
    return b_array


def count_function_Levenberg(coefficients, exponents, b_array):
    global users_for_main_user

    function_Levenberg = 0
    for i in range(len(users_for_main_user)):
        function_Levenberg += (b_array[i]) ** 2 * exponents[i]

    return [function_Levenberg, coefficients]


def diff_friend_and_user(user, friend, j):
    dfu = pd.read_csv('data/user_features.csv') #тут для юзера с айди 0
    u_par = dfu[str(j)]
    df = pd.read_csv('data/friend_features.csv')
    f_par = df[str(j)]
    dif = f_par.iloc(friend) - u_par.iloc[user] #xij - x0j
    return dif ** 2




def sort1(elem):
    return elem[1]


def sort2(elem):
    return elem[2]


def prepare_test_date(id_user):
    df = pd.read_csv("data/train.csv")
    df = df.sort_values('user_id')

    ind_start = 0
    for i in range(len(df)):
        if df["user_id"].iloc[i] == id_user:
            ind_start = i
            break

    test_data = []
    numb_friend = 0
    while df["user_id"].iloc[ind_start] == id_user:
        test_data.append([df['friend_id'].iloc[ind_start], df['friendship'].iloc[ind_start], df['timestamp'].iloc[ind_start]])
        if test_data[-1][1] == 1:
            numb_friend += 1
        ind_start += 1

    test_data.sort(key=sort1, reverse=True)

    friend_part = sorted(test_data[:numb_friend:], key=sort2, reverse=True)

    nfriend_part = sorted(test_data[numb_friend::], key=sort2)

    final_data = np.concatenate([friend_part, nfriend_part])
    return final_data


#Функция Миши



from cmath import sqrt


users_for_main_user = []



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


#Функция Миши



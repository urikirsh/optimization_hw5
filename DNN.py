import math
import numpy as np
import unittest
import numpy.testing as npt
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # necessary despite Pycharm thinking otherwise


def true_function_y(x, y):
    '''
    The function out DNN tries to learn, the "labels"
    '''
    return x * math.exp(-x**2 - y**2)


def phi_f(x: np.ndarray):
    '''
    Returns vector phi at vector x.
    '''
    return np.tanh(x).reshape((len(x), 1))


def phi_g(x: np.ndarray):
    '''
    Returns gradient of phi at vector x. A diagonal matrix
    '''
    try:
        g_i_i = 1 / (np.cosh(x)**2)
    except FloatingPointError:
        g_i_i = np.zeros(x.shape)
    return np.diag(np.ravel(g_i_i))


def dnn_forward(x: np.ndarray, params: dict):
    '''
    returns F(x, W)
    '''
    required_params = ['W1', 'W2', 'W3', 'b1', 'b2', 'b3']
    for par in required_params:
        assert par in params

    u1 = params['W1'].T @ x + params['b1']
    u2 = params['W2'].T @ phi_f(u1) + params['b2']
    u3 = params['W3'].T @ phi_f(u2) + params['b3']

    return u3


def error_f(out, y):
    '''
    returns Psi(r_i)
    '''
    return (out - y) ** 2


def dnn_error(x: np.ndarray, parameters: dict):
    '''
    Returns the value of Psi(x), given weights and biases
    '''

    out = dnn_forward(x, parameters)
    y = true_function_y(x[0], x[1])
    return error_f(out, y)


def analytic_calc_dir_grads_dnn_error(x: np.ndarray, parameters: dict, direction: str):
    '''
    Analaytic calculation of directional gradients
    :param x: The point at which we calculate the gradient
    :param parameters: DNN weights and biases
    :param direction: Name of the parameter that will be the direction of the gradient
    :return:
    '''
    assert direction in parameters

    y = true_function_y(x[0], x[1])
    out = dnn_forward(x, parameters)
    nabla_r_Psi = 2 * (out - y)
    if direction == 'b3':
        return nabla_r_Psi

    u1 = parameters['W1'].T @ x + parameters['b1']
    u2 = parameters['W2'].T @ phi_f(u1) + parameters['b2']

    if direction == 'W3':
        return nabla_r_Psi @ phi_f(u2).T

    b2_dir_der = (phi_g(u2) @ parameters['W3']) * nabla_r_Psi
    if direction == 'b2':
        return b2_dir_der.T

    if direction == 'W2':
        return b2_dir_der @ phi_f(u1).T

    b1_dir_der = phi_g(u1) @ parameters['W2'] @ b2_dir_der
    if direction == 'b1':
        return b1_dir_der.T

    assert direction == 'W1'

    return b1_dir_der @ x.T


def generate_bias(n: int, random=False):
    assert n > 0
    if random:
        return np.random.rand(n, 1) / math.sqrt(n)
    return np.zeros((n, 1))


def generate_weight(m: int, n: int):
    assert m > 0
    assert n > 0
    return np.random.randn(m, n) / math.sqrt(m * n)


def generate_params(random=True):
    '''
    Generate weights and biases for a DNN vie Xavier initialization
    :param random: If true returns random biases, if false biases are initialized as zeros
    :return: A dictionary of 3 weights and 3 biases
    '''
    params = dict()
    params['b1'] = generate_bias(4, random=random)
    params['b2'] = generate_bias(3, random=random)
    params['b3'] = generate_bias(1, random=random)
    params['W1'] = generate_weight(2, 4)
    params['W2'] = generate_weight(4, 3)
    params['W3'] = generate_weight(3, 1)
    return params


def numdiff_calc_dnn_error_grad(grad_of, x, params: dict, epsilon: float):
    '''
    calculate DNN error's gradients by numeric differences.
    '''
    assert epsilon > 0
    assert grad_of in params

    max_abs_val_of_x = abs(max(x.min(), x.max(), key=abs))
    x_dim = len(x)
    epsilon = pow(epsilon, 1 / x_dim) * max_abs_val_of_x
    assert epsilon > 0

    assert x.shape[1] == 1

    x_dim = params[grad_of].shape[0]
    y_dim = params[grad_of].shape[1]
    grad = np.zeros(params[grad_of].shape)
    for i in range(0, x_dim):
        for j in range(0, y_dim):
            params[grad_of][i][j] += epsilon
            right_f = dnn_error(x, params)
            params[grad_of][i][j] -= 2*epsilon
            left_f = dnn_error(x, params)
            diff = right_f - left_f
            assert diff.shape == (1, 1)
            diff = diff[0][0]
            grad[i][j] = diff / (2 * epsilon)
            # cleanup
            params[grad_of][i][j] += epsilon

    return grad.T


def pack_params(params):
    '''
    Takes an iterable of ndarrays and stacks them to one ndarray.
    '''
    pack_params.shapes = [param.shape for param in params]
    return np.hstack(np.ravel(param) for param in params).reshape(-1, 1)


def unpack_params(packed: np.ndarray):
    '''
    Unpacks parameters that have been packed with pack_params
    '''
    shapes = pack_params.shapes
    sizes = map(lambda x: x[0] * x[1], shapes)
    indexes = list(sizes)
    for i in range(len(indexes) - 1):
        indexes[i+1] += indexes[i]
    arrays = np.split(packed, indexes)
    return (arr.reshape(shape) for arr, shape in zip(arrays, shapes))


def dnn_error_ang_grad(x: np.ndarray, y, parameters):
    '''
    Return the DNN error and it's gradient as a stacked vector
    :param x: The DNN's output
    :param y: The true value (label)
    :param parameters: Weights and biases of the DNN, used for computing gradients
    :return:
    '''
    x = x.reshape((-1, 1))
    W1, W2, W3, b1, b2, b3 = unpack_params(parameters)
    param_dict = {'W1': W1, 'W2': W2, 'W3': W3, 'b1': b1, 'b2': b2, 'b3': b3}
    out = dnn_forward(x, param_dict)
    error = error_f(out, y)
    grad_W1 = analytic_calc_dir_grads_dnn_error(x, param_dict, 'W1').T
    grad_W2 = analytic_calc_dir_grads_dnn_error(x, param_dict, 'W2').T
    grad_W3 = analytic_calc_dir_grads_dnn_error(x, param_dict, 'W3').T
    grad_b1 = analytic_calc_dir_grads_dnn_error(x, param_dict, 'b1').T
    grad_b2 = analytic_calc_dir_grads_dnn_error(x, param_dict, 'b2').T
    grad_b3 = analytic_calc_dir_grads_dnn_error(x, param_dict, 'b3').T
    return np.array((error,
                     pack_params((grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3))))


def target_function(X, Y, parameters):
    return dnn_error_ang_grad(X.T, Y, parameters)


def get_target_f_of_params(X, Y):
    return lambda p: target_function(X=X, Y=Y, parameters=p)

'''
New code starts here. All code before originates from the previous exercise.
'''


def get_xavier_params():
    '''
    Xavier initialization of the parameters. Generate params was modified to make this
    happen
    '''
    params = generate_params(False)
    params = pack_params((params['W1'], params['W2'], params['W3'], params['b1'],
                         params['b2'], params['b3']))
    return params


def SGD(org_train_data, train_targets, params, alpha_0: float, decay_rate=0.5,
        num_epochs=1000, batch_size=32, shuffle=False):
    '''
    Stochastic Gradient Descent
    returns the optimized parameters as well as a list
    containing the L2 loss for each epoch
    '''

    train_data = np.copy(org_train_data)
    train_data = np.transpose(train_data)

    error_history = []

    for epoch in range(num_epochs):

        # Draw batch data
        if not shuffle:
            idx = np.random.randint(len(train_data), size=batch_size)
            batch_data = train_data[idx]
            batch_targets = train_targets[idx]
        else:
            merged_data = np.concatenate((train_data, train_targets), axis=1)
            np.random.shuffle(merged_data)
            train_data = merged_data[:, 0:2]
            train_targets = merged_data[:, 2]
            train_targets = train_targets.reshape(-1, 1)
            batch_data = train_data[:batch_size]
            batch_targets = train_targets[:batch_size]

        batch_grads = []
        batch_errors = []

        for i in range(batch_size):
            curr_error, curr_gradient = \
                target_function(batch_data[i], batch_targets[i], params)
            batch_grads.append(curr_gradient)
            batch_errors.append(curr_error)

        mean_err = np.mean(batch_errors, axis=0)
        error_history.append(mean_err)

        mean_grad = np.mean(batch_grads, axis=0)
        try:
            learning_rate = alpha_0 * math.exp(-epoch * decay_rate)
            params -= learning_rate * mean_grad
        except FloatingPointError:  # Underflow - the change in parameters is too small
            return params, error_history

    return params, error_history


def Adagrad(org_train_data, train_targets, params, alpha_0: float, decay_rate=0.5,
            num_epochs=100, batch_size=32, shuffle=False):
    '''
    returns the optimized parameters as well as a list
    containing the L2 loss for each epoch
    '''

    train_data = np.copy(org_train_data)
    train_data = np.transpose(train_data)

    error_history = []
    grad_squared = 0

    for epoch in range(num_epochs):
        if not shuffle:
            idx = np.random.randint(len(train_data), size=batch_size)
            batch_data = train_data[idx]
            batch_targets = train_targets[idx]
        else:
            merged_data = np.concatenate((train_data, train_targets), axis=1)
            np.random.shuffle(merged_data)
            train_data = merged_data[:, 0:2]
            train_targets = merged_data[:, 2]
            train_targets = train_targets.reshape(-1, 1)
            batch_data = train_data[:batch_size]
            batch_targets = train_targets[:batch_size]

        batch_grads = []
        batch_errors = []

        for i in range(batch_size):
            curr_error, curr_gradient = \
                target_function(batch_data[i], batch_targets[i], params)

            batch_grads.append(curr_gradient)
            batch_errors.append(curr_error)

        mean_err = np.mean(batch_errors, axis=0)
        error_history.append(mean_err)

        mean_grad = np.mean(batch_grads, axis=0)
        sq_sum = np.sum(np.square(mean_grad))
        assert sq_sum >= 0
        grad_squared += sq_sum
        learning_rate = alpha_0 * math.exp(-decay_rate * epoch)
        params -= learning_rate * mean_grad / (np.sqrt(grad_squared) + 1e-7)

    return params, error_history


def eval_test_set(X_test, Y_test, params):
    '''
    Gets the test set and labels, and optimized parameters. Evaluates the L2 loss over
    the train set for a DNN with given params and returns it
    '''
    losses = []
    X_test = np.transpose(X_test)
    for i in range(len(X_test)):
        losses.append(target_function(X_test[i], Y_test[i], params)[0])
    return np.mean(losses)


def __search_learning_rate(algorithm, alg_name: str, X_train, Y_train, X_test, Y_test,
                             params, alphas: list, decay_rates: list):
    results = pd.DataFrame()

    for init in alphas:
        for decay in decay_rates:
            learned_params, f_history = algorithm(X_train, Y_train, np.copy(params),
                                                  init, decay_rate=decay, num_epochs=900)
            train_loss = f_history[-1][0][0]

            test_loss = eval_test_set(X_test, Y_test, learned_params)
            print("Using", alg_name, "with initial learning rate", init,
                  "and learning decay rate", decay, "on training set, loss is",
                  train_loss)
            curr_res = pd.DataFrame.from_dict({"Algorithm": [alg_name],
                                               "Alpha 0": [init],
                                               "Decay rate": [decay],
                                               "Train loss": [train_loss],
                                               "Test loss": [test_loss]})
            results = pd.concat([results, curr_res])
    return results


def log_search_learning_rate(algorithm, alg_name: str, X_train, Y_train, X_test, Y_test,
                             params=None):
    init_learning_rates = np.logspace(-1, -6, 6, endpoint=True)
    decay_rates = np.logspace(0, -7, 8, endpoint=True)
    if params is None:
        params = get_xavier_params()
    return __search_learning_rate(algorithm, alg_name, X_train, Y_train, X_test, Y_test,
                                  params, init_learning_rates, decay_rates)


def lin_search_learning_rate(algorithm, alg_name: str, X_train, Y_train, X_test, Y_test,
                             params=None):
    alpha_part_1 = np.linspace(1e-2, 9e-2, 9, endpoint=True)
    alpha_part_2 = np.linspace(1e-1, 5e-1, 5, endpoint=True)
    init_learning_rates = np.concatenate((alpha_part_1, alpha_part_2))
    init_learning_rates = np.round(init_learning_rates, decimals=10)

    # decay_part_1 = np.linspace(1e-4, 9e-4, 9, endpoint=True)
    # decay_part_2 = np.linspace(1e-3, 1e-2, 10, endpoint=True)
    # decay_rates = np.concatenate((decay_part_1, decay_part_2))
    # decay_rates = np.round(decay_rates, decimals=10)
    decay_rates = [1e-6]
    if params is None:
        params = get_xavier_params()
    return __search_learning_rate(algorithm, alg_name, X_train, Y_train, X_test, Y_test,
                                  params, init_learning_rates, decay_rates)


def search_batch_size(algorithm, alg_name: str, X_train, Y_train, X_test, Y_test,
                      params, alpha_0, decay_rate):
    max_power_of_two = int(math.log(len(X_train[0]), 2))
    powers_of_two = np.logspace(4, max_power_of_two, (max_power_of_two - 4 + 1), base=2,
                                endpoint=True)
    batch_sizes = np.concatenate((powers_of_two, [len(X_train[0])]))

    results = pd.DataFrame()

    for batch in batch_sizes:
        batch = int(batch)
        learned_params, f_history = algorithm(X_train, Y_train, np.copy(params),
                                              alpha_0, decay_rate=decay_rate,
                                              num_epochs=1000, batch_size=batch)
        train_loss = f_history[-1][0][0]

        test_loss = eval_test_set(X_test, Y_test, learned_params)
        print("Using", alg_name, "with batch size", batch, ", training loss is",
              train_loss, "\ttest loss is", test_loss)
        curr_res = pd.DataFrame.from_dict({"Algorithm": [alg_name],
                                           "Batch size": [batch],
                                           "Train loss": [train_loss],
                                           "Test loss": [test_loss]})
        results = pd.concat([results, curr_res])
    return results


def plot_convergence(f_history: list, alg_name: str, data_set: str):
    # Plotting error graph
    f_history = [f_history[i][0][0] for i in range(0, len(f_history))]
    plt.figure(figsize=(8, 7))
    plt.plot(f_history)
    plt.semilogy()
    plt.xlabel('Number of iterations')
    plt.ylabel('$|F(x, W_k)-f(x_1, x_2)|^2$')
    plt.grid()
    plt.title(alg_name + ' of DNN trying to approximate $f(x_1, x_2) = x_1*exp(-x_1^2-x_2^2)$'
              + ' on ' + data_set + ' set')
    plt.show()


def main():
    np.seterr(all='raise')
    # plot the target function
    line = np.arange(-2, 2, .2)
    X1, X2 = np.meshgrid(line, line)
    vectorized_target_function = np.vectorize(true_function_y)
    Y = vectorized_target_function(X1, X2)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Y, cmap=plt.cm.coolwarm, alpha=.6)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1, x_2)$')
    plt.title('$f(x_1, x_2) = x_1*exp(-x_1^2-x_2^2)$')

    plt.show(block=False)

    # generate train and test data
    Ntrain = 500
    X_train = 4 * np.random.rand(2, Ntrain) - 2

    Ntest = 200
    X_test = 4 * np.random.rand(2, Ntest) - 2

    Y_train = np.zeros((Ntrain, 1))
    for i in range(0, Ntrain):
        Y_train[i] = vectorized_target_function(X_train[0][i], X_train[1][i])

    Y_test = np.zeros((Ntest, 1))
    for i in range(0, Ntest):
        Y_test[i] = vectorized_target_function(X_test[0][i], X_test[1][i])

    params = get_xavier_params()

    # sgd_tr_res = log_search_learning_rate(SGD, "SGD", X_train, Y_train, X_test, Y_test,
    #                                       params=params)
    # sgd_tr_res.to_csv('SGD_TR_losses.csv', index=False)
    # adagrad_tr_res = log_search_learning_rate(SGD, "Adagrad", X_train, Y_train, X_test,
    #                                           Y_test, params=params)
    # adagrad_tr_res.to_csv('Adagrad_TR_losses.csv', index=False)

    # sgd_tr_res = lin_search_learning_rate(SGD, 'SGD', X_train, Y_train, X_test, Y_test,
    #                                       params)
    # sgd_tr_res.to_csv('SGD_fine_TR_losses.csv', index=False)
    #
    # adagrad_tr_res = lin_search_learning_rate(Adagrad, 'Adagrad', X_train, Y_train, X_test,
    #                                           Y_test, params)
    # adagrad_tr_res.to_csv('Adagrad_fine_TR_losses.csv', index=False)

    opt_alpha_0 = 0.3  # Manually selected
    opt_decay = 1e-6  # Manually selected

    # batch_size_res_sgd = search_batch_size(SGD, 'SGD', X_train, Y_train, X_test, Y_test,
    #                                        params, opt_alpha_0, opt_decay)
    # batch_size_res_adagrad = search_batch_size(Adagrad, 'Adagrad', X_train, Y_train,
    #                                            X_test, Y_test, params, opt_alpha_0,
    #                                            opt_decay)
    # batch_size_results = pd.concat([batch_size_res_sgd, batch_size_res_adagrad])
    #
    # batch_size_results.to_csv('Batch_sizes.csv')

    opt_batch = 500  # Manually selected
    # learned_params, f_history = SGD(X_train, Y_train, np.copy(params), opt_alpha_0,
    #                                 decay_rate=opt_decay, num_epochs=1000,
    #                                 batch_size=opt_batch)
    # train_loss = f_history[-1][0][0]
    # test_loss = eval_test_set(X_test, Y_test, learned_params)
    # print("SGD with optimal parameters, with random draw has a training loss of",
    #       train_loss, "and test loss of", test_loss)
    #
    # learned_params, f_history = SGD(X_train, Y_train, np.copy(params), opt_alpha_0,
    #                                 decay_rate=opt_decay, num_epochs=1000,
    #                                 batch_size=opt_batch, shuffle=True)
    # train_loss = f_history[-1][0][0]
    # test_loss = eval_test_set(X_test, Y_test, learned_params)
    # print("SGD with optimal parameters, with reshuffle has a training loss of",
    #       train_loss, "and test loss of", test_loss)
    #
    # learned_params, f_history = Adagrad(X_train, Y_train, np.copy(params), opt_alpha_0,
    #                                     decay_rate=opt_decay, num_epochs=1000,
    #                                     batch_size=opt_batch)
    # train_loss = f_history[-1][0][0]
    # test_loss = eval_test_set(X_test, Y_test, learned_params)
    # print("Adagrad with optimal parameters, with random draw has a training loss of",
    #       train_loss, "and test loss of", test_loss)
    #
    # learned_params, f_history = Adagrad(X_train, Y_train, np.copy(params), opt_alpha_0,
    #                                     decay_rate=opt_decay, num_epochs=1000,
    #                                     batch_size=opt_batch, shuffle=True)
    # train_loss = f_history[-1][0][0]
    # test_loss = eval_test_set(X_test, Y_test, learned_params)
    # print("Adagrad with optimal parameters, with reshuffle has a training loss of",
    #       train_loss, "and test loss of", test_loss)

    opt_shuffle = True  # Manually selected

    learned_params, f_history = SGD(X_train, Y_train, np.copy(params), opt_alpha_0,
                                    decay_rate=opt_decay, num_epochs=1000,
                                    batch_size=opt_batch, shuffle=opt_shuffle)
    plot_convergence(f_history, 'SGD', 'train')

    learned_params, f_history = SGD(X_test, Y_test, np.copy(params), opt_alpha_0,
                                    decay_rate=opt_decay, num_epochs=1000,
                                    batch_size=200, shuffle=opt_shuffle)
    plot_convergence(f_history, 'SGD', 'test')

    learned_params, f_history = Adagrad(X_train, Y_train, np.copy(params), opt_alpha_0,
                                        decay_rate=opt_decay, num_epochs=1000,
                                        batch_size=opt_batch, shuffle=opt_shuffle)
    plot_convergence(f_history, 'Adagrad', 'train')

    learned_params, f_history = Adagrad(X_test, Y_test, np.copy(params), opt_alpha_0,
                                        decay_rate=opt_decay, num_epochs=1000,
                                        batch_size=200, shuffle=opt_shuffle)
    plot_convergence(f_history, 'Adagrad', 'test')

    print('success')


class task3_q_2 (unittest.TestCase):
    '''
    Unit test class, including the task 3 question 2 test
    '''

    def test_target_function(self):
        '''
        checks correctness of the target function, sanity check
        '''
        npt.assert_almost_equal(true_function_y(0, 0), 0)
        npt.assert_almost_equal(true_function_y(0, 17), 0)
        npt.assert_almost_equal(true_function_y(1, 0), np.exp(-1))


    def test_generate_params(self):
        '''
        Tests the generate params function
        '''
        params = generate_params()
        self.assertTrue(isinstance(params, dict))
        self.assertEqual(6, len(params))
        required_params = ['W1', 'W2', 'W3', 'b1', 'b2', 'b3']
        for par in required_params:
            self.assertTrue(par in params)

    def test_grad_numdiff(self):
        '''
        A singular test of correctness of our analytical gradients.
        '''
        params = generate_params()
        x = 2 * np.random.rand(2, 1) - 1
        epsilon = pow(2, -30)

        ready_tests = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        for test in ready_tests:
            anal = analytic_calc_dir_grads_dnn_error(x, params, test)
            numeric = numdiff_calc_dnn_error_grad(test, x, params, epsilon)
            npt.assert_almost_equal(numeric, anal)

    def test_stress_grad_numdiff(self):
        '''
        TASK 3 QUESTION 2 TEST
        '''
        for i in range(0, 100):
            self.test_grad_numdiff()

    def test_packing(self):
        '''
        check that packing and unpacking functions work
        '''
        a1 = np.array([[4, 5, 6], [41, 51, 63], [1, 2, 1]])
        a2 = np.array([[100]])
        a3 = np.array([[411, 225, 446, 55], [411, 225, 446, 55]])
        p = pack_params((a1, a2, a3))
        b1, b2, b3 = unpack_params(p)
        npt.assert_equal(a1, b1)
        npt.assert_equal(a2, b2)
        npt.assert_equal(a3, b3)


if __name__ == "__main__":
    main()
    # unittest.main()


import numpy as np

def feed_forward(inputs: list):
    inputs = [1] + inputs # add bias
    inputs = np.array(inputs)
    n = len(inputs)
    l1 = np.random.rand(n, 3)
    l2 = np.random.rand(3, 2)

    x = np.dot(inputs, l1)
    x = np.expit(x)
    x = np.dot(x, l2)
    x = np.expit(x)
    return np.argmax(x) # 0 for no lift, 1 for lift
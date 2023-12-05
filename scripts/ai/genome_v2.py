import numpy as np
import scipy.special as sp

def feed_forward(inputs: list):
    inputs = [1] + inputs # add bias
    inputs = np.array(inputs)
    n = len(inputs)
    l1 = np.random.rand(n, 3)
    l2 = np.random.rand(3, 2)

    x = sp.expit(sp.expit(inputs.dot(l1)).dot(l2))
    return np.argmax(x) # 0 for no lift, 1 for lift
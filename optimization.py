import numpy as np
from tqdm import tqdm

def gauss(t, r, window_size):
    """
    :param t: index of current point
    :param r: index of point in window
    :param window_size: size of window over which gaussian is applied
    :return:
            returns spatial gaussian weights over a window size
    """
    return np.exp((-9*(r-t)**2)/window_size**2)

def optimize_path(c, iterations=100, window_size=6):
    """
    :param c: original camera trajectory
    :param iterations: iteration number
    :param window_size: hyperparameter for the smoothness term
    :return:
            returns optimized gaussian smooth camera trajectory
    """
    lambda_t = 100
    p = np.empty_like(c)

    W = np.zeros((c.shape[2], c.shape[2]))
    for t in range(W.shape[0]):
        for r in range(int(-window_size/2), int(window_size/2)+1):
            if(t+r < 0 or t + r >= W.shape[1] or r == 0):
                continue
            W[t, t + r] = gauss(t, t + r, window_size)

    gamma = 1 + lambda_t * np.dot(W, np.ones((c.shape[2],)))

    bar = tqdm(total=c.shape[0]*c.shape[1])
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            P = np.asarray(c[i, j, :])
            for iteration in range(iterations):
                P = np.divide(c[i, j, :] + lambda_t*np.dot(W, P), gamma)
            p[i, j, :] = np.asarray(P)
            bar.update(1)

    bar.close()
    return p
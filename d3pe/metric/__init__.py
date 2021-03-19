import numpy as np
from scipy.stats import pearsonr, spearmanr

from typing import *

def RC_score(real_score : List[float], estimate_score : List[float]):
    '''
        Rank correlation mesures the rank relation coeffient between groudtruth and estimate values.
        The higher the value, the more you can trust the relative relationship estimated by certain algorithms.
    '''
    return spearmanr(real_score, estimate_score)[0]

def TopK_score(real_score : List[float], estimate_score : List[float], k : int = 1, mode : str = 'mean'):
    '''
        TopK score measure how the top k chosen by the estimate scores behave within the set of candidate policies.
        It reflect the chance of goodness if you are allow to run k tests on the real environment.
    '''
    max_score = max(real_score)
    min_score = min(real_score)
    index = np.argsort(estimate_score)[-k:]
    real_score = (np.array(real_score)[index] - min_score) / (max_score - min_score)
    if mode == 'mean':
        return real_score.mean()
    else:
        return real_score.max()
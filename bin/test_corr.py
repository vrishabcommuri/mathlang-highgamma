from context import mathlang
from mathlang import *
from mathlang.trf import *

import eelbrain as eel

from scipy import stats


if __name__ == '__main__':
    a = get_carrier(1, load_many_wavs_ssjamesmath, 180)
    b = get_envelope(1, load_many_spectrograms_ssjamesmath, 180)
    print(stats.pearsonr(a.x.flatten(),b.x.flatten()))
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from hyperparameters import PAST_HISTORY
from vis_dataset import StockDataset

MODEL_PATH = "adani.pt"
DSET_PATH = "hdfc.csv"

def scaleup (yout): 
    return yout*(2495. - 767.7) + 767.7

def sim_dataset(dset):
    j = 30
    print(dset.shape[0])
    while j < dset.shape[0] - PAST_HISTORY:
        idx = j+PAST_HISTORY
        arr = []
        for i in range(PAST_HISTORY):
            x = i-PAST_HISTORY
            arr.append(dset[idx+x:idx+PAST_HISTORY+x])
        j += 1
        if arr[-1].shape[0] < 30: break
        yield scaledn(np.array(arr))

if __name__ == "__main__":
    values = []
    real = []

    model = torch.load(MODEL_PATH)
    dl = DataLoader(StockDataset(DSET_PATH),1)
    for xin,y in tqdm(dl):
        y = scaleup(y)
        yout = model(xin.float())
        yout = scaleup(yout)
        real.append(y.detach().numpy())
        values.append(yout.detach().numpy())

    plt.plot(real)
    print("\n".join([str(float(v)) for v in values]))
    plt.plot(values,"r", alpha=0.7)
    plt.show()
    figure = plt.gcf()
    figure.set_size_inches(10,12)

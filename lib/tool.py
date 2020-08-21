from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import sys

def split_dataset(dataset, split_rate=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=split_rate)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def stats_bar(*message):
    sys.stdout.write("\r")
    for msg in message:
        sys.stdout.write(str(msg))
    sys.stdout.flush()
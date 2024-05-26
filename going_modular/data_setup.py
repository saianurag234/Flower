from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

from .common import *

NUM_WORKERS = os.cpu_count()

NORMALIZE_DICT = {
    'breast': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
}


def split_data_client(dataset, num_clients, seed):

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * (num_clients - 1)
    lengths += [len(dataset) - sum(lengths)]
    ds = random_split(dataset, lengths, torch.Generator().manual_seed(seed))
    return ds


def load_datasets(num_clients: int, batch_size: int, splitter=10):
    """
    This function is used to load the dataset and split it into train and test for each client.
    :param num_clients: the number of clients
    :param batch_size: the batch size
    :param seed: the seed for the random split
    :param num_workers: the number of workers
    :param splitter: percentage of the training data to use for validation. Example: 10 means 10% of the training data
    :param dataset: the name of the dataset in the data folder
    :param data_path: the path of the data folder
    :param data_path_val: the absolute path of the validation data (if None, no validation data)
    :return: the train and test data loaders
    """

    seed = 42

    list_transforms = [transforms.ToTensor(
    ), transforms.Normalize(**NORMALIZE_DICT['breast'])]

    transformer = transforms.Compose(list_transforms)

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create PyTorch datasets
    trainset = TensorDataset(X_train_tensor, y_train_tensor)
    testset = TensorDataset(X_test_tensor, y_test_tensor)

    datasets_train = split_data_client(trainset, num_clients, seed)

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for i in range(num_clients):
        len_val = int(len(datasets_train[i]) * splitter / 100)
        len_train = len(datasets_train[i]) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(
            datasets_train[i], lengths, torch.Generator().manual_seed(seed))
        trainloaders.append(DataLoader(
            ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))

    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader

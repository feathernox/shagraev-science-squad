import pickle
import numpy as np
import PIL

from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms


MYPATH = '../../../../storage/feathernox/'

MNIST_AUGMENTATION = transforms.Compose([
    transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.75, 1.33), shear=10,
                            resample=PIL.Image.BICUBIC),
    transforms.ToTensor()
])
MNIST_SIZES = [10, 60, 100, 300]

class MySubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        
    def __getitem__(self, index):
        index = self.indices[index]
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.indices)


class MyMNISTSubset(MySubset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices, transform=transform)
        
    def get_targets(self):
        return self.dataset.targets[self.indices].cpu().data.numpy()

    
def get_labeled_unlabeled_split(labels, class_size=10, n=5):
    labels_indices = {}
    for i in range(len(labels)):
        if labels[i] not in labels_indices:
            labels_indices[labels[i]] = [i]
        else:
            labels_indices[labels[i]].append(i)
    
    for _ in range(n):
        labeled = []
        unlabeled = []
        for k in labels_indices.keys():
            np.random.shuffle(labels_indices[k])
            labeled += labels_indices[k][:class_size]
            unlabeled += labels_indices[k][class_size:]
        yield np.array(labeled), np.array(unlabeled)
        
    
def create_mnist_train_val_test_split(
        val_size=10000, make_new=False,
        file_split='aux_files/mnist_train_test_split.pkl',
        random_state=None):
    mnist_train = MNIST(root=MYPATH, train=True, download=True)
    mnist_test = MNIST(root=MYPATH, train=False, download=True)
    
    if make_new:
        inds_train, inds_val = train_test_split(
            np.arange(len(mnist_train)),
            test_size=val_size,
            stratify=mnist_train.targets.cpu().data.numpy(),
            random_state=random_state
        )
        with open(file_split, 'wb') as f:
            pickle.dump([inds_train, inds_val], f)
    else:
        with open(file_split, 'rb') as f:
            inds_train, inds_val = pickle.load(f)
    
    subset_train = MyMNISTSubset(mnist_train, inds_train)
    subset_val = MyMNISTSubset(mnist_train, inds_val)
    
    return subset_train, subset_val, mnist_test


def create_mnist_labeled_unlabeled_splits(subset_train, class_size=10, n=5,
                                          path_split='aux_files/', make_new=False):
    if make_new:
        for i, split in enumerate(get_labeled_unlabeled_split(
                subset_train.get_targets(), class_size=class_size, n=n)):
            inds_labeled, inds_unlabeled = \
                subset_train.indices[split[0]], subset_train.indices[split[1]]
            with open(path_split + 'mnist_size_{}_split_{}.pkl'.format(class_size, i), 'wb') as f:
                pickle.dump([inds_labeled, inds_unlabeled], f)
    
    for i in range(n):
        with open(path_split + 'mnist_size_{}_split_{}.pkl'.format(class_size, i), 'rb') as f:
            inds_labeled, inds_unlabeled = pickle.load(f)
        mnist_labeled, mnist_unlabeled = \
            MyMNISTSubset(subset_train.dataset, inds_labeled), \
            MyMNISTSubset(subset_train.dataset, inds_unlabeled)
        yield mnist_labeled, mnist_unlabeled

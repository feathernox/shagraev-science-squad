import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader


def create_embeddings(model, loader):
    X_embeddings = []
    for X_batch, _ in tqdm(loader):
        batch_embeddings = model(X_batch.cuda()).cpu().data.numpy()
        X_embeddings.append(batch_embeddings)
    X_embeddings = np.vstack(X_embeddings)
    return X_embeddings


def extend_sample(model, labeled, unlabeled, val, test, batch_size=32):
    K_NEIGHBORS = [1, 3, 7, 15]
    
    model.train(False)
    
    labeled_loader = DataLoader(labeled, batch_size=batch_size, shuffle=False)
    labeled_X = create_embeddings(model, labeled_loader)
    labeled_y = labeled.get_targets()
        
    unlabeled_loader = DataLoader(unlabeled, batch_size=batch_size, shuffle=False)
    unlabeled_X = create_embeddings(model, unlabeled_loader)
    
    extended_samples = {}
    index = KDTree(unlabeled_X)
    for k_neighbors in K_NEIGHBORS:
        extended_indices = index.query(labeled_X, return_distance=False, k=k_neighbors)
        extended_y = np.tile(labeled_y.reshape(-1, 1), (1, k_neighbors)).ravel()  
        extended_X = []
        for i in tqdm(extended_indices.ravel()):
            extended_X.append(np.array(index.data[i]))
    
        extended_X = np.stack(extended_X)
        extended_X = np.vstack([labeled_X, extended_X])
        extended_y = np.hstack([labeled_y, extended_y])
        extended_samples[k_neighbors + 1] = {'X': extended_X, 'y': extended_y}
        
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    val_X = create_embeddings(model, val_loader)
    val_y = val.get_targets()
    
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    test_X = create_embeddings(model, test_loader)
    test_y = test.targets.cpu().data.numpy()
    
    result = {
        'extended': extended_samples,
        'val': {'X': val_X, 'y': val_y},
        'test': {'X': test_X, 'y': test_y}
    }
    return result

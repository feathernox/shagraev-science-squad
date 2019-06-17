import matplotlib.pyplot as plt
import pickle
import torch

from IPython.display import clear_output


class PlotCallback:
    def __init__(self, train_subset, patience=10,
                 path_model='aux_files/best_model.pkl',
                 path_plot='aux_files/graph.png',
                 path_callback='aux_files/callback.pkl'):
        self.train_len = len(train_subset)
        self.path_model = path_model
        self.path_plot = path_plot
        self.path_callback = path_callback
        
        self.n_epoch = 0
        self.n_batch = 0
        self.current_train_loss = 0
        self.train_loss = []
        self.val_loss = []
        self.val_accuracy = []
        
        self.patience = patience
        self.best_epoch = 0
        self.wait = 0
        
    def on_batch_end(self, train_loss_value):
        self.n_batch += 1
        self.current_train_loss += train_loss_value
        
    def on_epoch_end(self, model, val_loss, val_acc):
        clear_output()
        self.current_train_loss /= self.train_len
        self.train_loss.append(self.current_train_loss)
        self.val_loss.append(val_loss)
        self.val_accuracy.append(val_acc)
        self.current_train_loss = 0
        self.plot()
        
        flag_break = False
        if val_acc >= self.val_accuracy[self.best_epoch]:
            self.best_epoch = self.n_epoch
            torch.save(model.state_dict(), self.path_model)
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                flag_break = True
        
        self.n_epoch += 1
        
        with open(self.path_callback, 'wb') as f:
            pickle.dump(self, f)
        
        return flag_break
        
    def plot(self):
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, label='train')
        plt.plot(self.val_loss, label='val')
        plt.grid(ls=":")
        plt.legend(fontsize=12)
        plt.xlabel('num epoch', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracy)
        plt.grid(ls=":")
        plt.xlabel('num epoch', fontsize=14)
        plt.ylabel('val accuracy', fontsize=14)
        plt.savefig(self.path_plot)
        plt.show()

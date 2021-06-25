import torch as T
from torch import utils
import numpy as np
import time

from regression_net import Net


class ComprehensionDataset(utils.data.Dataset):

    def __init__(self, src_file, m_rows=None):
        all_xy = np.loadtxt(src_file, max_rows=m_rows,
        usecols=[0,1,2,3,4,5,6,7,8,9], delimiter=",",
        comments="#", skiprows=1, dtype=np.float32)

        tmp_x = all_xy[:,[0,1,2,3,4,5,6,7,8]]
        tmp_y = all_xy[:,9].reshape(-1,1)    # 2-D required

        self.x_data = T.tensor(tmp_x, \
        dtype=T.float32)
        self.y_data = T.tensor(tmp_y, \
        dtype=T.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        preds = self.x_data[idx,:]  # or just [idx]
        score = self.y_data[idx,:] 
        return (preds, score)       # tuple of two matrices 

def accuracy(model, ds, pct):
    # assumes model.eval()
    # percent correct within pct of true house price
    total = 0
    abs_delta = 0

    for i in range(len(ds)):
        (X, Y) = ds[i]            # (predictors, target)
        with T.no_grad():
            oupt = model(X)         # computed price

        abs_delta += np.abs(oupt.item() - Y.item())

        total += 1

    acc = abs_delta / total
    return acc

def main():
    train_file = "../data/gr3/train_regression.csv"
    train_ds = ComprehensionDataset(train_file)

    test_file = "../data/gr3/test_regression.csv"
    test_ds = ComprehensionDataset(test_file)

    bat_size = 10
    train_ldr = T.utils.data.DataLoader(train_ds,
        batch_size=bat_size, shuffle=True)

    # 2. create network
    net = Net()

    # 3. train model
    max_epochs = 500
    ep_log_interval = 50
    lrn_rate = 0.005

    loss_func = T.nn.MSELoss()
    # optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
    optimizer = T.optim.Adam(net.parameters(), lr=lrn_rate)

    print("\nbat_size = %3d " % bat_size)
    print("loss = " + str(loss_func))
    print("optimizer = Adam")
    print("max_epochs = %3d " % max_epochs)
    print("lrn_rate = %0.3f " % lrn_rate)

    print("\nStarting training with saved checkpoints")
    best_loss = float('inf')
    net.train()  # set mode
    for epoch in range(0, max_epochs):
        T.manual_seed(1+epoch)  # recovery reproducibility
        epoch_loss = 0  # for one full epoch

        for (batch_idx, batch) in enumerate(train_ldr):
            (X, Y) = batch                 # (predictors, targets)
            optimizer.zero_grad()          # prepare gradients
            oupt = net(X)                  # predicted prices
            loss_val = loss_func(oupt, Y)  # avg per item in batch
            epoch_loss += loss_val.item()  # accumulate avgs
            loss_val.backward()            # compute gradients
            optimizer.step()               # update wts

        if epoch % ep_log_interval == 0:
            print("epoch = %4d   loss = %0.4f" % (epoch, epoch_loss))

        # save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            fn = "saved_weights_regression.pt"

            info_dict = { 
                'epoch' : epoch,
                'net_state' : net.state_dict(),
                'optimizer_state' : optimizer.state_dict() 
            }
            T.save(info_dict, fn)

    print("Done ")

    # 4. evaluate model accuracy
    print("\nComputing model accuracy")
    net.eval()
    acc_train = accuracy(net, train_ds, 0.10) 
    print("Accuracy on train data = %0.4f" % acc_train)
    acc_test = accuracy(net, test_ds, 0.10) 
    print("Accuracy on test data = %0.4f" % acc_test)

if __name__ == '__main__':
    main()
from pathlib import Path
import copy
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import torch.optim as optim
import random
from sklearn.model_selection import train_test_split
import sys


class MLP(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        # Initialize the modules we need to build the network
        self.linear1 = nn.Linear(num_inputs, 10)
        self.linear2 = nn.Linear(10, 15)
        self.linear3 = nn.Linear(15, 5)
        self.output = nn.Linear(5, num_outputs)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.output(x)
        return x

def loss_fn(output, target, error):
    loss = torch.mean(torch.square((output - target)/error))
    return loss

def smooth(data, window_len=10, window='hanning', keep_original=False):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    """
    # TODO: add comnparison
    window_len += (window_len + 1) % 2
    s = np.r_['-1', data[:, window_len - 1:0:-1], data, data[:, -2:-window_len - 1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    # y=np.convolve(w/w.sum(),s,mode='valid')
    surplus_data = int((window_len - 1) * 0.5)
    return np.apply_along_axis(lambda m: np.convolve(m, w / w.sum(), mode='valid'), axis=1, arr=s)[:,
           surplus_data:-surplus_data]

def smooth_clusters(image, clusters, window_len = None):
    smoothed_clusters = np.zeros((len(clusters)), dtype = object)
    for i in range(len(clusters)):
        smoothed_clusters[i] = smooth(clusters[i])
    return smoothed_clusters

def derivative_clusters(image, clusters):
    dx = image.ddeltaE
    der_clusters = np.zeros((len(clusters)), dtype = object)
    for i in range(len(clusters)):
        der_clusters[i] = (clusters[i][:,1:]-clusters[i][:,:-1])/dx
    return der_clusters


def find_dE1(image, dy_dx, y_smooth, fct=0.7):
    # crossing
    # first positive derivative after dE=0:

    crossing = (dy_dx > 0)
    if not crossing.any():
        print("shouldn't get here")
        up = np.argmin(np.absolute(dy_dx)[np.argmax(y_smooth) + 1:]) + np.argmax(y_smooth) + 1
    else:
        up = np.argmax(crossing[np.argmax(y_smooth) + 1:]) + np.argmax(y_smooth) + 1
    pos_der = image.deltaE[up]
    return pos_der*fct


def determine_dE1_new(image, dy_dx_clusters, y_smooth_clusters, check_with_user=False):
    dy_dx_avg = np.zeros((len(y_smooth_clusters), image.l - 1))
    dE1_clusters = np.zeros(len(y_smooth_clusters))
    for i in range(len(y_smooth_clusters)):
        dy_dx_avg[i, :] = np.average(dy_dx_clusters[i], axis=0)
        y_smooth_cluster_avg = np.average(y_smooth_clusters[i], axis=0)
        dE1_clusters[i] = find_dE1(image, dy_dx_avg[i, :], y_smooth_cluster_avg)

    if not check_with_user:
        return dE1_clusters

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if len(colors) < len(y_smooth_clusters):
        print("thats too many clusters to effectively plot, man")
        return dE1_clusters
        # TODO: be kinder
    der_deltaE = image.deltaE[:-1]
    plt.figure()
    for i in range(len(y_smooth_clusters)):
        dx_dy_i_avg = dy_dx_avg[i, :]
        # dx_dy_i_std = np.std(dy_dx_clusters[i], axis = 0)
        ci_low = np.nanpercentile(dy_dx_clusters[i], 16, axis=0)
        ci_high = np.nanpercentile(dy_dx_clusters[i], 84, axis=0)
        plt.fill_between(der_deltaE, ci_low, ci_high, color=colors[i], alpha=0.2)
        plt.vlines(dE1_clusters[i], -3E3, 2E3, ls='dotted', color=colors[i])
        if i == 0:
            lab = "vacuum"
        else:
            lab = "sample cl." + str(i)
        plt.plot(der_deltaE, dx_dy_i_avg, color=colors[i], label=lab)
    plt.plot([der_deltaE[0], der_deltaE[-1]], [0, 0], color='black')
    plt.title("derivatives of EELS per cluster, and range of first \npositive derivative of EELSs per cluster")
    plt.xlabel("energy loss [eV]")
    plt.ylabel("dy/dx")
    plt.legend()
    plt.xlim(np.min(dE1_clusters) / 4, np.max(dE1_clusters) * 2)
    plt.ylim(-3e3, 2e3)

def find_scale_var(inp, min_out = 0.1, max_out=0.9):
    a = (max_out - min_out)/(inp.max()- inp.min())
    b = min_out - a*inp.min()
    return [a, b]


def scale(inp, ab):
    """
    min_inp = inp.min()
    max_inp = inp.max()

    outp = inp/(max_inp-min_inp) * (max_out-min_out)
    outp -= outp.min()
    outp += min_out

    return outp
    """

    return inp * ab[0] + ab[1]

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def train_nn_scaled(image, spectra, n_rep=500, n_epochs=30000, lr=1e-3, added_dE1=0.3, path_to_models="models",
                    display_step=1000):
    """training also on intensity, so only one model per image, instead of one model per cluster"""
    """
    if hasattr(image, "name"):
        path_to_models = image.name + "_" + path_to_models

    if not os.path.exists(path_to_models):
        Path(path_to_models).mkdir(parents=True, exist_ok=True)
    else:
        ans = input("The directory " + path_to_models + " already exists, if there are trained models " +
                    "in this folder, they will be overwritten. Do you want to continue? \n"+
                    "yes [y], no [n], define new path[dp]\n")
        if ans[0] == 'n':
            return
        elif not ans[0] == 'y':
            path_to_models = input("Please define the new path: \n")
    """

    print(spectra.shape)
    sys.exit()

    if display_step is None:
        print_progress = False
        display_step = 1E6
    else:
        print_progress = True


    loss_test_reps = np.zeros(n_rep)
    n_data = image.l

    # data_sigma = np.zeros((n_data,1))
    # sigma_clusters = np.zeros((image.n_clusters, image.l))
    # for cluster in range(image.n_clusters):
    #     ci_low = np.nanpercentile(np.log(spectra[cluster]), 16, axis=0)
    #     ci_high = np.nanpercentile(np.log(spectra[cluster]), 84, axis=0)
    #     sigma_clusters[cluster, :] = np.absolute(ci_high - ci_low)
        # data_sigma[cluster*image.l : (cluster+1)*image.l,0] = np.absolute(ci_high-ci_low)

    # new??? #TODO: verplaats dit naar determine_dE1
    wl1 = round(image.l / 20)
    wl2 = wl1 * 2
    units_per_bin = 4
    nbins = round(image.l / units_per_bin)  # 150
    spectra_smooth = smooth_clusters(image, spectra, wl1)
    dy_dx = derivative_clusters(image, spectra_smooth)
    smooth_dy_dx = smooth_clusters(image, dy_dx, wl2)
    # dE1s = find_clusters_dE1(image, smooth_dy_dx, spectra_smooth)

    dE1 = determine_dE1_new(image, smooth_dy_dx, spectra_smooth) * added_dE1  # dE1s, dy_dx)
    # image.dE1 = dE1

    # TODO: instead of the binned statistics, just use xth value to dischart -> neh says Juan
    times_dE1 = 3
    min_dE2 = image.deltaE.max() - image.ddeltaE * image.l * 0.05  # at least 5% zeros at end
    dE2 = np.minimum(times_dE1 * dE1,
                     min_dE2)  # minimal 1eV at the end with zeros #determine_dE2_new(image, spectra_smooth, smooth_dy_dx)#[0], nbins, dE1)

    if print_progress: print("dE1 & dE2:", np.round(dE1, 3), dE2)

    ab_deltaE = find_scale_var(image.deltaE)
    deltaE_scaled = scale(image.deltaE, ab_deltaE)

    # all_spectra = image.data
    # all_spectra[all_spectra<1] = 1
    # int_log_I = np.log(np.sum(all_spectra, axis=2)).flatten()

    all_spectra = np.empty((0, image.l))

    for i in range(len(spectra)):
        all_spectra = np.append(all_spectra, spectra[i], axis=0)

    int_log_I = np.log(np.sum(all_spectra, axis=1)).flatten()
    ab_int_log_I = find_scale_var(int_log_I)
    del all_spectra

    if not os.path.exists(path_to_models + "scale_var.txt"):
        np.savetxt(path_to_models + "scale_var.txt", ab_int_log_I)

    if not os.path.exists(path_to_models + "dE1.txt"):
        # np.savetxt(path_to_models+ "/dE1" + str(bs_rep_num) + ".txt", dE1)
        np.savetxt(path_to_models + "dE1.txt", np.vstack((image.clusters, dE1)))


    for i in range(n_rep):
        if print_progress: print("Started training on replica number {}".format(i) + ", at time ", dt.datetime.now())
        data = np.empty((0, 1))
        data_x = np.empty((0, 2))
        data_sigma = np.empty((0, 1))


        n_cluster = len(spectra[cluster])
        idx = random.randint(0, n_cluster - 1)
        # data[cluster*image.l : (cluster+1)*image.l,0] = np.log(spectra[cluster][idx])
        select1 = len(image.deltaE[image.deltaE < dE1[cluster]])
        select2 = len(image.deltaE[image.deltaE > dE2[cluster]])
        data = np.append(data, np.log(spectra[cluster][idx][:select1]))
        data = np.append(data, np.zeros(select2))

        pseudo_x = np.ones((select1 + select2, 2))
        pseudo_x[:select1, 0] = deltaE_scaled[:select1]
        pseudo_x[-select2:, 0] = deltaE_scaled[-select2:]
        int_log_I_idx_scaled = scale(np.log(np.sum(spectra[cluster][idx])), ab_int_log_I)
        pseudo_x[:, 1] = int_log_I_idx_scaled

        data_x = np.concatenate((data_x, pseudo_x))  # np.append(data_x, pseudo_x)

        data_sigma = np.append(data_sigma, sigma_clusters[cluster][:select1])
        data_sigma = np.append(data_sigma, 0.8 * np.ones(select2))

        model = MLP(num_inputs=1, num_outputs=1)
        model.apply(weight_reset)
        # optimizer = optim.RMSprop(model.parameters(), lr=6 * 1e-3, eps=1e-5, momentum=0.0, alpha = 0.9)
        optimizer = optim.Adam(model.parameters(), lr=lr)


        train_x, test_x, train_y, test_y, train_sigma, test_sigma = train_test_split(data_x, data, data_sigma,
                                                                                     test_size=0.4)

        N_test = len(test_x)
        N_train = len(train_x)

        test_x = test_x.reshape(N_test, 2)
        test_y = test_y.reshape(N_test, 1)
        train_x = train_x.reshape(N_train, 2)
        train_y = train_y.reshape(N_train, 1)
        train_sigma = train_sigma.reshape(N_train, 1)
        test_sigma = test_sigma.reshape(N_test, 1)

        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)
        train_sigma = torch.from_numpy(train_sigma)
        test_x = torch.from_numpy(test_x)
        test_y = torch.from_numpy(test_y)
        test_sigma = torch.from_numpy(test_sigma)

        # train_data_x, train_data_y, train_errors = get_batch(i)
        # loss_train = np.zeros(n_epochs)
        loss_test = np.zeros(n_epochs)
        loss_train_n = np.zeros(n_epochs)
        min_loss_test = 1e6  # big number
        n_stagnant = 0
        n_stagnant_max = 5
        for epoch in range(1, n_epochs + 1):
            model.train()
            output = model(train_x.float())
            loss_train = loss_fn(output, train_y, train_sigma)
            loss_train_n[epoch - 1] = loss_train.item()

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                output_test = model(test_x.float())
                loss_test[epoch - 1] = loss_fn(output_test, test_y, test_sigma).item()
                if epoch % display_step == 0 and print_progress:
                    print('Rep {}, Epoch {}, Training loss {}, Testing loss {}'.format(i, epoch,
                                                                                       round(loss_train.item(), 3),
                                                                                       round(loss_test[epoch - 1], 3)))
                    if round(loss_test[epoch - 1], 3) >= round(loss_test[epoch - 1 - display_step], 3):
                        n_stagnant += 1
                    else:
                        n_stagnant = 0
                    if n_stagnant >= n_stagnant_max:
                        if print_progress: print("detected stagnant training, breaking")
                        break
                if loss_test[epoch - 1] < min_loss_test:
                    loss_test_reps[i] = loss_test[epoch - 1]
                    min_loss_test = loss_test_reps[i]
                    min_model = copy.deepcopy(model)
                    # iets met copy.deepcopy(model)
                if epoch % saving_step == 0:
                    torch.save(min_model.state_dict(), path_to_models + "nn_rep" + str(save_idx))
                    with open(path_to_models + "costs" + ".txt", "w") as text_file:
                        text_file.write(str(min_loss_test))
        torch.save(min_model.state_dict(), path_to_models + "nn_rep" + str(save_idx))
        with open(path_to_models + "costs" + ".txt", "w") as text_file:
            text_file.write(str(min_loss_test))
        # np.savetxt(path_to_models+ "costs" + str(bs_rep_num) + ".txt", min_loss_test) # loss_test_reps[:epoch])

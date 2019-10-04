import time
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .utils import time_to_string


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Init values
        self.optimizer_alg = None
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.dropout = 0
        self.final_dropout = 0
        self.ann_rate = 0
        self.best_state = None
        self.best_opt = None
        self.train_functions = [
            {'name': 'train', 'weight': 1, 'f': None},
        ]
        self.val_functions = [
            {'name': 'val', 'weight': 1, 'f': None},
        ]
        self.acc_functions = {}
        self.acc = None

    def forward(self, *inputs):
        return None

    def mini_batch_loop(
            self, data, train=True
    ):
        losses = list()
        mid_losses = list()
        accs = list()
        n_batches = len(data)
        for batch_i, (x, y) in enumerate(data):
            # We train the model and check the loss
            if self.training:
                self.optimizer_alg.zero_grad()

            torch.cuda.synchronize()
            if isinstance(x, list):
                x_cuda = tuple(x_i.to(self.device) for x_i in x)
                pred_labels = self(*x_cuda)
            else:
                pred_labels = self(x.to(self.device))

            # Training losses
            if train:
                batch_losses = [
                    l_f['weight'] * l_f['f'](pred_labels, y)
                    for l_f in self.train_functions
                ]
                batch_loss = sum(batch_losses)
                if self.training:
                    batch_loss.backward()
                self.optimizer_alg.step()

            # Validation losses
            else:
                batch_losses = [
                    l_f['f'](pred_labels, y)
                    for l_f in self.val_functions
                ]
                batch_loss = sum([
                    l_f['weight'] * l
                    for l_f, l in zip(self.val_functions, batch_losses)
                ])
                mid_losses.append([l.tolist() for l in batch_losses])
                batch_accs = [
                    l_f['weight'] * l_f['f'](pred_labels, y)
                    for l_f in self.acc_functions
                ]
                accs.append([a.tolist() for a in batch_accs])

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            loss_value = batch_loss.tolist()
            losses.append(loss_value)

            # Curriculum dropout / Adaptive dropout
            # Here we could modify dropout to be updated for each batch.
            # (1 - rho) * exp(- gamma * t) + rho, gamma > 0

            self.print_progress(
                batch_i, n_batches, loss_value, np.mean(losses)
            )

        mean_loss = np.mean(losses)
        if train:
            return mean_loss
        else:
            mean_losses = np.mean(list(zip(*mid_losses)), axis=1)
            np_accs = np.array(list(zip(*accs)))
            mean_accs = np.mean(np.accs, axis=1) if np_accs.size > 0 else []
            return mean_loss, mean_losses, mean_accs

    def fit(
            self,
            train_loader,
            val_loader,
            epochs=50,
            patience=5,
            verbose=True
    ):
        # Init
        best_e = 0
        no_improv_e = 0
        l_names = ['train', ' val '] + [
            '{:^6s}'.format(l_f['name']) for l_f in self.val_functions
        ]
        acc_names = [
            '{:^6s}'.format(a_f['name']) for a_f in self.acc_functions
        ]
        self.best_state = deepcopy(self.state_dict())
        self.best_opt = deepcopy(self.optimizer_alg.state_dict())
        t_start = time.time()

        # Initial losses
        with torch.no_grad():
            self.t_val = time.time()
            self.eval()
            best_loss_tr = self.mini_batch_loop(train_loader)
            best_loss_val, best_losses, best_acc = self.mini_batch_loop(
                val_loader, False
            )
            if verbose:
                # Mid losses check
                epoch_s = '\033[32mInit     \033[0m'
                tr_loss_s = '\033[32m{:7.4f}\033[0m'.format(best_loss_tr)
                loss_s = '\033[32m{:7.4f}\033[0m'.format(best_loss_val)
                losses_s = [
                    '\033[36m{:8.4f}\033[0m'.format(l) for l in best_losses
                ]
                # Acc check
                acc_s = [
                    '\033[36m{:8.4f}\033[0m'.format(a) for a in best_acc
                ]
                t_out = time.time() - self.t_val
                t_s = time_to_string(t_out)

                drop_s = '{:5.3f}'.format(self.dropout)

                l_bars = '--|--'.join(
                    ['-' * 5] * 2 +
                    ['-' * 6] * (len(l_names[2:]) + len(acc_names)) +
                    ['-' * 3]
                )
                l_hdr = '  |  '.join(l_names + acc_names + ['drp'])
                print('\033[K', end='')
                whites = ' '.join([''] * 12)
                print('{:}Epoch num |  {:}  |'.format(whites, l_hdr))
                print('{:}----------|--{:}--|'.format(whites, l_bars))
                final_s = whites + ' | '.join(
                    [epoch_s, tr_loss_s, loss_s] +
                    losses_s + acc_s + [drop_s, t_s]
                )
                print(final_s)

        for self.epoch in range(epochs):
            # Main epoch loop
            self.t_train = time.time()
            self.train()
            loss_tr = self.mini_batch_loop(train_loader)
            improvement_tr = loss_tr < best_loss_tr
            if improvement_tr:
                best_loss_tr = loss_tr
                tr_loss_s = '\033[32m{:7.4f}\033[0m'.format(loss_tr)
            else:
                tr_loss_s = '{:7.4f}'.format(loss_tr)

            with torch.no_grad():
                self.t_val = time.time()
                self.eval()
                loss_val, mid_losses, acc = self.mini_batch_loop(
                    val_loader, False
                )

            # Mid losses check
            losses_s = [
                '\033[36m{:8.4f}\033[0m'.format(l) if pl > l
                else '{:8.4f}'.format(l) for pl, l in zip(
                    best_losses, mid_losses
                )
            ]
            best_losses = [
                l if pl > l else pl for pl, l in zip(
                    best_losses, mid_losses
                )
            ]
            # Acc check
            acc_s = [
                '\033[36m{:8.4f}\033[0m'.format(a) if pa < a
                else '{:8.4f}'.format(a) for pa, a in zip(
                    best_acc, acc
                )
            ]
            best_acc = [
                a if pa < a else pa for pa, a in zip(
                    best_acc, acc
                )
            ]

            # Patience check
            improvement_val = loss_val < best_loss_val
            loss_s = '{:7.4f}'.format(loss_val)
            if improvement_val:
                best_loss_val = loss_val
                epoch_s = '\033[32mEpoch {:03d}\033[0m'.format(self.epoch)
                loss_s = '\033[32m{:}\033[0m'.format(loss_s)
                best_e = self.epoch
                self.best_state = deepcopy(self.state_dict())
                self.best_opt = deepcopy(self.optimizer_alg.state_dict())
                no_improv_e = 0
            else:
                epoch_s = 'Epoch {:03d}'.format(self.epoch)
                no_improv_e += 1

            t_out = time.time() - self.t_train
            t_s = time_to_string(t_out)

            drop_s = '{:5.3f}'.format(self.dropout)
            self.dropout_update()

            if verbose:
                print('\033[K', end='')
                whites = ' '.join([''] * 12)
                final_s = whites + ' | '.join(
                    [epoch_s, tr_loss_s, loss_s] +
                    losses_s + acc_s + [drop_s, t_s]
                )
                print(final_s)

            if no_improv_e == int(patience / (1 - self.dropout)):
                break

        self.epoch = best_e
        self.load_state_dict(self.best_state)
        t_end = time.time() - t_start
        t_end_s = time_to_string(t_end)
        if verbose:
            print(
                    'Training finished in {:} epochs ({:}) '
                    'with minimum loss = {:f} (epoch {:d})'.format(
                        self.epoch + 1, t_end_s, best_loss_val, best_e
                    )
            )

    def dropout_update(self):
        if self.final_dropout <= self.dropout:
            self.dropout = max(
                self.final_dropout, self.dropout - self.ann_rate
            )

    def print_progress(self, batch_i, n_batches, b_loss, mean_loss):
        init_c = '\033[0m' if self.training else '\033[38;5;238m'
        whites = ' '.join([''] * 12)
        percent = 20 * (batch_i + 1) // n_batches
        progress_s = ''.join(['-'] * percent)
        remainder_s = ''.join([' '] * (20 - percent))
        loss_name = 'train_loss' if self.training else 'val_loss'

        if self.training:
            t_out = time.time() - self.t_train
        else:
            t_out = time.time() - self.t_val
        time_s = time_to_string(t_out)

        t_eta = (t_out / (batch_i + 1)) * (n_batches - (batch_i + 1))
        eta_s = time_to_string(t_eta)
        epoch_hdr = '{:}Epoch {:03} ({:03d}/{:03d}) [{:}>{:}] '
        loss_s = '{:} {:f} ({:f}) {:} / ETA {:}'
        batch_s = (epoch_hdr + loss_s).format(
            init_c + whites, self.epoch, batch_i + 1, n_batches,
            progress_s, remainder_s,
            loss_name, b_loss, mean_loss, time_s, eta_s + '\033[0m'
        )
        print('\033[K', end='', flush=True)
        print(batch_s, end='\r', flush=True)

    def save_model(self, net_name):
        torch.save(self.state_dict(), net_name)

    def load_model(self, net_name):
        self.load_state_dict(torch.load(net_name))


class Autoencoder(BaseModel):
    def __init__(
            self,
            conv_filters,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            n_inputs=1,
            pooling=False,
            dropout=0,
    ):
        super().__init__()
        # Init
        self.pooling = pooling
        self.device = device
        self.dropout = dropout
        # Down path
        self.down = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(
                    f_in, f_out, 3,
                    padding=1,
                ),
                nn.ReLU(),
            ) for f_in, f_out in zip(
                [n_inputs] + conv_filters[:-2], conv_filters[:-1]
            )
        ])

        self.u = nn.Sequential(
            nn.Conv3d(
                conv_filters[-2], conv_filters[-1], 3,
                padding=1
            ),
            nn.ReLU(),
        )

        # Up path
        down_out = conv_filters[-2::-1]
        up_out = conv_filters[:0:-1]
        deconv_in = map(sum, zip(down_out, up_out))
        self.up = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(
                    f_in, f_out, 3,
                    padding=1
                ),
                nn.ReLU(),
            ) for f_in, f_out in zip(
                deconv_in, down_out
            )
        ])

    def forward(self, input_s):
        down_inputs = []
        for c in self.down:
            c.to(self.device)
            input_s = F.dropout3d(
                c(input_s), self.dropout, self.training
            )
            down_inputs.append(input_s)
            if self.pooling:
                input_s = F.max_pool3d(input_s, 2)

        self.u.to(self.device)
        input_s = F.dropout3d(self.u(input_s), self.dropout, self.training)

        for d, i in zip(self.up, down_inputs[::-1]):
            d.to(self.device)
            if self.pooling:
                input_s = F.dropout3d(
                    d(
                        torch.cat(
                            (F.interpolate(input_s, size=i.size()[2:]), i),
                            dim=1
                        )
                    ),
                    self.dropout,
                    self.training
                )
            else:
                input_s = F.dropout3d(
                    d(torch.cat((input_s, i), dim=1)),
                    self.dropout,
                    self.training
                )

        return input_s


class AutoencoderDouble(Autoencoder):
    def __init__(
            self,
            conv_filters,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            n_inputs=1,
            pooling=False,
            dropout=0,
    ):
        super().__init__(conv_filters, device, n_inputs, pooling, dropout)
        # Down path
        self.down = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(
                    f_in, f_out, 3,
                    padding=1,
                ),
                nn.ReLU(),
                nn.InstanceNorm3d(f_out),
                nn.Conv3d(
                    f_out, f_out, 3,
                    padding=1,
                ),
                nn.ReLU(),
            ) for f_in, f_out in zip(
                [n_inputs] + conv_filters[:-2], conv_filters[:-1]
            )
        ])

        self.u = nn.Sequential(
            nn.Conv3d(
                conv_filters[-2], conv_filters[-1], 3,
                padding=1
            ),
            nn.ReLU(),
            nn.InstanceNorm3d(conv_filters[-1]),
            nn.Conv3d(
                conv_filters[-1], conv_filters[-1], 3,
                padding=1
            ),
            nn.ReLU(),
        )

        # Up path
        down_out = conv_filters[-2::-1]
        up_out = conv_filters[:0:-1]
        deconv_in = map(sum, zip(down_out, up_out))
        self.up = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(
                    f_in, f_out, 3,
                    padding=1
                ),
                nn.ReLU(),
                nn.InstanceNorm3d(f_out),
                nn.ConvTranspose3d(
                    f_out, f_out, 3,
                    padding=1
                ),
                nn.ReLU(),
            ) for f_in, f_out in zip(
                deconv_in, down_out
            )
        ])


class ResBlock(BaseModel):
    def __init__(self, filters_in, filters_out, kernel=3):
        super().__init__()
        self.conv = nn.Conv3d(
            filters_in, filters_out, kernel,
            padding=kernel // 2
        )
        self.res = nn.Conv3d(
            filters_in, filters_out, 1,
        )

    def forward(self, inputs):
        return self.conv(inputs) + self.res(inputs)


class ResBlockTranspose(BaseModel):
    def __init__(self, filters_in, filters_out, kernel=3):
        super().__init__()
        self.conv = nn.ConvTranspose3d(
            filters_in, filters_out, kernel,
            padding=kernel // 2
        )
        self.res = nn.ConvTranspose3d(
            filters_in, filters_out, 1,
        )

    def forward(self, inputs):
        return self.conv(inputs) + self.res(inputs)


class ResAutoencoder(Autoencoder):
    def __init__(
            self,
            conv_filters,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            n_inputs=1,
            pooling=False,
            dropout=0,
    ):
        super().__init__(conv_filters, device, n_inputs, pooling, dropout)
        # Down path
        self.down = nn.ModuleList([
            nn.Sequential(
                ResBlock(f_in, f_out, 3),
                nn.ReLU(),
            ) for f_in, f_out in zip(
                [n_inputs] + conv_filters[:-2], conv_filters[:-1]
            )
        ])

        self.u = nn.Sequential(
            ResBlock(conv_filters[-2], conv_filters[-1], 3),
            nn.ReLU(),
        )

        # Up path
        down_out = conv_filters[-2::-1]
        up_out = conv_filters[:0:-1]
        deconv_in = map(sum, zip(down_out, up_out))
        self.up = nn.ModuleList([
            nn.Sequential(
                ResBlockTranspose(f_in, f_out, 3),
                nn.ReLU(),
            ) for f_in, f_out in zip(
                deconv_in, down_out
            )
        ])

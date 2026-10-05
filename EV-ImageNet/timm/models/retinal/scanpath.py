import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchex.nn as exnn
from torch.distributions import Normal


class CircularPad(nn.Module):
    def __init__(self, pad_top):
        super(CircularPad, self).__init__()
        self.pad_top = pad_top

    def forward(self, x):
        top_pad_left = x[:, :, : self.pad_top, : x.shape[3] // 2]
        top_pad_right = x[:, :, : self.pad_top, x.shape[3] // 2:]
        top_pad = torch.cat([top_pad_right, top_pad_left], 3)
        x = torch.cat([top_pad, x], 2)
        return x

class CNN_in_polar_coords(nn.Module):
    """
    CNN module with padding along the angular axis.
    Args:
         kernel_sizes_conv2d: a list of kernel sizes for conv.
         strides_conv2d: a list of strides for conv.
         kernel_sizes_pool: a list of kernel sizes for max pooling.
         kernel_dims: a list of input and output dims for conv.
                     The first element is the input channel dim of
                     the input images. The size is
                     len(kernel_sizes_conv2d) + 1.
    Returns:
        3d tensor
    """

    def __init__(
            self,
            kernel_sizes_conv2d,
            kernel_sizes_pool,
            kernel_dims,
            strides_pool,
            pool_type="max",
    ):
        super(CNN_in_polar_coords, self).__init__()
        layers = []
        for layer in range(len(kernel_sizes_conv2d)):
            layers.append(
                exnn.PeriodicPad2d(pad_left=kernel_sizes_conv2d[layer][1] - 1)
            )
            layers.append(
                torch.nn.ReplicationPad2d(
                    (0, 0, 0, (kernel_sizes_conv2d[layer][0] - 1) // 2)
                )
            )
            layers.append(CircularPad(kernel_sizes_conv2d[layer][0] // 2))
            layers.append(
                nn.Conv2d(
                    kernel_dims[layer],
                    kernel_dims[layer + 1],
                    kernel_sizes_conv2d[layer],
                )
            )
            pad_size = kernel_sizes_pool[layer][1] - strides_pool[layer][1]
            layers.append(exnn.PeriodicPad2d(pad_left=pad_size))
            if pool_type == "max":
                pool = nn.MaxPool2d
            elif pool_type == "avg":
                pool = nn.AvgPool2d
            else:
                raise ValueError("pool_type should be either 'max' or 'avg'")
            if all(ks == 1 for ks in kernel_sizes_pool[layer]):
                pass
            else:
                layers.append(
                    nn.MaxPool2d(kernel_sizes_pool[layer], stride=strides_pool[layer])
                )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(kernel_dims[layer + 1], momentum=0.01))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x

class glimpse_network(nn.Module):
    """
    A network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.
    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.
    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.
    In other words:
        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`
    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`.
    - h_l: hidden layer size of the fc layer for `l`.
    - g: size of the square patches in the glimpses extracted
      by the retina.
    - k: number of patches to extract per glimpse.
    - s: scaling factor that controls the size of successive patches.
    - c: number of channels in each image.
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
      coordinates [x, y] for the previous timestep `t-1`.
    Returns
    -------- g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    """

    def __init__(self, h_g, h_l):
        super(glimpse_network, self).__init__()

        # glimpse layer
        self.fc1 = exnn.Linear(h_g)

        # location layer
        self.fc2 = exnn.Linear(h_l)

        self.fc3 = exnn.Linear(h_g + h_l)
        self.fc4 = exnn.Linear(h_g + h_l)

    def forward(self, x, l_t_prev):
        # generate glimpse phi from image x

        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        # feed phi and l to respective fc layers
        phi_out = F.relu(self.fc1(x))
        l_out = F.relu(self.fc2(l_t_prev))

        what = self.fc3(phi_out)
        where = self.fc4(l_out)

        # feed to fc layer
        g_t = F.relu(what + where)

        return g_t

class ScaleNetwork(nn.Module):
    """The location network.

    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        std: standard deviation of the normal distribution.
        h_t: the hidden state vector of the core network for
            the current time step `t`.

    Returns:
        mu: a 2D vector of shape (B, 2).
        l_t: a 2D vector of shape (B, 2).
    """

    def __init__(self, input_size, output_size, std):
        super().__init__()

        self.std = std

        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, hid_size)
        self.fc_st = nn.Linear(hid_size, output_size // 2)
        self.fc_rt = nn.Linear(hid_size, output_size // 2)
        self.fc_st.weight.data.zero_()
        self.fc_st.bias.data.copy_(torch.tensor([0.01], dtype=torch.float))
        self.fc_rt.weight.data.zero_()
        self.fc_rt.bias.data.copy_(torch.tensor([0.01], dtype=torch.float))

    def forward(self, h_t):
        # compute mean
        feat = F.relu(self.fc(h_t))
        smu = self.fc_st(feat)
        rmu = self.fc_rt(feat)

        # reparametrization trick
        s_t = torch.distributions.Normal(smu, self.std).rsample()
        r_t = torch.distributions.Normal(rmu, self.std).rsample()
        s_t = s_t
        r_t = r_t

        mu = torch.cat([smu, rmu], 1)
        l_t = torch.cat([s_t, r_t], 1)

        log_pi = Normal(mu, self.std).log_prob(l_t)
        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi = torch.sum(log_pi, dim=1)

        # bound between [-1, 1]

        # l_t = torch.clamp(l_t, 0, 1)

        return log_pi, l_t


class CoreNetwork(nn.Module):
    """The core network.

    An RNN that maintains an internal state by integrating
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args:
        input_size: input size of the rnn.
        hidden_size: hidden size of the rnn.
        g_t: a 2D tensor of shape (B, hidden_size). The glimpse
            representation returned by the glimpse network for the
            current timestep `t`.
        h_t_prev: a 2D tensor of shape (B, hidden_size). The
            hidden state vector for the previous timestep `t-1`.

    Returns:
        h_t: a 2D tensor of shape (B, hidden_size). The hidden
            state vector for the current timestep `t`.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t, h_t_prev):
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = F.relu(h1 + h2)
        return h_t

import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNCoreNetwork(nn.Module):
    """The core network using PyTorch's RNN with support for multiple layers.

    An RNN that maintains an internal state by integrating
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Args:
        input_size: input size of the rnn.
        hidden_size: hidden size of the rnn.
        num_layers: the number of recurrent layers (default is 1).

    Returns:
        h_t: a 2D tensor of shape (B, hidden_size). The hidden
            state vector for the current timestep `t`.
    """

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the RNN layer with multiple layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, g_t, h_t_prev):
            # Ensure the input is of shape (B, 1, input_size)
            g_t = g_t.unsqueeze(1)  # Add sequence dimension (B, 1, input_size)
            
            # Check if the previous hidden state is the correct shape
            if h_t_prev.dim() == 2:  # If the shape is (B, hidden_size)
                # Expand to (num_layers, B, hidden_size)
                h_t_prev = h_t_prev.unsqueeze(0)  # Shape becomes (1, B, hidden_size)
                h_t_prev = h_t_prev.expand(self.num_layers, -1, -1)  # Expand to (num_layers, B, hidden_size)

            # Ensure that h_t_prev is contiguous in memory
            h_t_prev = h_t_prev.contiguous()

            # Pass the input (g_t) and previous hidden state (h_t_prev) through the RNN
            out, h_t = self.rnn(g_t, h_t_prev)  # out: (B, seq_len, hidden_size), h_t: (num_layers, B, hidden_size)
            
            # We only care about the hidden state for the current time step, so we return the last hidden state (h_t)
            return h_t[-1]  # Shape: (B, hidden_size)

class LSTMCoreNetwork(nn.Module):
    """The core network using PyTorch's LSTM with support for multiple layers.

    An LSTM that maintains an internal state by integrating
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Args:
        input_size: input size of the LSTM.
        hidden_size: hidden size of the LSTM.
        num_layers: the number of recurrent layers (default is 1).

    Returns:
        h_t: a 2D tensor of shape (B, hidden_size). The hidden
            state vector for the current timestep `t`.
    """

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layer with multiple layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, g_t, h_t_prev):
        """
        Forward pass for LSTM.

        Args:
            g_t: Input for the current time step (B, 1, input_size).
            h_t_prev: A tuple of (h_0, c_0) for the previous hidden and cell state.
                Both h_0 and c_0 should have the shape (num_layers, B, hidden_size).

        Returns:
            h_t: The hidden state for the current timestep (B, hidden_size).
        """
        # Ensure the input is of shape (B, 1, input_size)
        g_t = g_t.unsqueeze(1)  # Add sequence dimension (B, 1, input_size)
        
        # Check if the previous hidden state is in the correct format (tuple)
        if isinstance(h_t_prev, tuple):
            h_0, c_0 = h_t_prev
            if h_0.dim() == 2:  # If h_0 and c_0 are 2D (B, hidden_size), expand them
                h_0 = h_0.unsqueeze(0).expand(self.num_layers, -1, -1)  # (num_layers, B, hidden_size)
                c_0 = c_0.unsqueeze(0).expand(self.num_layers, -1, -1)  # (num_layers, B, hidden_size)
            h_t_prev = (h_0.contiguous(), c_0.contiguous())
        else:
            # If h_t_prev is not a tuple, we need to handle it differently
            raise ValueError("h_t_prev should be a tuple of (h_0, c_0)")

        # Pass the input (g_t) and previous hidden state (h_t_prev) through the LSTM
        out, (h_t, c_t) = self.lstm(g_t, h_t_prev)  # out: (B, seq_len, hidden_size), h_t: (num_layers, B, hidden_size)

        # We only care about the hidden state for the current time step, so we return the last hidden state (h_t)
        return out, h_t[-1], c_t[-1]  # Shape: (B, hidden_size)
    
class ActionNetwork(nn.Module):
    """The action network.

    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.

    Returns:
        a_t: output probability vector over the classes.
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t


# class LocationNetwork(nn.Module):
#     """The location network.

#     Uses the internal state `h_t` of the core network to
#     produce the location coordinates `l_t` for the next
#     time step.

#     Concretely, feeds the hidden state `h_t` through a fc
#     layer followed by a tanh to clamp the output beween
#     [-1, 1]. This produces a 2D vector of means used to
#     parametrize a two-component Gaussian with a fixed
#     variance from which the location coordinates `l_t`
#     for the next time step are sampled.

#     Hence, the location `l_t` is chosen stochastically
#     from a distribution conditioned on an affine
#     transformation of the hidden state vector `h_t`.

#     Args:
#         input_size: input size of the fc layer.
#         output_size: output size of the fc layer.
#         std: standard deviation of the normal distribution.
#         h_t: the hidden state vector of the core network for
#             the current time step `t`.

#     Returns:
#         mu: a 2D vector of shape (B, 2).
#         l_t: a 2D vector of shape (B, 2).
#     """

#     def __init__(self, input_size, output_size, std):
#         super().__init__()

#         self.std = std

#         hid_size = input_size // 2
#         self.fc = nn.Linear(input_size, hid_size)
#         self.fc_lt = nn.Linear(hid_size, output_size)
#         self.inhibition = nn.Linear(hid_size,output_size)
        
#         nn.init.constant_(self.inhibition.weight, 0.0)  # 将权重初始化为 0
#         nn.init.constant_(self.inhibition.bias, 0.0)    # 将偏置初始化为 0
#     def forward(self, h_t):
#         # compute mean
#         feat = F.relu(self.fc(h_t))
#         mu = torch.tanh(self.fc_lt(feat))
        
#         # alpha = torch.sigmoid(self.inhibition(feat))  # 预测抑制因子
#         # inhibition = 1 - alpha * torch.exp(((torch.abs(mu) - 1) ** 2))
#         # inhibition = torch.clamp(inhibition, 0.5, 1)
#         # mu = mu * inhibition  # 使用抑制因子调整 l_t
        
#         # reparametrization trick
#         l_t = torch.distributions.Normal(mu, self.std).rsample()
#         l_t = l_t

#         log_pi = Normal(mu, self.std).log_prob(l_t)

#         # we assume both dimensions are independent
#         # 1. pdf of the joint is the product of the pdfs
#         # 2. log of the product is the sum of the logs
#         log_pi = torch.sum(log_pi, dim=1)

#         # bound between [-1, 1]
#         l_t = torch.clamp(l_t, -1, 1)

#         alpha = torch.sigmoid(self.inhibition(feat))  # 预测抑制因子，更偏好中心
#         inhibition = torch.exp(-alpha*((torch.abs(l_t) - 0) ** 2))
#         inhibition = torch.clamp(inhibition, 0.5, 1)
#         l_t = l_t * inhibition  # 使用抑制因子调整 l_t
#         return log_pi, l_t
class LocationNetwork(nn.Module):
    """The location network.

    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        std: standard deviation of the normal distribution.
        h_t: the hidden state vector of the core network for
            the current time step `t`.

    Returns:
        mu: a 2D vector of shape (B, 2).
        l_t: a 2D vector of shape (B, 2).
    """

    def __init__(self, input_size, output_size, std):
        super().__init__()

        self.std = std

        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, hid_size)
        self.fc_lt = nn.Linear(hid_size, output_size)
        
        nn.init.constant_(self.fc_lt.weight, 0.0)
        nn.init.constant_(self.fc_lt.bias, 0.0)
    def forward(self, h_t):
        # compute mean
        feat = F.relu(self.fc(h_t.detach()))
        mu_arw = self.fc_lt(feat)
        mu = torch.tanh(mu_arw)

        l_t = torch.distributions.Normal(mu, self.std).sample()
        log_pi = Normal(mu, self.std).log_prob(l_t)

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi = torch.sum(log_pi, dim=1)

        l_t = torch.clamp(l_t, min=-0.7, max=0.7)

        return log_pi, l_t


class SRNetwork(nn.Module):

    def __init__(self, input_size, output_size_s, wscale, std):
        super().__init__()

        self.std = std
        self.wscale = wscale
        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, hid_size)
        self.fc_s = nn.Linear(hid_size, output_size_s) # 希望0，1
        
        nn.init.constant_(self.fc_s.weight, 0)  # 将权重初始化为 0
        nn.init.constant_(self.fc_s.bias, 1.0/wscale)    # 将偏置初始化为 0
    def forward(self, h_t):
        # compute mean
        feat = F.relu(self.fc(h_t.detach()))
        mu = torch.sigmoid(self.fc_s(feat)) # 希望0，1
        l_sr = torch.distributions.Normal(mu, self.std).sample()
        
        
        log_pi = Normal(mu, self.std).log_prob(l_sr)
        log_pi = torch.sum(log_pi, dim=1)

        # bound between [-1, 1]
        l_sr = torch.clamp(l_sr, 1/self.wscale/self.wscale, 1)
        return log_pi, l_sr
    
class ScaleNetwork2(nn.Module):
    """The location network.

    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        std: standard deviation of the normal distribution.
        h_t: the hidden state vector of the core network for
            the current time step `t`.

    Returns:
        mu: a 2D vector of shape (B, 2).
        l_t: a 2D vector of shape (B, 2).
    """

    def __init__(self, input_size, output_size, std):
        super().__init__()

        self.std = std

        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, hid_size)
        self.fc_lt = nn.Linear(hid_size, output_size)

    def forward(self, h_t):
        # compute mean
        feat = F.relu(self.fc(h_t))
        mu = torch.sigmoid(self.fc_lt(feat))

        # reparametrization trick
        l_t = torch.distributions.Normal(mu, self.std).rsample()
        l_t = l_t
        log_pi = Normal(mu, self.std).log_prob(l_t)

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi = torch.sum(log_pi, dim=1)

        # bound between [-1, 1]
        l_t = torch.clamp(l_t, 0, 1)

        return log_pi, l_t


class BaselineNetwork(nn.Module):
    """The baseline network.

    This network regresses the baseline in the
    reward function to reduce the variance of
    the gradient update.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.

    Returns:
        b_t: a 2D vector of shape (B, 1). The baseline
            for the current time step `t`.
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = self.fc(h_t)
        return b_t


class core_network(nn.Module):
    """
    An RNN that maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.
    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.
    In other words:
        `h_t = relu( fc(h_t_prev) + fc(g_t) )`
    Args
    ----
    - input_size: input size of the rnn.
    - hidden_size: hidden size of the rnn.
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    - h_t_prev: a 2D tensor of shape (B, hidden_size). The
      hidden state vector for the previous timestep `t-1`.
    Returns
    -------
    - h_t: a 2D tensor of shape (B, hidden_size). The hidden
      state vector for the current timestep `t`.
    """

    def __init__(self, hidden_size):
        super(core_network, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = exnn.Linear(hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t, h_t_prev):
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = F.relu(h1 + h2)
        return h_t


class action_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the final output classification.
    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.
    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.
    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.
    Returns
    -------
    - a_t: output probability vector over the classes.
    """

    def __init__(self, input_size, output_size):
        super(action_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = self.fc(h_t)
        return a_t


class location_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.
    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.
    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.
    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.
    Returns
    -------
    - mu: a 2D vector of shape (B, 2).
    - l_t: a 2D vector of shape (B, 2).
    """

    def __init__(self, input_size, output_size, std):
        super(location_network, self).__init__()
        self.std = std
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        # compute mean
        mu = torch.clamp(self.fc(h_t), min=-1.0, max=1.0)

        # reparametrization trick
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std)
        l_t = mu + noise

        # bound between [-1, 1]
        # l_t = torch.tanh(l_t)
        l_t = l_t

        return mu, l_t


class baseline_network(nn.Module):
    """
    Regresses the baseline in the reward function
    to reduce the variance of the gradient update.
    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network
      for the current time step `t`.
    Returns
    -------
    - b_t: a 2D vector of shape (B, 1). The baseline
      for the current time step `t`.
    """

    def __init__(self, input_size, output_size):
        super(baseline_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = F.relu(self.fc(h_t))
        return b_t

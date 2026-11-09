"""Inference architecture extracted from LeapFactual; see provenance JSON."""
import torch
from torch import nn, Tensor
from typing import List

def get_norm_layer(channels, norm_type='bn'):
    if norm_type == 'bn':
        return nn.BatchNorm2d(channels, eps=0.0001)
    elif norm_type == 'gn':
        return nn.GroupNorm(8, channels, eps=0.0001)
    else:
        ValueError('norm_type must be bn or gn')

class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type='bn'):
        super(ResDown, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)
        self.conv1 = nn.Conv2d(channel_in, max(channel_out // 2, 1) + channel_out, kernel_size, 2, kernel_size // 2)
        self.norm2 = get_norm_layer(max(channel_out // 2, 1), norm_type=norm_type)
        self.conv2 = nn.Conv2d(max(channel_out // 2, 1), channel_out, kernel_size, 1, kernel_size // 2)
        self.act_fnc = nn.LeakyReLU()
        self.channel_out = channel_out

    def forward(self, x):
        x = self.act_fnc(self.norm1(x))
        x_cat = self.conv1(x)
        skip = x_cat[:, :self.channel_out]
        x = x_cat[:, self.channel_out:]
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + skip

class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, scale_factor=2, norm_type='bn'):
        super(ResUp, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)
        self.conv1 = nn.Conv2d(channel_in, max(channel_in // 2, 1) + channel_out, kernel_size, 1, kernel_size // 2)
        self.norm2 = get_norm_layer(max(channel_in // 2, 1), norm_type=norm_type)
        self.conv2 = nn.Conv2d(max(channel_in // 2, 1), channel_out, kernel_size, 1, kernel_size // 2)
        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.act_fnc = nn.LeakyReLU()
        self.channel_out = channel_out

    def forward(self, x_in):
        x = self.up_nn(self.act_fnc(self.norm1(x_in)))
        x_cat = self.conv1(x)
        skip = x_cat[:, :self.channel_out]
        x = x_cat[:, self.channel_out:]
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + skip

class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type='bn'):
        super(ResBlock, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)
        first_out = max(channel_in // 2, 1) if channel_in == channel_out else max(channel_in // 2, 1) + channel_out
        self.conv1 = nn.Conv2d(channel_in, first_out, kernel_size, 1, kernel_size // 2)
        self.norm2 = get_norm_layer(max(channel_in // 2, 1), norm_type=norm_type)
        self.conv2 = nn.Conv2d(max(channel_in // 2, 1), channel_out, kernel_size, 1, kernel_size // 2)
        self.act_fnc = nn.LeakyReLU()
        self.skip = channel_in == channel_out
        self.bttl_nk = max(channel_in // 2, 1)

    def forward(self, x_in):
        x = self.act_fnc(self.norm1(x_in))
        x_cat = self.conv1(x)
        x = x_cat[:, :self.bttl_nk]
        if self.skip:
            skip = x_in
        else:
            skip = x_cat[:, self.bttl_nk:]
        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        return x + skip

class WAE_MMD(nn.Module):

    def __init__(self, in_channels: int, total_latent_dim: int, num_classes: int, img_size: int=64, hidden_dims: List=[16, 32, 64, 32], reg_weight: int=100, kernel_type: str='imq', latent_var: float=2.0, **kwargs) -> None:
        super(WAE_MMD, self).__init__()
        self.num_classes = num_classes
        self.total_latent_dim = total_latent_dim
        self.img_size = img_size
        self.num_layers = len(hidden_dims)
        self.embedding_shape = [hidden_dims[-1], img_size // 2 ** self.num_layers, img_size // 2 ** self.num_layers]
        self.embedding_length = torch.prod(torch.tensor(self.embedding_shape))
        self.num_input_dim = in_channels * img_size ** 2
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var
        local_in_channels = in_channels
        modules = []
        for h_dim in hidden_dims:
            modules.append(ResBlock(local_in_channels, local_in_channels))
            modules.append(ResDown(local_in_channels, h_dim))
            local_in_channels = h_dim
        modules.append(ResBlock(local_in_channels, local_in_channels))
        self.encoder = nn.Sequential(*modules)
        self.fc_in = nn.Sequential(nn.Linear(self.embedding_length, 512), nn.Linear(512, total_latent_dim))
        self.fc_out = nn.Sequential(nn.Linear(total_latent_dim, 512), nn.Linear(512, self.embedding_length))
        hidden_dims.reverse()
        if self.num_classes > 0:
            self.fc_cond = nn.Sequential(nn.Linear(self.num_classes, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, self.embedding_length))
            self.cond_embed = nn.ModuleList()
            for i in range(len(hidden_dims)):
                self.cond_embed.append(nn.Sequential(ResBlock(hidden_dims[i], hidden_dims[i]), ResUp(hidden_dims[i], channel_out=hidden_dims[i + 1] if i + 1 < len(hidden_dims) else 8)))
            self.z_discriminator = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(self.total_latent_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, num_classes))
        self.decoder = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.decoder.append(nn.Sequential(ResBlock(hidden_dims[i] * (2 if num_classes > 0 else 1), hidden_dims[i]), ResUp(hidden_dims[i], channel_out=hidden_dims[i + 1] if i + 1 < len(hidden_dims) else 8)))
        hidden_dims.reverse()
        self.output = nn.Sequential(ResBlock(8, 8), nn.Conv2d(in_channels=8, out_channels=in_channels, kernel_size=5, stride=1, padding=2), nn.Sigmoid())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = torch.flatten(self.encoder(input), start_dim=1)
        return self.fc_in(result)

    def decode(self, z: Tensor, cond: Tensor=None) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        if cond is not None:
            cond_middle = self.fc_cond(cond).view(-1, *self.embedding_shape)
            decoder_middle = self.fc_out(z).view(-1, *self.embedding_shape)
            for l_cond, l_dec in zip(self.cond_embed, self.decoder):
                decoder_input = torch.cat([decoder_middle, cond_middle], dim=1)
                decoder_middle = l_dec(decoder_input)
                cond_middle = l_cond(cond_middle)
            return self.output(decoder_middle)
        else:
            result = self.fc_out(z).view(-1, *self.embedding_shape)
            for layer in self.decoder:
                result = layer(result)
            return self.output(result)

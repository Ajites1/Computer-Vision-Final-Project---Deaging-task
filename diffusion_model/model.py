import torch
import torch.nn as nn



def reverse_diffusion_step(x, predicted_noise, beta_t):
    """
    Performs a single reverse diffusion step to denoise the input.

    Args:
        x (torch.Tensor): Noisy tensor at step `t`.
        predicted_noise (torch.Tensor): Noise predicted by the neural network.
        beta_t (float): Noise variance at step `t`.
    
    Returns:
        torch.Tensor: Denoised tensor for step `t-1`.
    """
    return (x - torch.sqrt(beta_t) * predicted_noise) / torch.sqrt(1 - beta_t)

class UNet(nn.Module):
    def __init__(self, input_channels, num_steps):
        super(UNet, self).__init__()
        
        # Define the encoder layers
        self.encoder1 = self.conv_block(input_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        
        # Define the decoder layers
        self.decoder1 = self.conv_block(256, 128)
        self.decoder2 = self.conv_block(128, 64)
        self.decoder3 = self.conv_block(64, input_channels)
        
        # Timestep embedding layer
        self.time_embed = nn.Linear(num_steps, 256)

    def conv_block(self, in_channels, out_channels):
        # Convolution block with ReLU and batch normalization
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, t):
        """
        Forward pass of U-Net with timestep embedding.
        
        Args:
            x (torch.Tensor): Noisy input data.
            t (torch.Tensor): Timestep encoding.
        
        Returns:
            torch.Tensor: Predicted noise at this timestep.
        """
        # Apply timestep embedding
        t_embed = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)  # Reshape for broadcasting
        
        # Encoder pass
        e1 = self.encoder1(x + t_embed)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        # Decoder pass with skip connections
        d1 = self.decoder1(e3 + e2)
        d2 = self.decoder2(d1 + e1)
        d3 = self.decoder3(d2)
        
        return d3

from timm.models import VisionTransformer
import torch.nn as nn 
import torch
import math



# Helper function to calculate FFT FLOPS
# FFT of size N requires approximately 5*N*log2(N) operations
def compute_fft_flops(batch_size, signal_length):
    """Compute FLOPS for FFT operations"""
    return batch_size * signal_length * math.log2(signal_length) * 5

# Helper function to calculate circular convolution FLOPS
def compute_circular_convolution_flops(batch_size, num_signals, signal_length):
    """
    Compute FLOPS for FFT-based circular convolution
    - FFT of signal: 5*N*log2(N)
    - FFT of kernel: 5*N*log2(N)
    - Element-wise multiplication: N
    - IFFT: 5*N*log2(N)
    Total: ~15*N*log2(N) + N per operation
    """
    n = signal_length
    fft_forward = compute_fft_flops(batch_size * num_signals, n)
    fft_inverse = compute_fft_flops(batch_size * num_signals, n)
    element_wise_mult = batch_size * num_signals * n
    total_flops = fft_forward + element_wise_mult + fft_inverse
    return total_flops

# Circular convolution 
def batch_circular_convolution_fft(x, h):
    # x: [B//R, R, D]
    # h: [R, D] → need to expand to [1, R, D] for broadcasting
    h = h.unsqueeze(0)  # shape becomes [1, R, D]

    # Apply FFT along the last dimension
    x_fft = torch.fft.fft(x, dim=-1)
    h_fft = torch.fft.fft(h, dim=-1)

    # Element-wise complex multiplication in frequency domain
    result_fft = x_fft * h_fft

    # Inverse FFT and keep real part
    return torch.fft.ifft(result_fft, dim=-1).real

def batch_circular_correlation_fft(x, h):
    """
    x: Tensor of shape [B, 1, D]
    h: Tensor of shape [R, D]
    Returns: Tensor of shape [B, R, D] with circular correlation results
    """
    # Ensure shapes are compatible
    x = x  # [B, 1, D]
    h = h.unsqueeze(0)  # [1, R, D]

    # Compute FFTs
    X = torch.fft.fft(x, dim=-1)        # [B, 1, D]
    H = torch.fft.fft(h, dim=-1).conj() # [1, R, D]

    # Broadcast multiply: result is [B, R, D]
    Y = X * H

    # Inverse FFT and take real part
    return torch.fft.ifft(Y, dim=-1).real

# Main Encoder 
class Encoder(nn.Module):
    def __init__(self,
                 R,
                 keys,
                 *args, **kwargs):
        
        self.R = R
        self.keys = keys
        self.conv_flops = 0  # Track FLOPS from circular convolutions

        super().__init__(*args, **kwargs)

    def forward(self, x, *args, **kwargs):

        if self.training:

            # Flatten into B x d
            x  = torch.flatten(x, start_dim=1)    
            
            # Store dimensions 
            batch_dim, features_dim = x.shape

            # Reshape in B/R x R x d 
            x = x.reshape(batch_dim // self.R, self.R, features_dim)

            # Calculate FLOPS for circular convolution
            # Shape: [batch_dim // self.R, R, features_dim]
            batch_over_r = batch_dim // self.R
            conv_flops = compute_circular_convolution_flops(
                batch_size=batch_over_r,
                num_signals=self.R,
                signal_length=features_dim
            )
            self.conv_flops += conv_flops

            # Do batch circular convolution 
            x = batch_circular_convolution_fft(x, self.keys)

            # Sum over the R elements
            x = x.sum(dim=1)  # shape: [B//R, D]

        return x

# Main Decoder 
class Decoder(nn.Module):

    def __init__(self,
                 keys,
                 shape,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Store keys 
        self.keys = keys
        self.shape = shape 
        self.corr_flops = 0  # Track FLOPS from circular correlations

    def forward(self, x, *args, **kwargs):

        if self.training:

            # from B/R x d  -> B/R x 1 x d 
            x = x.unsqueeze(1)

            # Calculate FLOPS for circular correlation
            # Shape before reshape: [B/R, 1, D]
            batch_over_r = x.shape[0]
            num_signals = self.keys.shape[0]  # R
            signal_length = x.shape[-1]       # D
            
            corr_flops = compute_circular_convolution_flops(
                batch_size=batch_over_r,
                num_signals=num_signals,
                signal_length=signal_length
            )
            self.corr_flops += corr_flops

            # Decode into a B/R x R x D tensor 
            x = batch_circular_correlation_fft(x, self.keys)

            # Get dimensions 
            batch_over_R, R, features_dim = x.shape
            batch_size = batch_over_R *  R

            # Reshape into B x D 
            x = x.reshape(batch_size, features_dim)

            # Reshape into B X T X F
            x = x.reshape(self.shape)

        return x



class model(nn.Module):

    def __init__(self, 
                 model: VisionTransformer,
                 channel,
                 split_index,
                 R,
                 batch_size,
                 *args, **kwargs): 
        
        super().__init__(*args, **kwargs)

        # Get dimensions 
        self.batch_size = batch_size
        self.n_tokens, self.token_dim = self.get_dimensions(model)

        # Instantiate keys
        self.keys = self.instantiate_keys(R, self.n_tokens * self.token_dim)
        
        # Build model 
        self.model = self.build_model(model, channel, split_index, R)

        # Store compression 
        self.compression = 1 / R 

        # Store channel 
        self.channel = channel

        # Variable to store communication 
        self.communication = 0 

        # Variable to store circular convolution FLOPS
        self.circular_conv_flops = 0

        # Store name 
        self.name = "C3-SL"



    # Get the dimensions of activations of the model 
    def get_dimensions(self, model):

        img_size = model.default_cfg['input_size'][-1]
        patch_size = model.patch_embed.patch_size[0]

        n_tokens = (img_size // patch_size) ** 2 + 1 
        token_dim = model.embed_dim

        return n_tokens, token_dim




    def instantiate_keys(self, R, flat_activation_size):

        # Get device 
        device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create keys 
        keys = torch.normal(0, 1/flat_activation_size, size=(R, flat_activation_size)).to(device) 
        keys = keys / keys.norm(dim=1, keepdim=True)

        return keys 
    
    # Function to build model 
    def build_model(self, model, channel, split_index,R):

        # Get encoder and decoder
        encoder = Encoder(R, self.keys)
        decoder = Decoder(self.keys, shape = (self.batch_size, self.n_tokens, self.token_dim))

        # Split the original model 
        blocks_before = model.blocks[:split_index]
        blocks_after = model.blocks[split_index:]

        # Add comm pipeline and compression modules 
        model.blocks = nn.Sequential(*blocks_before, encoder, channel, decoder, *blocks_after)
        # model.blocks = nn.Sequential(*blocks_before, encoder)

        return model 

    # Forward 
    def forward(self, x):
        batch_size = x.shape[0]
        if self.training: 
            self.communication += self.compression * batch_size
        
        return self.model.forward(x)
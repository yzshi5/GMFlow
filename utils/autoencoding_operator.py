"""
Autoencoder model for 3D data compression and reconstruction.
This module contains the UNO_11 autoencoder architecture and training utilities.

Minimize over a batch size of 512.
"""

# Standard library imports
import os
import time
from pathlib import Path
from datetime import datetime

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def kernel_loc(in_chan=3, up_dim=32):
    """
    Kernel network applied on grid.
    
    Args:
        in_chan (int): Number of input channels
        up_dim (int): Hidden dimension size
        
    Returns:
        nn.Sequential: Kernel network layers
    """
    layers = nn.Sequential(
        nn.Linear(in_chan, up_dim, bias=True), 
        torch.nn.GELU(),
        nn.Linear(up_dim, 2*up_dim, bias=True), 
        torch.nn.GELU(),
        nn.Linear(2*up_dim, 1, bias=False)
    )
    return layers


# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

class SpectralConv3d_Uno(nn.Module):
    """
    3D Fourier layer that performs FFT, linear transform, and Inverse FFT.
    
    Args:
        in_codim (int): Input co-domain dimension
        out_codim (int): Output co-domain dimension
        dim1 (int): Default output grid size along x (1st dimension)
        dim2 (int): Default output grid size along y (2nd dimension)
        dim3 (int): Default output grid size along time t (3rd dimension)
        modes1 (int, optional): Number of Fourier modes for 1st dimension
        modes2 (int, optional): Number of Fourier modes for 2nd dimension
        modes3 (int, optional): Number of Fourier modes for 3rd dimension
        
    Note:
        The ratio of input and output grid sizes implicitly sets the expansion
        or contraction factor along each dimension. Number of modes must be
        compatible with input and output grid sizes:
        - modes1 <= min(dim1/2, input_dim1/2)
        - modes2 <= min(dim2/2, input_dim2/2)
        - modes3 <= min(dim3/2, input_dim3/2)
    """
    
    def __init__(self, in_codim, out_codim, dim1, dim2, dim3, modes1=None, modes2=None, modes3=None):
        super(SpectralConv3d_Uno, self).__init__()
        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        if modes1 is not None:
            self.modes1 = modes1 
            self.modes2 = modes2
            self.modes3 = modes3 
        else:
            self.modes1 = dim1 
            self.modes2 = dim2
            self.modes3 = dim3//2+1

        self.scale = (1 / (2*in_codim))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        """
        Complex multiplication for 3D tensors.
        
        Args:
            input: Input tensor
            weights: Weight tensor
            
        Returns:
            torch.Tensor: Result of complex multiplication
        """
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x, dim1=None, dim2=None, dim3=None):
        """
        Forward pass of the spectral convolution layer.
        
        Args:
            x: Input tensor
            dim1 (int, optional): Output grid size along x
            dim2 (int, optional): Output grid size along y
            dim3 (int, optional): Output grid size along t
            
        Returns:
            torch.Tensor: Output tensor
            
        Shape:
            Input: (batch, in_codim, input_dim1, input_dim2, input_dim3)
            Output: (batch, out_codim, dim1, dim2, dim3)
        """
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
            self.dim3 = dim3   

        batchsize = x.shape[0]

        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1], norm = 'forward')

        out_ft = torch.zeros(batchsize, self.out_channels, self.dim1, self.dim2, self.dim3//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(self.dim1, self.dim2, self.dim3), norm='forward')
        return x

class pointwise_op_3D(nn.Module):
    """
    3D pointwise convolution operation with Fourier domain processing.
    
    Args:
        in_codim (int): Input co-domain dimension
        out_codim (int): Output co-domain dimension
        dim1 (int): Output dimension along x
        dim2 (int): Output dimension along y
        dim3 (int): Output dimension along t
    """
    
    def __init__(self, in_codim, out_codim, dim1, dim2, dim3):
        super(pointwise_op_3D, self).__init__()
        self.conv = nn.Conv3d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.dim3 = int(dim3)

    def forward(self, x, dim1=None, dim2=None, dim3=None):
        """
        Forward pass of pointwise 3D convolution.
        
        Args:
            x: Input tensor
            dim1 (int, optional): Output dimension along x
            dim2 (int, optional): Output dimension along y
            dim3 (int, optional): Output dimension along t
            
        Returns:
            torch.Tensor: Output tensor after convolution and interpolation
        """
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
            dim3 = self.dim3
        x_out = self.conv(x)

        # Apply Fourier transform
        ft = torch.fft.rfftn(x_out, dim=[-3, -2, -1])
        ft_u = torch.zeros_like(ft)
        
        # Apply frequency domain filtering
        ft_u[:, :, :(dim1//2), :(dim2//2), :(dim3//2)] = ft[:, :, :(dim1//2), :(dim2//2), :(dim3//2)]
        ft_u[:, :, -(dim1//2):, :(dim2//2), :(dim3//2)] = ft[:, :, -(dim1//2):, :(dim2//2), :(dim3//2)]
        ft_u[:, :, :(dim1//2), -(dim2//2):, :(dim3//2)] = ft[:, :, :(dim1//2), -(dim2//2):, :(dim3//2)]
        ft_u[:, :, -(dim1//2):, -(dim2//2):, :(dim3//2)] = ft[:, :, -(dim1//2):, -(dim2//2):, :(dim3//2)]
        
        # Inverse Fourier transform
        x_out = torch.fft.irfftn(ft_u, s=(dim1, dim2, dim3))

        # Final interpolation
        x_out = torch.nn.functional.interpolate(
            x_out, 
            size=(dim1, dim2, dim3), 
            mode='trilinear', 
            align_corners=True
        )
        return x_out

class OperatorBlock_3D(nn.Module):
    """
    3D operator block combining spectral convolution and pointwise operations.
    
    Args:
        in_codim (int): Input co-domain dimension
        out_codim (int): Output co-domain dimension
        dim1 (int): Output dimension along x
        dim2 (int): Output dimension along y
        dim3 (int): Output dimension along t
        modes1 (int): Number of Fourier modes for 1st dimension
        modes2 (int): Number of Fourier modes for 2nd dimension
        modes3 (int): Number of Fourier modes for 3rd dimension
        Normalize (bool): If True, performs InstanceNorm3d on the output
        Non_Lin (bool): If True, applies pointwise nonlinearity (GELU)
    """
    
    def __init__(self, in_codim, out_codim, dim1, dim2, dim3, modes1, modes2, modes3, 
                 Normalize=False, Non_Lin=True):
        super(OperatorBlock_3D, self).__init__()
        self.conv = SpectralConv3d_Uno(in_codim, out_codim, dim1, dim2, dim3, modes1, modes2, modes3)
        self.w = pointwise_op_3D(in_codim, out_codim, dim1, dim2, dim3)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm3d(int(out_codim), affine=True)


    def forward(self, x, dim1=None, dim2=None, dim3=None):
        """
        Forward pass of the 3D operator block.
        
        Args:
            x: Input tensor
            dim1 (int, optional): Output dimension along x
            dim2 (int, optional): Output dimension along y
            dim3 (int, optional): Output dimension along t
            
        Returns:
            torch.Tensor: Output tensor
            
        Shape:
            Input: (batch, in_codim, input_dim1, input_dim2, input_dim3)
            Output: (batch, out_codim, dim1, dim2, dim3)
        """
        x1_out = self.conv(x, dim1, dim2, dim3)
        x2_out = self.w(x, dim1, dim2, dim3)
        x_out = x1_out + x2_out
        
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
            
        return x_out
    
    
# =============================================================================
# MAIN AUTOENCODER MODEL
# =============================================================================

class AutoEncoderOperator(nn.Module):
    """
    UNO_11 Autoencoder model for 3D data compression and reconstruction.
    
    This model implements an encoder-decoder architecture with residual connections
    and spectral convolutions for efficient 3D data processing.
    
    Args:
        in_width (int): Input channel width
        width (int): Hidden layer width
        pad (int): Padding for non-periodic domains
        factor (float): Scaling factor for dimensions
    """
    
    def __init__(self, in_width, width, pad=0, factor=1):
        super(AutoEncoderOperator, self).__init__()

        self.in_width = in_width  # input channel
        self.width = width
        self.padding = pad  # pad the domain if input is non-periodic
        self.factor = factor

        # Initial projection layers
        self.fc_n1 = nn.Linear(self.in_width, self.width//2)
        self.fc0 = nn.Linear(self.width//2, self.width)
        
        # Input shape: [3, 128, 64, 48]
        # Encoder layers (downsampling)
        self.conv_init = OperatorBlock_3D(self.width, self.width, 96, 48, 48, 48, 24, 16)
        self.conv_init_res = OperatorBlock_3D(2*self.width, self.width, 96, 48, 48, 48, 24, 16, Normalize=True)
        
        self.conv0 = OperatorBlock_3D(self.width, self.width, 64, 32, 32, 32, 16, 16)
        self.conv0_res = OperatorBlock_3D(2*self.width, self.width, 64, 32, 32, 32, 16, 16, Normalize=True)
          
        self.conv1 = OperatorBlock_3D(self.width, self.width, 32, 16, 16, 16, 8, 8)
        self.conv1_res = OperatorBlock_3D(2*self.width, 1, 32, 16, 16, 16, 8, 8, Normalize=True)

        # self.to_bottleneck = nn.Conv3d(self.width, 1, 1) # width -> 1
        
        # Bottleneck (must be 1 channel for F2ID metric)
        self.conv2 = OperatorBlock_3D(1, self.width, 64, 32, 32, 16, 8, 8)
        self.conv2_res = OperatorBlock_3D(self.width+1, self.width, 64, 32, 32, 16, 8, 8, Normalize=True)
        
        # Decoder layers (upsampling)
        self.conv3 = OperatorBlock_3D(self.width, self.width, 96, 48, 32, 32, 16, 16)
        self.conv3_res = OperatorBlock_3D(2*self.width, self.width, 96, 48, 32, 32, 16, 16, Normalize=True)

        self.conv_end = OperatorBlock_3D(self.width, self.width, 128, 64, 48, 48, 24, 16)
        self.conv_end_res = OperatorBlock_3D(2*self.width, self.width, 128, 64, 48, 48, 24, 16)
        
        # Final projection layers
        self.fc1 = nn.Linear(1*self.width, 1*self.width)
        self.fc2 = nn.Linear(1*self.width, in_width-3) # -3 for the grid coordinates

    def forward(self, x, output_encode=False, decode=False, pad=[0, 0, 16], factor=3/4):
        """
        Forward pass of the UNO_11 autoencoder.
        
        Args:
            x: Input tensor
            output_encode (bool): If True, return encoded representation
            decode (bool): If True, perform decoding from encoded representation
            pad (list): Padding values for each dimension
            factor (float): Scaling factor for dimensions
            
        Returns:
            torch.Tensor: Reconstructed output or encoded representation
        """
        
        # Encoding path
        if decode == False:
            # Add positional encoding
            x = x.permute(0, 2, 3, 4, 1)  # shape [batch, ndim1, ndim2, ndim, n_channel]
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1)

            x_fc_1 = self.fc_n1(x)
            x_fc_1 = F.gelu(x_fc_1)

            x_fc0 = self.fc0(x_fc_1)
            x_fc0 = F.gelu(x_fc0)

            x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
    
            # Calculate dimensions with optional padding
            if pad is None:
                D1_raw, D2_raw, D3_raw = x_fc0.shape[-3], x_fc0.shape[-2], x_fc0.shape[-1]
                D1, D2, D3 = x_fc0.shape[-3], x_fc0.shape[-2], x_fc0.shape[-1]
            else:
                D1_raw, D2_raw, D3_raw = x_fc0.shape[-3], x_fc0.shape[-2], x_fc0.shape[-1]  # (128, 64, 48)
                D1, D2, D3 = D1_raw+pad[0], D2_raw+pad[1], D3_raw+pad[2]  # (128, 64, 64)
                
            ## Residual connection (encoder part)
            x_init = self.conv_init(x_fc0,int(factor*D1),int(factor*D2), int(factor*D3))
            x_init = torch.cat([F.interpolate(x_init, size=(x_fc0.shape[2], x_fc0.shape[3], x_fc0.shape[4]), mode='trilinear',
                                            align_corners=True), x_fc0], dim=1)
            x_init = self.conv_init_res(x_init,int(factor*D1),int(factor*D2), int(factor*D3))
            
            x_c0 = self.conv0(x_init,D1//2,D2//2, D3//2)
            x_c0 = torch.cat([F.interpolate(x_c0, size=(x_init.shape[2], x_init.shape[3], x_init.shape[4]), mode='trilinear',
                                            align_corners=True), x_init], dim=1)
            x_c0 = self.conv0_res(x_c0,D1//2,D2//2, D3//2)
            
            x_c1 = self.conv1(x_c0,D1//4,D2//4, D3//4)
            x_c1 = torch.cat([F.interpolate(x_c1, size=(x_c0.shape[2], x_c0.shape[3], x_c0.shape[4]), mode = 'trilinear',
                                            align_corners=True), x_c0], dim=1)
            x_c1 = self.conv1_res(x_c1, D1//4, D2//4, D3//4)          

            mid_output = x_c1.detach().clone()

            x_c2 = self.conv2(x_c1,D1//2,D2//2, D3//2)
            x_c2 = torch.cat([F.interpolate(x_c1, size=(x_c2.shape[2], x_c2.shape[3], x_c2.shape[4]), mode = 'trilinear',
                                            align_corners=True), x_c2], dim=1)
            x_c2 = self.conv2_res(x_c2, D1//2, D2//2, D3//2) 

            x_c3 = self.conv3(x_c2, int(factor*D1), int(factor*D2), D3//2)
            x_c3 = torch.cat([F.interpolate(x_c2, size=(x_c3.shape[2], x_c3.shape[3], x_c3.shape[4]), mode = 'trilinear',
                                            align_corners=True), x_c3], dim=1)
            x_c3 = self.conv3_res(x_c3,int(factor*D1), int(factor*D2), D3//2)                
            #x_c5 = torch.cat([x_c5, x_fc0], dim=1)

            x_end = self.conv_end(x_c3, D1_raw, D2_raw, D3_raw)
            x_end = torch.cat([F.interpolate(x_c3, size=(x_end.shape[2], x_end.shape[3], x_end.shape[4]), mode = 'trilinear',
                                            align_corners=True), x_end], dim=1)
            x_end = self.conv_end_res(x_end,D1_raw, D2_raw, D3_raw)  
            
            x_end = x_end.permute(0, 2, 3, 4,  1)

            x_fc1 = self.fc1(x_end)
            x_fc1 = F.gelu(x_fc1)

            #x_fc1 = torch.cat([x_fc1, x_fc_1], dim=3)
            x_out = self.fc2(x_fc1)
            x_out = x_out.permute(0, 4, 1, 2, 3)

            if output_encode:
                return mid_output
            else:
                return x_out
        # decode == True
        else: 
            #only accept input resolution
            if pad is None:
                D1_raw, D2_raw, D3_raw = x.shape[-3]*4,x.shape[-2]*4,x.shape[-1]*4
                D1,D2,D3 = x.shape[-3]*4,x.shape[-2]*4,x.shape[-1]*4
            else:
                D1, D2, D3 = x.shape[-3]*4,x.shape[-2]*4,x.shape[-1]*4  #(128, 64, 64)
                D1_raw, D2_raw, D3_raw = D1-pad[0], D2-pad[1], D3-pad[2]#(32, 16, 16)*4    
               
            
            x_c1 = x

            x_c2 = self.conv2(x_c1,D1//2,D2//2, D3//2)
            x_c2 = torch.cat([F.interpolate(x_c1, size=(x_c2.shape[2], x_c2.shape[3], x_c2.shape[4]), mode = 'trilinear',
                                            align_corners=True), x_c2], dim=1)
            x_c2 = self.conv2_res(x_c2, D1//2, D2//2, D3//2) 

            x_c3 = self.conv3(x_c2, int(factor*D1), int(factor*D2), D3//2)
            x_c3 = torch.cat([F.interpolate(x_c2, size=(x_c3.shape[2], x_c3.shape[3], x_c3.shape[4]), mode = 'trilinear',
                                            align_corners=True), x_c3], dim=1)
            x_c3 = self.conv3_res(x_c3,int(factor*D1), int(factor*D2), D3//2)                
            #x_c5 = torch.cat([x_c5, x_fc0], dim=1)

            x_end = self.conv_end(x_c3, D1_raw, D2_raw, D3_raw)
            x_end = torch.cat([F.interpolate(x_c3, size=(x_end.shape[2], x_end.shape[3], x_end.shape[4]), mode = 'trilinear',
                                            align_corners=True), x_end], dim=1)
            x_end = self.conv_end_res(x_end,D1_raw, D2_raw, D3_raw)  
            
            x_end = x_end.permute(0, 2, 3, 4,  1)

            x_fc1 = self.fc1(x_end)
            x_fc1 = F.gelu(x_fc1)

            #x_fc1 = torch.cat([x_fc1, x_fc_1], dim=3)
            x_out = self.fc2(x_fc1)
            x_out = x_out.permute(0, 4, 1, 2, 3)

            return x_out        
        
    def get_grid(self, shape, device):
        """
        Generate positional encoding grid.
        
        Args:
            shape: Input tensor shape
            device: Device to place grid on
            
        Returns:
            torch.Tensor: Positional encoding grid
        """
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        
        # Create coordinate grids
        gridx = torch.tensor(np.linspace(0, 2*np.pi, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        
        gridy = torch.tensor(np.linspace(0, 2*np.pi, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

    
# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class LpLoss(object):
    """
    Lp loss function for relative error calculation.
    
    Args:
        d (int): Dimension parameter
        p (int): Norm order (1 for L1, 2 for L2, etc.)
        size_average (bool): Whether to average over batch
        reduction (bool): Whether to reduce the loss
    """
    
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        """
        Calculate relative Lp loss.
        
        Args:
            x: Predicted tensor
            y: Target tensor
            
        Returns:
            torch.Tensor: Relative loss value
        """
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    
# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def create_log_filename(base_name="autoencoder_training", log_dir="logs"):
    """
    Create a log filename with timestamp.
    
    Args:
        base_name (str): Base name for the log file
        log_dir (str): Directory to store log files
    
    Returns:
        str: Full path to the log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{base_name}_{timestamp}.log"
    return os.path.join(log_dir, log_filename)


def train_encoder(model, optimizer, train_loader, epochs, scheduler=None, saved_model=False, 
                 save_int=500, save_path=None, my_loss=None, device=None, log_file=None):
    """
    Training function for autoencoder model with logging support.
    
    Args:
        model: The autoencoder model
        optimizer: Optimizer for training
        train_loader: Data loader for training data
        epochs: Number of training epochs
        scheduler: Learning rate scheduler (optional)
        saved_model: Whether to save model checkpoints
        save_int: Interval for saving model checkpoints
        save_path: Path to save model checkpoints
        my_loss: Loss function
        device: Device to run training on
        log_file: Path to log file (optional)
    """
    
    # Setup logging if log_file is provided
    if log_file is not None:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Setup logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also print to console
            ]
        )
        logger = logging.getLogger(__name__)
        
        # Log training start
        logger.info("="*50)
        logger.info("AUTOENCODER TRAINING STARTED")
        logger.info("="*50)
        logger.info(f"Model: {type(model).__name__}")
        logger.info(f"Device: {device}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {train_loader.batch_size}")
        logger.info(f"Total batches per epoch: {len(train_loader)}")
        logger.info(f"Optimizer: {type(optimizer).__name__}")
        logger.info(f"Loss function: {type(my_loss).__name__}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        if scheduler:
            logger.info(f"Scheduler: {type(scheduler).__name__}")
        logger.info("="*50)
    
    for ep in range(1, epochs+1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            pred = model(batch)
            optimizer.zero_grad()
            
            loss = my_loss(pred, batch)
            loss.backward()
            optimizer.step()
            
            tr_loss += loss.item()
            batch_count += 1
            
            # Log batch-level information every 100 batches
            if log_file is not None and batch_count % 500 == 0:
                logger.info(f"Epoch {ep}/{epochs} - Batch {batch_count}/{len(train_loader)} - Current Loss: {loss.item():.6f}")
            
        tr_loss /= len(train_loader)
        
        if scheduler:
            scheduler.step()
            if log_file is not None:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Learning rate updated to: {current_lr:.2e}")
        
        t1 = time.time()
        epoch_time = t1 - t0
        
        # Print and log epoch information
        epoch_info = f'tr @ epoch {ep}/{epochs} | Loss {tr_loss:.6f} | {epoch_time:.2f} (s)'
        print(epoch_info)
        
        if log_file is not None:
            logger.info(epoch_info)
            logger.info(f"Epoch {ep} completed in {epoch_time:.2f} seconds")
        
        if saved_model == True:
            if ep % save_int == 0:
                model_path = save_path / f'Encoder_epoch_{ep}.pt'
                torch.save(model.state_dict(), model_path)
                if log_file is not None:
                    logger.info(f"Model saved to: {model_path}")
    
    # Log training completion
    if log_file is not None:
        logger.info("="*50)
        logger.info("AUTOENCODER TRAINING COMPLETED")
        logger.info("="*50)
        logger.info(f"Final training loss: {tr_loss:.6f}")
        logger.info(f"Total training time: {time.time() - t0:.2f} seconds")
        logger.info("="*50)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Configuration parameters
    in_width = 3 + 3  # 6 channels (3 input + 3 grid coordinates)
    width = 32
    
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the UNO_11 autoencoder model
    model = AutoEncoderOperator(in_width=in_width, width=width)
    
    # Calculate and print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model initialized with:")
    print(f"  in_width: {in_width}")
    print(f"  width: {width}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Print model architecture summary
    print(f"\nModel architecture:")
    print(model)
    
    # Test with a sample input to verify the model works
    model = model.to(device)
    
    # Create a sample input tensor (batch_size=2, channels=3, height=128, width=64, depth=48)
    sample_input = torch.randn(2, 3, 128, 64, 48).to(device)
    
    print(f"\nTesting with sample input shape: {sample_input.shape}")
    
    with torch.no_grad():
        # Test encoding
        encoded = model(sample_input, output_encode=True)
        print(f"Encoded shape: {encoded.shape}")
        
        # Test full forward pass
        output = model(sample_input)
        print(f"Output shape: {output.shape}")
        
        # Test decoding from encoded representation
        decoded = model(encoded, decode=True)
        print(f"Decoded shape: {decoded.shape}")
    
    print("Model initialization and testing completed successfully!")
    

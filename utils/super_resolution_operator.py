## minimize over a batch size of 512.
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F
import logging
import os
from datetime import datetime
def kernel_loc(in_chan=3, up_dim=32):
    """
        Kernel network apply on grid
    """
    layers = nn.Sequential(
                nn.Linear(in_chan, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, 2*up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(2*up_dim, 1, bias=False)
            )
    return layers


class SpectralConv3d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim,dim1,dim2,dim3, modes1=None, modes2=None, modes3=None):
        super(SpectralConv3d_Uno, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Default output grid size along x (or 1st dimension of output domain) 
        dim2 = Default output grid size along y ( or 2nd dimension of output domain)
        dim3 = Default output grid size along time t ( or 3rd dimension of output domain)
        Ratio of grid size of the input and output grid size (dim1,dim2,dim3) implecitely 
        set the expansion or contraction farctor along each dimension.
        modes1, modes2, modes3 = Number of fourier modes to consider for the ontegral operator
                                Number of modes must be compatibale with the input grid size 
                                and desired output grid size.
                                i.e., modes1 <= min( dim1/2, input_dim1/2).
                                      modes2 <= min( dim2/2, input_dim2/2)
                                Here input_dim1, input_dim2 are respectively the grid size along 
                                x axis and y axis (or first dimension and second dimension) of the input domain.
                                Other modes also have the same constrain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension   
        """
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

    # Complex multiplication
    def compl_mul3d(self, input, weights):

        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x, dim1 = None,dim2=None,dim3=None):
        """
        dim1,dim2,dim3 are the output grid size along (x,y,t)
        input shape = (batch, in_codim, input_dim1, input_dim2, input_dim3)
        output shape = (batch, out_codim, dim1,dim2,dim3)
        """
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
            self.dim3 = dim3   

        batchsize = x.shape[0]

        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1], norm = 'forward')

        out_ft = torch.zeros(batchsize, self.out_channels, self.dim1, self.dim2, self.dim3//2 + 1, dtype=torch.cfloat, device=x.device)
        #print('out_ft:{}, dim1:{}, dim2:{}, dim3:{}'.format(out_ft.shape, dim1, dim2, dim3))
        #print('x_ft:{}, modes1:{}, modes2:{}, modes3:{}, weights1:{}'.format(x_ft.shape, self.modes1, self.modes2, self.modes3, self.weights1.shape))
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(self.dim1, self.dim2, self.dim3), norm = 'forward')
        return x

class pointwise_op_3D(nn.Module):
    def __init__(self, in_codim, out_codim,dim1, dim2,dim3):
        super(pointwise_op_3D,self).__init__()
        self.conv = nn.Conv3d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.dim3 = int(dim3)

    def forward(self,x, dim1 = None, dim2 = None, dim3 = None):
        """
        dim1,dim2,dim3 are the output dimensions (x,y,t)
        """
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
            dim3 = self.dim3
        x_out = self.conv(x)

        ft = torch.fft.rfftn(x_out,dim=[-3,-2,-1])
        ft_u = torch.zeros_like(ft)
        ft_u[:, :, :(dim1//2), :(dim2//2), :(dim3//2)] = ft[:, :, :(dim1//2), :(dim2//2), :(dim3//2)]
        ft_u[:, :, -(dim1//2):, :(dim2//2), :(dim3//2)] = ft[:, :, -(dim1//2):, :(dim2//2), :(dim3//2)]
        ft_u[:, :, :(dim1//2), -(dim2//2):, :(dim3//2)] = ft[:, :, :(dim1//2), -(dim2//2):, :(dim3//2)]
        ft_u[:, :, -(dim1//2):, -(dim2//2):, :(dim3//2)] = ft[:, :, -(dim1//2):, -(dim2//2):, :(dim3//2)]
        
        x_out = torch.fft.irfftn(ft_u, s=(dim1, dim2, dim3))

        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2,dim3),mode = 'trilinear',align_corners=True)
        return x_out

class OperatorBlock_3D(nn.Module):
    """
    Normalize = if true performs InstanceNorm3d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv3d_Uno class.
    """
    def __init__(self, in_codim, out_codim,dim1, dim2,dim3,modes1,modes2,modes3, Normalize = False,Non_Lin = True):
        super(OperatorBlock_3D,self).__init__()
        self.conv = SpectralConv3d_Uno(in_codim, out_codim, dim1,dim2,dim3,modes1,modes2,modes3)
        self.w = pointwise_op_3D(in_codim, out_codim, dim1,dim2,dim3)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm3d(int(out_codim),affine=True)


    def forward(self,x, dim1 = None, dim2 = None, dim3 = None):
        """
        input shape = (batch, in_codim, input_dim1, input_dim2, input_dim3)
        output shape = (batch, out_codim, dim1,dim2,dim3)
        """
        x1_out = self.conv(x,dim1,dim2,dim3)
        x2_out = self.w(x,dim1,dim2,dim3)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out
    

### 
class SuperResolutionOperator(nn.Module):
    def __init__(self, in_width, width,pad=0, factor = 1, last_conv_model_time=24):
        super(SuperResolutionOperator, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  # pad the domain if input is non-periodic

        self.fc_n1 = nn.Linear(self.in_width, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)
        
        # The input will be : [3, 128, 64, 48] -> [3, 256, 128, 96]
        
        self.conv0 = OperatorBlock_3D(self.width, self.width, 192, 96, 72, 64, 32, 24)
        self.conv0_res = OperatorBlock_3D(2*self.width, self.width, 192, 96, 72, 64, 32, 24, Normalize = True)
          
        self.conv1 = OperatorBlock_3D(self.width, self.width, 256, 128, 96, 80, 40, 24)
        #self.conv1_res = OperatorBlock_3D(2*self.width, self.width, 256, 128, 96, 80, 40, 24) # try 24 and 32, 
        self.conv1_res = OperatorBlock_3D(2*self.width, self.width, 256, 128, 96, 80, 40, last_conv_model_time) # try 24 and 32, 
        
        self.fc1 = nn.Linear(1*self.width, 1*self.width)
        self.fc2 = nn.Linear(1*self.width, in_width-3) # -3 for the grid coordinates

    def forward(self, x, pad=None, factor=1.5, query_scale=1):

        # query_scale only used at inference time
        
        x = x.permute(0, 2, 3, 4, 1) # shape [batch, ndim1, ndim2, ndim, n_channel]
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x_fc_1 = self.fc_n1(x)
        x_fc_1 = F.gelu(x_fc_1)

        x_fc0 = self.fc0(x_fc_1)
        x_fc0 = F.gelu(x_fc0)

        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)

        if pad is None:
            D1_raw, D2_raw, D3_raw = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]
            D1,D2,D3 = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]
            D1, D2, D3 = int(query_scale*D1), int(query_scale*D2), D3 # unchanged
        else:
            D1_raw, D2_raw, D3_raw = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1] #(128, 64, 48)
            D1, D2, D3 = D1_raw+pad[0], D2_raw+pad[1], D3_raw+pad[2] #(128, 64, 64)

        ## Residual connection (encoder part)
        x_c0 = self.conv0(x_fc0,int(factor*D1),int(factor*D2), int(factor*D3))
        x_c0 = torch.cat([F.interpolate(x_fc0, size=(x_c0.shape[2], x_c0.shape[3], x_c0.shape[4]), mode='trilinear',
                                        align_corners=True), x_c0], dim=1)
        x_c0 = self.conv0_res(x_c0,int(factor*D1),int(factor*D2), int(factor*D3))

        x_c1 = self.conv1(x_c0,D1*2,D2*2, D3*2)
        x_c1 = torch.cat([F.interpolate(x_c0, size=(x_c1.shape[2], x_c1.shape[3], x_c1.shape[4]), mode = 'trilinear',
                                        align_corners=True), x_c1], dim=1)
        x_c1 = self.conv1_res(x_c1,D1*2,D2*2, D3*2)          

        x_c3 = x_c1.permute(0, 2, 3, 4,  1)

        x_fc1 = self.fc1(x_c3)
        x_fc1 = F.gelu(x_fc1)

        #x_fc1 = torch.cat([x_fc1, x_fc_1], dim=3)
        x_out = self.fc2(x_fc1)
        x_out = x_out.permute(0, 4, 1, 2, 3)

        return x_out
      
        
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 2*np.pi, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 2*np.pi, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

    
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class MixedLoss(object):
    def __init__(self, alpha=0.5, size_average=True, reduction=True):
        """
        Initializes the mixed loss function.

        Args:
            alpha (float): Weight for the L2 loss term. The L1 loss term will use weight (1 - alpha).
            size_average (bool): Whether to average the loss over the batch.
            reduction (bool): Whether to reduce the loss (sum or average) over the batch.
        """
        self.alpha = alpha
        # Create an LpLoss instance for L2 (p=2) and one for L1 (p=1)
        self.l2_loss = LpLoss(p=2, size_average=size_average, reduction=reduction)
        self.l1_loss = LpLoss(p=1, size_average=size_average, reduction=reduction)

    def __call__(self, x, y):
        # Compute the weighted sum of L2 and L1 losses
        return self.alpha * self.l2_loss(x, y) + (1 - self.alpha) * self.l1_loss(x, y)


def create_log_filename(base_name="training", log_dir="logs", save_path=None):
    """
    Create a log filename with timestamp
    
    Args:
        base_name (str): Base name for the log file
        log_dir (str): Directory to store log files (used if save_path is None)
        save_path: Path object for model checkpoints (if provided, logs will be saved in same directory)
    
    Returns:
        str: Full path to the log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{base_name}_{timestamp}.log"
    
    # If save_path is provided, use the same directory as checkpoints
    if save_path is not None:
        if hasattr(save_path, 'parent'):
            # If save_path is a Path object
            log_dir = str(save_path.parent)
        else:
            # If save_path is a string
            log_dir = os.path.dirname(str(save_path))
    
    return os.path.join(log_dir, log_filename)


# use KL divergence
def train_encoder(model, optimizer, train_loader, epochs, scheduler=None, saved_model=False, save_int=500, save_path=None, my_loss=None, device=None, log_file=None):
    """
    Training function with logging support
    
    Args:
        model: The neural network model
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
        logger.info("TRAINING STARTED")
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
        
        for batch, batch_sup in train_loader:
            batch = batch.to(device)
            batch_sup = batch_sup.to(device)

            #print(f'batch:{batch.shape}, batch_sup:{batch_sup.shape}')
            
            pred = model(batch)
            #print('pred:{}'.format(pred.shape))
            optimizer.zero_grad()
            
            loss = my_loss(pred, batch_sup)
            loss.backward()
            optimizer.step()
            
            tr_loss += loss.item()
            batch_count += 1
            
            # Log batch-level information every 100 batches
            if log_file is not None and batch_count % 500 == 0:
                logger.info(f"Epoch {ep}/{epochs} - Batch {batch_count}/{len(train_loader)} - Current Loss: {loss.item():.6f}")
            
        tr_loss /= len(train_loader)
        #tr_losses.append(tr_loss)
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
        logger.info("TRAINING COMPLETED")
        logger.info("="*50)
        logger.info(f"Final training loss: {tr_loss:.6f}")
        logger.info(f"Total training time: {time.time() - t0:.2f} seconds")
        logger.info("="*50)


if __name__ == "__main__":
    # Create log file for this session
    log_file = create_log_filename("model_test", "logs")
    print(f"Log file will be created at: {log_file}")
    
    # Initialize the SuperResolutionOperator model
    in_width = 3 + 3  # 6 channels (3 input + 3 grid coordinates)
    width = 24
    last_conv_model_time = 32

    model = SuperResolutionOperator(in_width=in_width, width=width, last_conv_model_time=last_conv_model_time)
    
    # Calculate and print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model initialized with:")
    print(f"  in_width: {in_width}")
    print(f"  width: {width}")
    print(f"  last_conv_model_time: {last_conv_model_time}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Print model architecture summary
    print(f"\nModel architecture:")
    print(model)
    
    # Test with a sample input to verify the model works
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create a sample input tensor (batch_size=2, channels=3, height=128, width=64, depth=48)
    sample_input = torch.randn(2, 3, 128, 64, 48).to(device)
    
    print(f"\nTesting with sample input shape: {sample_input.shape}")
    
    with torch.no_grad():
        output = model(sample_input)
        print(f"Output shape: {output.shape}")
    
    print("Model initialization and testing completed successfully!")
    
    # Example of how to use the logging functionality in training:
    print(f"\nTo use logging in training, call train_encoder with log_file parameter:")
    print(f"Example: train_encoder(model, optimizer, train_loader, epochs=100, log_file='{log_file}')")
    
    # Example of saving logs and checkpoints in the same directory:
    from pathlib import Path
    checkpoint_dir = Path("checkpoints")
    log_file_same_dir = create_log_filename("training", save_path=checkpoint_dir)
    print(f"\nTo save logs and checkpoints in the same directory:")
    print(f"checkpoint_dir = Path('checkpoints')")
    print(f"log_file = create_log_filename('training', save_path=checkpoint_dir)")
    print(f"train_encoder(model, optimizer, train_loader, epochs=100, save_path=checkpoint_dir, log_file=log_file)")
    print(f"This would save logs to: {log_file_same_dir}")
    print(f"And checkpoints to: {checkpoint_dir}/Encoder_epoch_*.pt")
    
    # Setup a simple logging demonstration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*50)
    logger.info("MODEL INITIALIZATION COMPLETED")
    logger.info("="*50)
    logger.info(f"Model: SuperResolutionOperator")
    logger.info(f"in_width: {in_width}")
    logger.info(f"width: {width}")
    logger.info(f"last_conv_model_time: {last_conv_model_time}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Device: {device}")
    logger.info(f"Sample input shape: {sample_input.shape}")
    logger.info(f"Sample output shape: {output.shape}")
    logger.info("="*50)
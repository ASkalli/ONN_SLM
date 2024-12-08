import torch
import torch.nn as nn
import torch.nn.functional as F



class Lin_ONN(nn.Module):
    # Class simulating an optical linear neural network
    def __init__(self, input_size=784, SLM_size=100, output_size=10):
        
        super(Lin_ONN, self).__init__()
        
        # Precompute parameter sizes and shapes
        self.param_shapes = []
        self.param_slices = []
        start_idx = 0
        for param in self.parameters():
            param_size = param.numel()
            self.param_shapes.append(param.shape)
            self.param_slices.append(slice(start_idx, start_idx + param_size))
            start_idx += param_size
        
        # SLM parameters (trainable)
        self.SLM_params = nn.Parameter(torch.rand(size=(SLM_size,SLM_size)) * 2 * torch.pi)  # Phase modulation parameters
        self.SLM_size = SLM_size

    def forward(self, X):
        # Apply FFT (forward transform) and reuse X for storing the result
        X = torch.fft.fft2(X)
        
        # Apply FFT shift to center the low frequencies at 0, this is necessary because we want to multiply in the fourier domain
        X = torch.fft.fftshift(X)
        
        # Resize using interpolation, separating real and imaginary parts
        real = F.interpolate(X.real, size=(self.SLM_size, self.SLM_size), mode='bilinear', align_corners=False)
        imag = F.interpolate(X.imag, size=(self.SLM_size, self.SLM_size), mode='bilinear', align_corners=False)
        
        # Combine real and imaginary parts into a complex tensor
        X = real + 1j * imag
        # Apply phase modulation (element-wise multiplication in Fourier domain)
        X *= torch.exp(1j * self.SLM_params)

        # Apply Inverse FFT to bring it back to real space
        X = torch.fft.ifft2(X)

        # Compute intensity (magnitude squared) for camera detection
        X = torch.abs(X) ** 2

        # Output projection
        logits = X

        return logits

    
    def count_parameters(self):
        """
        Counts the number of trainable parameters in a PyTorch model.

        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_params(self):
        """
        method that returns parameters of the model as a 1D array.
        """
        params_list = []
        for param in self.parameters():
            #view(-1) flattens the tensor
            params_list.append(param.view(-1))    
        full_params = torch.cat(params_list)
        return full_params
    
    def set_params(self, params_to_send):
        """
        Efficiently set parameters in the network.
        """
        params_to_send_tensor = torch.from_numpy(params_to_send)
        for param, param_shape, param_slice in zip(self.parameters(), self.param_shapes, self.param_slices):
            param.data.copy_(params_to_send_tensor[param_slice].view(param_shape))
 
    def forward_pass_params(self, params_to_send, X):
        """
        This method is a forward pass that also takes in the parameters of the neural network as a variable,
        to use in online learning.
        """
        self.set_params(params_to_send)
        logits = self.forward(X)
        return logits

# Transform to convert images to Tensor and then threshold them
def to_boolean_tensor(image):
    """
    Convert the image to a tensor and apply a threshold of 0.5 (or any other threshold you prefer)
    """
    return (image>0.1).float()
    
def padd_images(image_tensor,pad_length):
    """
    Padd image tensor    
    """
    padded_tensor = F.pad(image_tensor, (pad_length, pad_length, pad_length, pad_length))  # (left, right, top, bottom)
    
    return padded_tensor

def gaussian_beam(n_pixels):
    """
    Generates a centered Gaussian field beam of amplitude 1 centered in 0
    
    in a square array of size n_pixels * n_pixels
    """
    
    # Number of points in each dimension
    N = n_pixels

    # Create 1D grids for x and y
    x = torch.linspace(-round(N / 2), round(N / 2), N)
    y = torch.linspace(-round(N / 2), round(N / 2), N)

    # Create 2D meshgrid
    X, Y = torch.meshgrid(x, y, indexing='ij')  # Use indexing='ij' for Cartesian indexing

    # Gaussian parameters
    x0, y0 = 0, 0  # Center of the Gaussian
    fwhm = 28     # Full-width at half-maximum

    # Convert FWHM to standard deviation (sigma)
    sigma = fwhm / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0))))

    # Compute the 2D Gaussian field
    E = torch.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    return E



def generate_blaze_grating_phase(size, period, angle):
    """
    Generates a blaze grating in phase using PyTorch.

    Parameters:
        size (tuple): The size of the array (height, width).
        period (float): The period of the grating in pixels.
        angle (float): The orientation angle of the grating in degrees.

    Returns:
        torch.Tensor: A 2D tensor representing the blaze grating in phase.
    """
    # Define the grid
    height, width = size
    y = torch.linspace(-height // 2, height // 2, height)
    x = torch.linspace(-width // 2, width // 2, width)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Convert angle to radians
    theta = torch.deg2rad(torch.tensor(angle))

    # Compute the grating phase
    phase = 2 * torch.pi * (X * torch.cos(theta) + Y * torch.sin(theta)) / period

    # Wrap the phase to the range [0, 2π]
    blaze_phase = phase % (2 * torch.pi)

    return blaze_phase
    
#class  for building train and test loaders so we can use other datasets with pytorch loop
class Custom_dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)  # Convert and squeeze labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
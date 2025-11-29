import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2
        self.modes3 = modes3

        scale = (1 / (in_channels * out_channels))
        # Weights are complex
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,z ), (in_channel, out_channel, x,y,z) -> (batch, out_channel, x,y,z)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Compute Fourier coeff
        # rfftn: Real -> Complex FFT
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Prepare Output tensor
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Multiply relevant modes (Corner filtering)
        # Ensure we don't exceed dimensions if grid is small
        m1 = min(self.modes1, x_ft.size(-3))
        m2 = min(self.modes2, x_ft.size(-2))
        m3 = min(self.modes3, x_ft.size(-1))
        
        # Upper corner
        out_ft[:, :, :m1, :m2, :m3] = \
            self.compl_mul3d(x_ft[:, :, :m1, :m2, :m3], self.weights1[:self.in_channels, :self.out_channels, :m1, :m2, :m3])

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, modes=8, width=20, in_channels=3, out_channels=2):
        super(FNO3d, self).__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.modes3 = modes
        self.width = width
        
        # Input projection (Grid + Features -> Hidden Width)
        self.p = nn.Linear(in_channels, self.width) 
        
        # Fourier Layers
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)

        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w1 = nn.Conv3d(self.width, self.width, 1)

        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w2 = nn.Conv3d(self.width, self.width, 1)

        # Output projection
        self.q = nn.Linear(self.width, 128)
        self.out = nn.Linear(128, out_channels) 

    def forward(self, x):
        # x shape: (Batch, GridX, GridY, GridZ, Channels)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3) # (Batch, Channels, X, Y, Z)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x = x.permute(0, 2, 3, 4, 1) # Back to (Batch, X, Y, Z, Channels)
        x = self.q(x)
        x = F.gelu(x)
        x = self.out(x)
        return x

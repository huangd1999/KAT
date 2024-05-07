import torch
import torch.nn as nn
import torch.nn.functional as F

class RealImagConv1d(nn.Module):
    """A 1D convolution layer that operates separately on the real and imaginary parts."""
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(RealImagConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels * 2, out_channels * 2, kernel_size, padding=padding)
    
    def forward(self, x):
        # Expecting x to have dimensions [batch, channels*2, length] where channels are interleaved real and imaginary parts
        return self.conv(x)

class TimeFrequencyFusionBlock(nn.Module):
    def __init__(self, channels, freq_dim):
        super(TimeFrequencyFusionBlock, self).__init__()
        # Time domain branch
        self.time_branch = nn.Sequential(
            nn.Conv1d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(),
        )
        
        # Frequency domain branch, simplified to operate on real and imaginary parts separately
        self.freq_branch = nn.Sequential(
            RealImagConv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(),
        )
        
        # Fusion layer
        self.fusion = nn.Linear(channels * 4, channels * 2)  # Adjusting for the doubled channels (real+imaginary)
    
    def forward(self, x):
        # Convert x to interleaved real and imaginary parts if coming as complex
        if torch.is_complex(x):
            x = torch.view_as_real(x).permute(0, 3, 1, 2).flatten(start_dim=1, end_dim=2)
        
        # Time domain processing
        time_out = self.time_branch(x)
        
        # Frequency domain processing (no need to explicitly transform to frequency domain in this adjusted approach)
        freq_out = self.freq_branch(x)
        
        # Fusion of time and frequency signals
        fused_out = torch.cat([time_out, freq_out], dim=1)
        
        # Fusion layer to combine features
        out = self.fusion(fused_out.transpose(1, 2)).transpose(1, 2)
        
        return F.relu(out)  # ReLU can now be applied since we're working with real values

# Demonstrating usage
batch_size, channels, seq_length = 4, 16, 1024
# Simulating real and imaginary parts by doubling the channel size
input = torch.randn(batch_size, channels * 2, seq_length)
model = TimeFrequencyFusionBlock(channels, seq_length)
output = model(input)

print(output.shape)

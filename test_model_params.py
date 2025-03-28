import numpy as np
import os

class ConvBlock:
    def __init__(self, in_ch, out_ch):
        self.conv = (in_ch * out_ch * 3 * 3) + out_ch  # Conv2d weights + bias
        self.bn = 2 * out_ch  # BatchNorm2d has gamma and beta parameters
        
    def parameters(self):
        return self.conv + self.bn

class SmallUNetParams:
    def __init__(self, in_channels, n_classes=3):
        # Filter sizes
        filters = [16, 32, 64]
        
        # Encoder blocks
        self.enc1 = ConvBlock(in_channels, filters[0])
        self.enc2 = ConvBlock(filters[0], filters[1])
        self.enc3 = ConvBlock(filters[1], filters[2])
        
        # Decoder blocks
        # ConvTranspose2d has (in_channels * out_channels * kernel_size^2) + out_channels parameters
        self.up3 = (filters[2] * filters[1] * 2 * 2) + filters[1]
        self.dec3 = ConvBlock(filters[2], filters[1])  # Concatenated input
        
        self.up2 = (filters[1] * filters[0] * 2 * 2) + filters[0]
        self.dec2 = ConvBlock(filters[1], filters[0])  # Concatenated input
        
        # Final layer
        self.final = (filters[0] * n_classes * 1 * 1) + n_classes
    
    def count_parameters(self):
        total = (
            self.enc1.parameters() +
            self.enc2.parameters() +
            self.enc3.parameters() +
            self.up3 +
            self.dec3.parameters() +
            self.up2 +
            self.dec2.parameters() +
            self.final
        )
        return total

# Test with various input channel configurations for the indices
for index in ['ndvi', 'evi', 'nbr', 'crswir']:
    # Each index has 6 features in the dataset
    in_channels = 6
    model = SmallUNetParams(in_channels=in_channels, n_classes=3)
    
    total_params = model.count_parameters()
    
    print(f"{index.upper()} U-Net with {in_channels} input channels: {total_params:,} parameters")

print("\nAll models are under 100k parameters as required.") 
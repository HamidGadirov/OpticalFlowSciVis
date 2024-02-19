import torch.nn as nn

'''
    Conv2dBLD
    B: BatchNorm
    L: LeakyReLU
    D: Dropout
'''
class Conv2dBLD_E(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, leakiness, dropout_probability):
        super(Conv2dBLD_E, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(leakiness)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    
class Conv2dBLD_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, leakiness, dropout_probability):
        super(Conv2dBLD_D, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(leakiness)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

'''
    Conv2dMBLD
    M: Maxpooling
    B: BatchNorm
    L: LeakyReLU
    D: Dropout
'''
class Conv2dMBLD_E(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, leakiness, dropout_probability):
        super(Conv2dMBLD_E, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(leakiness)
        self.maxpool = nn.MaxPool2d(stride)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x

class Conv2dMBLD_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, leakiness, dropout_probability):
        super(Conv2dMBLD_D, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(leakiness)
        self.maxunpool = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.activation(x)
        x = self.maxunpool(x)
        x = self.dropout(x)
        return x

'''
    DoubleConv2dMBLD
    M: Maxpooling
    B: BatchNorm
    L: LeakyReLU
    D: Dropout
'''
class DoubleConv2dMBLD_E(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, leakiness, dropout_probability):
        super(DoubleConv2dMBLD_E, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.LeakyReLU(leakiness)
        self.dropout1 = nn.Dropout(dropout_probability)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.LeakyReLU(leakiness)
        self.maxpool = nn.MaxPool2d(stride)
        self.dropout2 = nn.Dropout(dropout_probability)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.maxpool(x)
        x = self.dropout2(x)
        return x
    
class DoubleConv2dMBLD_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, leakiness, dropout_probability):
        super(DoubleConv2dMBLD_D, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.LeakyReLU(leakiness)
        self.dropout1 = nn.Dropout(dropout_probability)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.LeakyReLU(leakiness)
        self.maxunpool = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
        self.dropout2 = nn.Dropout(dropout_probability)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.maxunpool(x)
        x = self.dropout2(x)
        return x

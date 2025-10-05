import math
from typing import Tuple, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SincBandpass(nn.Module):
    """Differentiable Sinc-based bandpass filter layer"""
    def __init__(self, sample_rate: float = 30.0, min_freq: float = 0.7, max_freq: float = 4.0, 
                 filter_length: int = 65, learnable: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.filter_length = filter_length
        
        # Initialize cutoff frequencies as learnable parameters
        if learnable:
            self.low_freq = nn.Parameter(torch.tensor(min_freq))
            self.high_freq = nn.Parameter(torch.tensor(max_freq))
        else:
            self.register_buffer('low_freq', torch.tensor(min_freq))
            self.register_buffer('high_freq', torch.tensor(max_freq))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        
        # Create time indices
        n = torch.arange(self.filter_length, device=x.device, dtype=x.dtype) - self.filter_length // 2
        
        # Avoid division by zero
        n_safe = torch.where(n == 0, torch.tensor(1e-7, device=x.device), n.float())
        
        # Normalized frequencies
        low_norm = 2 * self.low_freq / self.sample_rate
        high_norm = 2 * self.high_freq / self.sample_rate
        
        # Sinc functions for low and high cutoffs
        sinc_low = torch.sin(math.pi * low_norm * n_safe) / (math.pi * n_safe)
        sinc_high = torch.sin(math.pi * high_norm * n_safe) / (math.pi * n_safe)
        
        # Handle n=0 case
        sinc_low = torch.where(n == 0, low_norm, sinc_low)
        sinc_high = torch.where(n == 0, high_norm, sinc_high)
        
        # Bandpass = highpass - lowpass = sinc_high - sinc_low
        bandpass_filter = sinc_high - sinc_low
        
        # Apply Hamming window
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(self.filter_length, device=x.device) / (self.filter_length - 1))
        bandpass_filter = bandpass_filter * window
        
        # Normalize
        bandpass_filter = bandpass_filter / torch.sum(bandpass_filter)
        
        # Reshape for conv1d: (out_channels, in_channels//groups, kernel_size)
        filter_kernel = bandpass_filter.view(1, 1, -1).repeat(C, 1, 1)
        
        # Apply convolution with padding
        padding = self.filter_length // 2
        filtered = F.conv1d(x, filter_kernel, padding=padding, groups=C)
        
        return filtered


class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        
        # Global average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        
        # Combine and apply sigmoid
        attention = torch.sigmoid(avg_out + max_out).view(B, C, 1)
        
        return x * attention


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class MultiScaleBranch(nn.Module):
    """Multi-scale branch with different kernel sizes and dilations"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MDAR(nn.Module):
    """
    Enhanced Multi-scale Dilated Attention RPPG network with:
    - Multi-branch architecture with different kernel sizes
    - Channel attention mechanisms
    - Optional differentiable bandpass preprocessing
    - Multi-task heads for waveform, HR, and confidence estimation
    """

    def __init__(self, in_features: int = 4, hidden_channels: int = 64, dropout: float = 0.3,
                 sample_rate: float = 30.0, use_bandpass: bool = True, multitask: bool = True):
        super().__init__()
        self.multitask = multitask
        
        # Optional differentiable bandpass preprocessing
        self.use_bandpass = use_bandpass
        if use_bandpass:
            self.bandpass = SincBandpass(sample_rate=sample_rate, learnable=True)
        
        # Input projection
        self.proj = nn.Conv1d(in_features, hidden_channels, kernel_size=1)
        
        # Multi-scale branches
        self.branch1 = MultiScaleBranch(hidden_channels, hidden_channels//4, kernel_size=3, dilation=1)
        self.branch2 = MultiScaleBranch(hidden_channels, hidden_channels//4, kernel_size=5, dilation=2) 
        self.branch3 = MultiScaleBranch(hidden_channels, hidden_channels//4, kernel_size=7, dilation=4)
        self.branch4 = MultiScaleBranch(hidden_channels, hidden_channels//4, kernel_size=9, dilation=8)
        
        # Channel attention
        self.channel_attention = ChannelAttention(hidden_channels)
        
        # Depthwise separable convolutions
        self.ds_conv1 = DepthwiseSeparableConv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        self.ds_conv2 = DepthwiseSeparableConv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2)
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels//8, 1, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(p=dropout)
        self.bn_final = nn.BatchNorm1d(hidden_channels)
        
        # Multi-task heads
        self.waveform_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        
        if multitask:
            # Heart rate estimation head (global pooling + FC)
            self.hr_head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(hidden_channels, hidden_channels//4),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_channels//4, 1)
            )
            
            # Confidence estimation head 
            self.confidence_head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(hidden_channels, hidden_channels//4),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_channels//4, 1),
                nn.Sigmoid()  # Confidence in [0,1]
            )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        
        # Optional bandpass preprocessing
        if self.use_bandpass:
            x = self.bandpass(x)
        
        # Input projection
        x = self.proj(x)
        
        # Multi-scale branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # Concatenate branches
        x = torch.cat([b1, b2, b3, b4], dim=1)
        
        # Channel attention
        x = self.channel_attention(x)
        
        # Depthwise separable convolutions with residual
        residual = x
        x = self.ds_conv1(x)
        x = self.dropout(x)
        x = self.ds_conv2(x)
        x = self.dropout(x)
        x = x + residual  # Residual connection
        
        # Temporal attention
        temporal_weights = self.temporal_attention(x)
        x = x * temporal_weights
        
        x = self.bn_final(x)
        
        # Waveform prediction
        waveform = self.waveform_head(x).squeeze(1)  # (B, T)
        
        if not self.multitask:
            return waveform
        
        # Multi-task outputs
        hr = self.hr_head(x).squeeze(-1)  # (B,)
        confidence = self.confidence_head(x).squeeze(-1)  # (B,)
        
        return {
            'waveform': waveform,
            'heart_rate': hr,
            'confidence': confidence
        }



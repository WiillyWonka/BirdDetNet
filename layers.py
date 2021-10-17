import torch
import torch.nn as nn
import kindle
from kindle import Model
from kindle.utils.torch_utils import count_model_params
from kindle.generator import GeneratorAbstract
import numpy as np
from typing import Any, Dict, List, Union
import os


def autopad_pool(k):
    assert( k % 2 != 0)
    pad = k // 2
    return pad

class ConvBNReLU(nn.Module):
    def __init__(self,in_channel : int,out_channel: int,use_3x3 : bool,dilation_ratio : int):
        super(ConvBNReLU,self).__init__()
        if dilation_ratio == 1:
            self.conv = nn.Conv2d(in_channel,out_channel,3,1,1,dilation=dilation_ratio,bias=False) if use_3x3 is True else nn.Conv2d(in_channel,out_channel,1,1,dilation=dilation_ratio,bias=False)
        else:
            self.conv = nn.Conv2d(in_channel, out_channel, 3, 1, padding=dilation_ratio, dilation=dilation_ratio, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU()

    def forward(self,x : torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class CAM(nn.Module):
    def __init__(self, channels : int,kernel_size : int,padding : int,reduction_rate : int,use_3x3 : bool):
        super(CAM, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size,1,padding)
        self.red_conv = nn.Conv2d(channels,channels//reduction_rate,1,1,bias=False)
        if use_3x3==False:
            self.up_conv = nn.Conv2d(channels//reduction_rate,channels,1,1,bias=False)
        else:
            self.up_conv = nn.Conv2d(channels//reduction_rate, channels, 3, 1, 1,bias=False)
        self.sigm = nn.Sigmoid()

    def forward(self,x : torch.Tensor) -> torch.Tensor:
        return x * (self.sigm(self.up_conv(self.red_conv(self.pool(x)))))

class BDNblockCSP(nn.Module):
    def __init__(self, channels : int,out_channels: int,cam_kernel_size:int, cam_reduction_rate:int,e : float,dilation_ratio : int,use_cam : bool,cam_3x3 :bool,skip_1x1 : bool):
        super(BDNblockCSP, self).__init__()
        c_ = int(channels * e)
        self.cv1 = ConvBNReLU(channels,c_,False,dilation_ratio)
        self.cv2 = ConvBNReLU(channels,c_,False,dilation_ratio)
        self.cv3 = ConvBNReLU(channels, channels, False,dilation_ratio)
        self.s1 = ConvBNReLU(c_,c_,True,dilation_ratio)
        self.s2 = ConvBNReLU(c_,c_,True,dilation_ratio)
        self.c3x3= nn.Conv2d(c_,c_,3,1,1,bias=False)
        self.bn = nn.BatchNorm2d(c_)
        self.act = nn.ReLU()
        self.cam = CAM(channels,cam_kernel_size,autopad_pool(cam_kernel_size),cam_reduction_rate,cam_3x3) if use_cam is True else nn.Identity()
        self.skip_conn = nn.Conv2d(c_,c_,1,1,bias=False) if skip_1x1 is True else nn.Identity()

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x_1 = self.cv1(x)
        x_2 = self.cv2(x)
        x_2 = self.act(self.skip_conn(x_2)+self.bn(self.c3x3(self.s2(self.s1(x_2)))))
        x = self.cv3(torch.cat((x_1,x_2),dim=1))
        return self.cam(x)

class UpsampleBlock(nn.Module):
    def __init__(self, channels : int,out_channels:int,fuse_type : str):
        super(UpsampleBlock,self).__init__()
        self.s1 = ConvBNReLU(channels//2,channels//2,True,1) if fuse_type is "sum" else ConvBNReLU(channels,channels//2,True,1)
        self.up = nn.ConvTranspose2d(channels,channels//2,2,2,bias=False)
        self.fuse_type = fuse_type

    def forward(self,x1 : torch.Tensor,x2 : torch.Tensor) -> torch.Tensor:
        x1_1 = self.up(x1)
        if self.fuse_type == "sum":
            x = x1_1 + x2
        else:
            x = torch.cat((x1_1,x2),dim=1)
        return self.s1(x)

class Head(nn.Module):
    def __init__(self, channels : int,out_channels : int,inter_channels : int,dilation_ratio: int):
        super(Head, self).__init__()
        self.s1 =  ConvBNReLU(channels,inter_channels,True,dilation_ratio)
        self.s2 =  ConvBNReLU(inter_channels,inter_channels,True,dilation_ratio)
        self.out = nn.Conv2d(inter_channels,out_channels,1,1,bias=False)

    def forward(self,x : torch.Tensor) -> torch.Tensor:
        return self.out(self.s2(self.s1(x)))

class BDNblockCSPGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self._get_divisible_channel(self.args[0] * self.width_multiply)

    @property
    def in_channel(self) -> int:
        if isinstance(self.from_idx, list):
            raise Exception("from_idx can not be a list.")
        return self.in_channels[self.from_idx]

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        module = self(repeat=repeat)
        module.eval()
        module_out = module(torch.zeros([1, *list(size)]))
        return list(module_out.shape[-3:])

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        kwargs = self._get_kwargs(BDNblockCSP, args)
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        if repeat > 1:
            module = [BDNblockCSP(**self.kwargs) for _ in range(repeat)]
        else:
            module = BDNblockCSP(**self.kwargs)

        return self._get_module(module)

class UpsampleBlockGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return self._get_divisible_channel(self.args[0] * self.width_multiply)

    @property
    def in_channel(self) -> int:
        return self.in_channels[self.from_idx[0]]

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        module = self(repeat=repeat)
        module.eval()
        module_out = module(torch.zeros([1, *list(size)[0]]),torch.zeros([1, *list(size)[1]]))
        return list(module_out.shape[-3:])

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        kwargs = self._get_kwargs(UpsampleBlock, args)
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        if repeat > 1:
            module = [UpsampleBlock(**self.kwargs) for _ in range(repeat)]
        else:
            module = UpsampleBlock(**self.kwargs)

        return self._get_module(module)

class HeadGenerator(GeneratorAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        return int(self.args[0])

    @property
    def in_channel(self) -> int:
        if isinstance(self.from_idx, list):
            raise Exception("from_idx can not be a list.")
        return self.in_channels[self.from_idx]

    @torch.no_grad()
    def compute_out_shape(self, size: np.ndarray, repeat: int = 1) -> List[int]:
        module = self(repeat=repeat)
        module.eval()
        module_out = module(torch.zeros([1, *list(size)]))
        return list(module_out.shape[-3:])

    @property
    def kwargs(self) -> Dict[str, Any]:
        args = [self.in_channel, self.out_channel, *self.args[1:]]
        kwargs = self._get_kwargs(Head, args)
        return kwargs

    def __call__(self, repeat: int = 1) -> nn.Module:
        if repeat > 1:
            module = [Head(**self.kwargs) for _ in range(repeat)]
        else:
            module = Head(**self.kwargs)
        return self._get_module(module)








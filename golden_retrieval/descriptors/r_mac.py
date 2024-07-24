import torch
from torch import nn
import numpy as np

from ..registry import DESCRIPTORS


@DESCRIPTORS.register
class RMAC(nn.Module):
    """
    Regional Maximum activation of convolutions (R-MAC).
    c.f. https://arxiv.org/pdf/1511.05879.pdf
    """
    
    def __init__(self, input_shape) -> None:
        super().__init__()
        self.iou_target: float = 0.4
        self.levels: int = 3
        self.regions: list = self.__get_regions(input_shape)
        
    """
    Get regions based on input shape.
    Args:
        input_shape (List[int, int, int, int]): Input shape (batch_size, height, width, channels).
    Returns:
        List[List[int, int, int, int]]: Regions (start_y, start_x, end_y, end_x).
    """
    def __get_regions(self, input_shape: 'list[int, int, int, int]') -> 'list[list[int, int, int, int]]':
        _, _, inp_height, inp_width = input_shape
        min_edge = inp_height
        n_h, n_w = 1, 1
        if inp_height != inp_width:
            min_edge = min(inp_height, inp_width)
            left_space = max(inp_height, inp_width) - min_edge
            iou_now = (min_edge**2 - min_edge*(left_space/np.array(range(1,7)))) / (min_edge**2)
            idx = np.argmin(np.abs(iou_now-self.iou_target)) + 2
            if inp_height > inp_width:
                n_h = idx
            else:
                n_w = idx
            
        regions = []
        for level in range(self.levels):
            region_size = int(2 * min_edge / (level+2))
            step_size_h = (inp_height - region_size) // n_h
            step_size_w = (inp_width - region_size) // n_w

            for y in range(n_h):
                for x in range(n_w):
                    st_y = y * step_size_h
                    ed_y = st_y + region_size - 1
                    st_x = x * step_size_w
                    ed_x = st_x + region_size - 1
                    regions.append([st_y, st_x, ed_y, ed_x])
            n_h += 1
            n_w += 1
        
        return regions
    
    def forward(self, feature)-> 'torch.Tensor':
        final_feature = None
        for region in self.regions:
            region_feature = (feature[:, :, region[0]:region[2], region[1]:region[3]].max(dim=3)[0])
            region_feature = region_feature.max(dim=2)[0]
            region_feature = region_feature / torch.norm(region_feature, dim=1, keepdim=True)
            if final_feature is None:
                final_feature = region_feature
            else:
                final_feature = final_feature + region_feature
        
        return final_feature
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

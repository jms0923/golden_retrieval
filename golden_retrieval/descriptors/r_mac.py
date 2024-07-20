import torch

from ..registry import DESCRIPTORS


@DESCRIPTORS.register
class RMAC():
    """
    Regional Maximum activation of convolutions (R-MAC).
    c.f. https://arxiv.org/pdf/1511.05879.pdf
    """
    
    def __init__(self) -> None:
        super(RMAC, self).__init__()
        self.regions = dict()
        
    def __get_regions(self) -> None:
        pass
    
    def __call__(self) -> None:
        pass
    
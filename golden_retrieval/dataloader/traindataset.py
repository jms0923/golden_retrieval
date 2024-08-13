import os
import pickle
import torch

from utils import pil_loader, default_loader, cid2filename


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, mode, name, imsize=None, loader=default_loader):
        db_root = os.path.join(dataroot, 'train', name)
        img_root = os.path.join(db_root, 'ims')

        '''
        pickle format
        db =
        {
            'val' =
            {
                'cids' = [],
                'qidxs' = [],
                'pidxs' = [],
                'cluster' = []
            },
            'train' =
            {
                'cids' = [],
                'qidxs' = [],
                'pidxs' = [],
                'cluster' = []
            }
        }
        '''
        pkl_path = os.path.join(db_root, '{}.pkl'.format(name))
        with open(pkl_path, 'rb') as f:
            db = pickle.load(f)
            db = db[mode]
        
        self.images = [cid2filename(db['cids'][i], img_root) for i in range(len(db['cids']))]
        self.name = name
        self.mode = mode
        self.imsize = imsize
        self.cluster = db['cluster']
        self.qpool = db['qidxs']
        self.ppool = db['pidxs']
        self.loader = default_loader

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
testdb = TrainDataset('/home/jo/develop/golden_retrieval/dataset', 'train', 'retrieval-SfM-120k')
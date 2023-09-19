import h5py
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MrHiSumDataset(Dataset):

    def __init__(self, mode):
        self.mode = mode
        self.dataset = 'dataset/mrsum.h5'
        self.split_file = 'dataset/mrsum_split.json'
        
        self.video_data = h5py.File(self.dataset, 'r')

        with open(self.split_file, 'r') as f:
            self.data = json.loads(f.read())

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.data[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features and gtscores
        test mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """
        video_name = self.data[self.mode + '_keys'][index]
        d = {}
        d['video_name'] = video_name
        d['features'] = torch.Tensor(np.array(self.video_data[video_name + '/features']))
        d['gtscore'] = torch.Tensor(np.array(self.video_data[video_name + '/gtscore']))

        if self.mode != 'train':
            n_frames = d['features'].shape[0]
            cps = np.array(self.video_data[video_name + '/change_points'])
            d['n_frames'] = np.array(n_frames)
            d['picks'] = np.array([i for i in range(n_frames)])
            d['change_points'] = cps
            d['n_frame_per_seg'] = np.array([cp[1]-cp[0] for cp in cps])
            d['gt_summary'] = np.expand_dims(np.array(self.video_data[video_name + '/gt_summary']), axis=0)
        
        return d
    

class BatchCollator(object):
    def __call__(self, batch):
        video_name, features, gtscore= [],[],[]
        # cps, nseg, n_frames, picks, gt_summary = [], [], [], [], []

        try:
            for data in batch:
                video_name.append(data['video_name'])
                features.append(data['features'])
                gtscore.append(data['gtscore'])
                # cps.append(data['change_points'])
                # nseg.append(data['n_frame_per_seg'])
                # n_frames.append(data['n_frames'])
                # picks.append(data['picks'])
                # gt_summary.append(data['gt_summary'])
        except:
            print('Error in batch collator')

        lengths = torch.LongTensor(list(map(lambda x: x.shape[0], features)))
        max_len = max(list(map(lambda x: x.shape[0], features)))

        mask = torch.arange(max_len)[None, :] < lengths[:, None]
        
        frame_feat = pad_sequence(features, batch_first=True)
        gtscore = pad_sequence(gtscore, batch_first=True)

        batch_data = {'video_name' : video_name, 'features' : frame_feat, 'gtscore':gtscore, 'mask':mask}
        # batch_data = {'video_name' : video_name, 'features' : frame_feat, 'gtscore':gtscore, 'mask':mask, \
        #               'n_frames': n_frames, 'picks': picks, 'n_frame_per_seg': nseg, 'change_points': cps, \
        #                 'gt_summary': gt_summary}
        return batch_data
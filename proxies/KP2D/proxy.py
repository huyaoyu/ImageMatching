
# Original file is created by
# Copyright 2020 Toyota Research Institute. 
# Example usage: python scripts/eval_keypoint_net.sh --pretrained_model /data/models/kp2d/v4.pth --input_dir /data/datasets/kp2d/HPatches/

import cv2
import numpy as np

import torch

from third_party.KP2D.kp2d.networks.keypoint_net import KeypointNet
from third_party.KP2D.kp2d.networks.keypoint_resnet import KeypointResnet
from third_party.KP2D.kp2d.utils.image import to_color_normalized, to_gray_normalized

from proxies import Proxy
from proxies.utils.image_utils import resize_by_longer_edge
from proxies.utils.tensor_utils import convert_2_tensor
from proxies.register import ( PROXIES, register )

def get_k_best(scores, points, descriptors, k):
    """ Select the k most probable points (and strip their probability).
    points has shape (num_points, 3) where the last coordinate is the probability.

    Parameters
    ----------
    scores: numpy.ndarray (N,)
        The confidence score of individual keypoint.
    points: numpy.ndarray (N,3)
        Keypoint vector, consisting of (x,y,probability).
    descriptors: numpy.ndarray (N,256)
        Keypoint descriptors.
    k: int
        Number of keypoints to select, based on probability.
    Returns
    -------
    
    selected_points: numpy.ndarray (k,2)
        k most probable keypoints.
    selected_descriptors: numpy.ndarray (k,256)
        Descriptors corresponding to the k most probable keypoints.
    """
    sortedIdx = scores.argsort()

    sorted_score = scores[sortedIdx]
    sorted_prob  = points[sortedIdx, :]
    sorted_desc  = descriptors[sortedIdx, :]

    start = min(k, points.shape[0])

    selected_scores      = sorted_score[-start:]
    selected_points      = sorted_prob[-start:, :]
    selected_descriptors = sorted_desc[-start:, :]

    return selected_scores, selected_points, selected_descriptors

@register(PROXIES)
class ProxyKP2D(Proxy):
    @classmethod
    def get_default_init_args(cls):
        return dict(
            type=cls.__name__,
            pretrained_model='v4.ckpt',
            conf_threshold=0,
            n_matches=1000,
            longer_edge=1024
        )

    def __init__(self, pretrained_model, conf_threshold, n_matches, longer_edge=1024):
        '''
        pretrained_model (string): The filename of the pretrained model.
        conf_threshold (float): The confidence threshold. Use zero to trust all.
        n_matches (int): Only keep the best n_matches results.
        longer_edge (int): The input image will be reshaped to have its longer edge equal this value. 
            Set zero to disable.
        '''
        super(ProxyKP2D, self).__init__(task_type='extraction')

        self.pretrained_fn  = pretrained_model
        self.conf_threshold = conf_threshold
        self.n_matches      = n_matches
        self.longer_edge    = longer_edge

        self.ori_shape = None # The original shape of the input image.
        self.new_shape = None # The shape of the images after re-scale, if there are any.

    def initialize(self):
        super().initialize()

        print('Initializing KP2D...')

        # Load the checkpoint.
        checkpoint = torch.load(self.pretrained_fn)
        model_args = checkpoint['config']['model']['params']

        # Check model type
        if 'keypoint_net_type' in checkpoint['config']['model']['params']:
            net_type = checkpoint['config']['model']['params']
        else:
            net_type = KeypointNet # default when no type is specified

        # Create and load keypoint net
        if net_type is KeypointNet:
            keypoint_net = KeypointNet(use_color=model_args['use_color'],
                                    do_upsample=model_args['do_upsample'],
                                    do_cross=model_args['do_cross'])
        else:
            keypoint_net = KeypointResnet() # KeypointResnet needs (320,256)

        keypoint_net.load_state_dict(checkpoint['state_dict'])
        keypoint_net = keypoint_net.cuda()
        keypoint_net.eval()
        keypoint_net.training = False

        # Save the model.
        self.model = keypoint_net

        print('KP2D initialized. ')

    def __call__(self, inputs):
        img_0 = inputs['img_0']
        img_1 = inputs['img_1']

        # Save the image size.
        assert( img_0.shape == img_1.shape ), \
            f'img_0.shape = {img_0.shape}, img_1.shape = {img_1.shape}'
        self.ori_shape = img_0.shape

        # Rezie.
        if ( self.longer_edge > 0 ):
            img_0 = resize_by_longer_edge(img_0, self.longer_edge)
            img_1 = resize_by_longer_edge(img_1, self.longer_edge)
        self.new_shape = img_0.shape

        t0 = convert_2_tensor(img_0)
        t1 = convert_2_tensor(img_1)

        if img_0.shape[1] == 3:
            ref = to_color_normalized(t0.cuda())
            tst = to_color_normalized(t1.cuda())
        else:
            ref = to_gray_normalized(t0.cuda())
            tst = to_gray_normalized(t1.cuda())

        if ref.shape[1] == 1:
            ref = ref.repeat(1, 3, 1, 1)
            tst = tst.repeat(1, 3, 1, 1)

        merged = torch.cat((ref, tst), dim=0)

        with torch.no_grad():
            score, coord, desc = self.model(merged)
            score_0, score_1 = torch.split(score, 1, dim=0)
            coord_0, coord_1 = torch.split(coord, 1, dim=0)
            desc_0,  desc_1  = torch.split(desc,  1, dim=0)
        
        B, C, Hc, Wc = desc_0.shape

        assert(B == 1)

        # Get CPU NumPy arrays. 
        coord_0 = coord_0.view(2, -1).t().cpu().numpy()
        coord_1 = coord_1.view(2, -1).t().cpu().numpy()

        score_0 = score_0.view(-1,).cpu().numpy()
        score_1 = score_1.view(-1,).cpu().numpy()

        desc_0 = desc_0.view(C, Hc, Wc).view(C, -1).t().cpu().numpy()
        desc_1 = desc_1.view(C, Hc, Wc).view(C, -1).t().cpu().numpy()
        
        # Filter based on confidence threshold
        if self.conf_threshold > 0:
            mask_thresh_0 = score_0 > self.conf_threshold
            mask_thresh_1 = score_1 > self.conf_threshold
            score_0 = score_0[mask_thresh_0]
            score_1 = score_1[mask_thresh_1]
            coord_0 = coord_0[mask_thresh_0, :]
            coord_1 = coord_1[mask_thresh_1, :]
            desc_0  = desc_0[mask_thresh_0, :]
            desc_1  = desc_1[mask_thresh_1, :]
        
        # Select k best key points and descriptors.
        score_0, coord_0, desc_0 = get_k_best( score_0, coord_0, desc_0, self.n_matches )
        score_1, coord_1, desc_1 = get_k_best( score_1, coord_1, desc_1, self.n_matches )

        # The outputs.
        outputs = {
            'coord_0': coord_0, 'desc_0': desc_0, 'score_0': score_0,
            'coord_1': coord_1, 'desc_1': desc_1, 'score_1': score_1,
        }

        return outputs


# Original file is created Magic Leap 

import cv2
import numpy as np

import torch
torch.set_grad_enabled(False)

from third_party.SuperGlue.models.matching import Matching
from third_party.SuperGlue.models.utils import (
    AverageTimer, VideoStreamer,
    make_matching_plot_fast, frame2tensor)

from proxies import Proxy
from proxies.utils.image_utils import resize_by_longer_edge
from proxies.register import ( PROXIES, register )

@register(PROXIES)
class ProxySuperGlue(Proxy):
    @classmethod
    def get_default_init_args(cls):
        return dict(
            type=cls.__name__,
            superglue='indoor',
            keypoint_threshold=0.005,
            max_keypoints=-1,
            nms_radius=4,
            sinkhorn_iterations=20,
            match_threshold=0.2,
            longer_edge=1024,
        )

    def __init__(self, 
        superglue,
        keypoint_threshold=0.005,
        max_keypoints=-1,
        nms_radius=4,
        sinkhorn_iterations=20,
        match_threshold=0.2,
        longer_edge=1024):
        '''
        superglue (string): The type of the SuperGlue model, should be "indoor" or "outdoor".
        keypoint_threshold (float): SuperPoint keypoint detector confidence threshold.
        max_keypoints (int): Maximum number of keypoints detected by Superpoint.
        nms_radius (int): uperPoint Non Maximum Suppression (NMS) radius.
        sinkhorn_iterations (int): Number of Sinkhorn iterations performed by SuperGlue
        match_threshold (float): SuperGlue match threshold
        longer_edge (int): The input image will be reshaped to have its longer edge equal this value. 
            Set zero to disable.
        '''
        super(ProxySuperGlue, self).__init__(task_type='matching')

        self.superglue           = superglue
        self.keypoint_threshold  = keypoint_threshold
        self.max_keypoints       = max_keypoints
        self.nms_radius          = nms_radius
        self.sinkhorn_iterations = sinkhorn_iterations
        self.match_threshold     = match_threshold
        self.longer_edge         = longer_edge

        self.device = None # Used by the SuperGlue model.

        self.ori_shape = None # The original shape of the input image.
        self.new_shape = None # The shape of the images after re-scale, if there are any.

    def initialize(self):
        super().initialize()

        print('Initializing SuperGlue...')

        self.device = 'cuda' # Force using CUDA

        print('Running inference on device \"{}\"'.format(self.device))
        config = {
            'superpoint': {
                'nms_radius': self.nms_radius,
                'keypoint_threshold': self.keypoint_threshold,
                'max_keypoints': self.max_keypoints
            },
            'superglue': {
                'weights': self.superglue,
                'sinkhorn_iterations': self.sinkhorn_iterations,
                'match_threshold': self.match_threshold,
            }
        }
        matching = Matching(config).eval().to(self.device)

        # Save the model.
        self.model = matching

        print('SuperGlue initialized. ')

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

        t0 = frame2tensor(img_0, self.device)
        t1 = frame2tensor(img_1, self.device)

        pred = self.model({'image0': t0, 'image1': t1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1  = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid  = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf  = conf[valid]

        # The outputs.
        outputs = {
            'coord_0': mkpts0, 
            'coord_1': mkpts1,
            'confidence': mconf,
        }

        return outputs

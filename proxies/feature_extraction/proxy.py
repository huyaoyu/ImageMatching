
# Original file is created Magic Leap 

import cv2
import numpy as np

from feature_extraction.feature_point_ocv import FeaturePointOCV

from proxies import Proxy
from proxies.utils.image_utils import resize_by_longer_edge
from proxies.register import ( PROXIES, register )

@register(PROXIES)
class ProxyFeatExtOCV(Proxy):
    @classmethod
    def get_default_init_args(cls):
        return dict(
            type=cls.__name__,
            feature_type='surf',
            longer_edge=1024,
        )

    def __init__(self, 
        feature_type='surf',
        longer_edge=1024,
        **kwargs):
        '''
        superglue (string): The type of feature.
        '''
        super(ProxyFeatExtOCV, self).__init__(task_type='extraction')

        self.feature_type = feature_type
        self.init_kwargs  = kwargs

        self.longer_edge = longer_edge
        self.ori_shape = None # The original shape of the input image.
        self.new_shape = None # The shape of the images after re-scale, if there are any.

    def initialize(self):
        super().initialize()

        print('Initializing FeatureExtractionOCV...')

        # Save the model.
        self.model = FeaturePointOCV(self.feature_type, **self.init_kwargs)

        print('FeatureExtractionOCV initialized. ')

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

        # Compute.
        results = self.model( ( img_0, img_1 ) )

        # Convert the keypoints into numpy arrays.
        coord_0 = np.array(
            [ kp.pt for kp in results[0][0] ],
            dtype=np.float32 
            )

        coord_1 = np.array(
            [ kp.pt for kp in results[1][0] ],
            dtype=np.float32 
            )

        # The outputs.
        outputs = {
            'coord_0': coord_0, 'desc_0': results[0][1], 
            'coord_1': coord_1, 'desc_1': results[1][1],
        }

        return outputs

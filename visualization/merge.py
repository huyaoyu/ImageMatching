
import cv2
import numpy as np

def merge_two_single_channels(ref, tst):
    # Convert ref to a 3-channel grayscale image.
    ref = np.tile( np.expand_dims(ref, axis=-1), (1, 1, 3) )

    # Convert tst to a green image.
    g = np.zeros_like( ref )
    g[:, :, 1] = tst

    # Merge. 
    return cv2.addWeighted( ref, 0.7, g, 0.3, 0 )

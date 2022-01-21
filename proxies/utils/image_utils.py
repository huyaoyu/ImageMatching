
import cv2
import numpy as np

def find_nearest_next(value, base):
    assert( value >= base ), f'value ({value}) musbt be larger than or equal to base ({base}). '
    return np.ceil( value / base ) * base

def resize_by_longer_edge(img, target_longer_edge, base=16):
    '''Resize img to a new shape such that its longer edge has a size 
    specified by target_longer_edge.

    Arguments:
    img (Array): The input image.
    target_longer_edge (int): The size of the longer edge.
    base (int): The base uint of size.
    
    Returns:
    The resized img.
    '''
    assert( target_longer_edge % base == 0 ), \
        f'Assuming target_longer_edge {target_longer_edge} is a multiple of {base}. '

    # Figure out the longer edge.
    H, W = img.shape[:2]

    if ( H >= W ):
        newH = target_longer_edge
        newW = int(find_nearest_next(round(1.0 * newH / H * W), base))
    else:
        newW = target_longer_edge
        newH = int(find_nearest_next(round(1.0 * newW / W * H), base))

    return cv2.resize(img, (newW, newH), interpolation=cv2.INTER_CUBIC)


import numpy as np

def scale_homography_matrix( hMat, curDstShape, oriDstShape, curSrcShape, oriSrcShape ):
    # Two scale factors for the souce image.
    fx = curSrcShape[1] / oriSrcShape[1]
    fy = curSrcShape[0] / oriSrcShape[0]

    # The scale matrix.
    fSrc = np.eye( 3, dtype=np.float32 )
    fSrc[0, 0] = fx
    fSrc[1, 1] = fy

    # Two scale factors for the destination image.
    fx = curDstShape[1] / oriDstShape[1]
    fy = curDstShape[0] / oriDstShape[0]

    # The inverse scale matix
    fDstInv = np.eye( 3, dtype=np.float32 )
    fDstInv[0, 0] = 1. / fx
    fDstInv[1, 1] = 1. / fy

    return fDstInv @ hMat @ fSrc

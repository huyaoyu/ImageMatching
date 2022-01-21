
import cv2
import numpy as np
# import time

class HomographyCalculator(object):
    def __init__(self):
        super(HomographyCalculator, self).__init__()
    
        self.matcher  = cv2.BFMatcher(cv2.NORM_L2)

        self.h_mat = None
        self.good_matches = None
        self.good_kp_src = None
        self.good_kp_dst = None
    
    def compute_homography_by_matched_results(self, srcKP, dstKP):
        H, mask = cv2.findHomography( srcKP, dstKP, cv2.RANSAC, 3.0 )
        self.h_mat = H
        return H, mask

    def __call__(self, kpSrc, kpDst, dsSrc, dsDst):

        # timeStart = time.time()

        # Matching.
        matches = self.matcher.knnMatch( dsSrc, dsDst, k=2 )
        goodMatches = [ m for m, n in matches if m.distance < 0.7 * n.distance ]
        
        # timeDetectionAndMatching = time.time()
        
        nGoodMatches = len(goodMatches)
        print('len(goodMatches) = %d' % (nGoodMatches))
        if ( nGoodMatches < 4 ):
            print('Abort homography computation. ')
            return None, None

        # Homography.
        srcMatchedIdx = [ m.queryIdx for m in goodMatches ]
        dstMatchedIdx = [ m.trainIdx for m in goodMatches ]

        srcKP = kpSrc[srcMatchedIdx, :].reshape( (-1, 1, 2) )
        dstKP = kpDst[dstMatchedIdx, :].reshape( (-1, 1, 2) )

        # srcKP = np.array( [ kpDst[ m.queryIdx ] for m in goodMatches ], dtype=np.float32 ).reshape( (-1, 1, 2) )
        # dstKP = np.array( [ kpSrc[ m.trainIdx ] for m in goodMatches ], dtype=np.float32 ).reshape( (-1, 1, 2) )
        
        H, mask = self.compute_homography_by_matched_results( srcKP, dstKP )

        # timeHomography = time.time()
        
        # print( f'Feature extraction and matching: {timeDetectionAndMatching - timeStart}' )
        # print( f'Homography: {timeHomography - timeDetectionAndMatching}' )
        
        self.good_matches = goodMatches
        self.good_kp_src = srcKP
        self.good_kp_dst = dstKP

        return H, goodMatches

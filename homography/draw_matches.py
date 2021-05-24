
import cv2
import numpy as np

def draw_matches(imgSrc, imgDst, hMat, kpSrc, kpDst, goodMatches ):
    imgMatches = np.empty(
            (max(imgDst.shape[0], imgSrc.shape[0]), imgDst.shape[1]+imgSrc.shape[1], 3), 
            dtype=np.uint8)

    ocvKPDst = [ cv2.KeyPoint( c[0], c[1], 1 ) for c in kpDst ]
    ocvKPSrc = [ cv2.KeyPoint( c[0], c[1], 1 ) for c in kpSrc ]

    cv2.drawMatches(
        imgSrc, ocvKPSrc, 
        imgDst, ocvKPDst, 
        goodMatches, 
        imgMatches, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return imgMatches

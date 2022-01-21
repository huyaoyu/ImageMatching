
import cv2

class FeaturePointOCV(object):
    def __init__(self, feature_type='surf', **kwargs):
        super(FeaturePointOCV, self).__init__()

        if ( feature_type == 'surf' ):
            self.detector = cv2.xfeatures2d_SURF.create(
                hessianThreshold=kwargs['hessian'])
        else:
            raise Exception(f'{feature_type} is not supported. ')

    def __call__(self, imgs):
        results = []
        for img in imgs:
            kp, desc = self.detector.detectAndCompute(img, mask=None)
            results.append( [ kp, desc ] )
        return results

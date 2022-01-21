
import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

def draw_matches_ocv(imgSrc, imgDst, kpSrc, kpDst, goodMatches ):
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

# Copied from SuperGlue.
def plot_image_pair(imgs, dpi=300, size=12, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)

# Copied from SuperGlue.
def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)

# Copied from SuperGlue.
def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
                 for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)

def draw_matches_plt(imgSrc, imgDst, kpSrc, kpDst, confidence):
    plot_image_pair((imgDst, imgSrc))
    plot_keypoints(kpDst, kpSrc, color='k', ps=4)
    plot_keypoints(kpDst, kpSrc, color='w', ps=2)
    plot_matches(kpDst, kpSrc, cm.jet(confidence))

    # convert canvas to image
    fig = plt.gcf()
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
            sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # RGB to BGR.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img
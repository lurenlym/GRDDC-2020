import os
import cv2
import numpy as np
CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

V = [1, 0.7, 0.7, 0.7,
               0.7, 0.7, 0.7, 0.7,
              0.7, 0.7,0.7, 0.7,
              0.7, 0.7, 0.7, 0.7,
               0.7, 0.7, 0.7]

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    # label read

    labelMat=[]
    for i,line in enumerate(PALETTE):    
        labelMat.append(line)
    # BGR转RGB
    labelMat = np.array(labelMat,dtype="uint8")[:,[2,1,0]]
    return labelMat
def encode_segmap(mask,labels):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(labels):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = V[ii]
        # break
    label_mask = label_mask.astype(int)
    return label_mask


if __name__ == "__main__":
    root = '/lym/DATASET/GRDDC/test1/test'
    segroot = '/lym/mmsegmentation/infer_output/deeplabv3plus_r101-d8_512x1024_40k_cityscapes/test1grddc'
    segmask = ''
    imglist = os.listdir(root)
    label = get_pascal_labels()
    saveroot = '/lym/DATASET/GRDDC/test1/test4channel'
    if os.path.exists(saveroot) is False:
        os.makedirs(saveroot)
    for imgname in imglist:
        imgfullname = os.path.join(root,imgname)
        rawimg = cv2.imread(imgfullname)
        segfullname = os.path.join(segroot,imgname[:-4]+'.png')
        segimg = cv2.imread(segfullname)
        if segimg is None:
            continue
        label_mask = encode_segmap(segimg,label)
        newimg=np.zeros((rawimg.shape[0], rawimg.shape[1],3), dtype=np.uint8)
        newimg[:,:-100,0]=rawimg[:,:-100,0]*label_mask[:,:-100]
        newimg[:,:-100,1]=rawimg[:,:-100,1]*label_mask[:,:-100]
        newimg[:,:-100,2]=rawimg[:,:-100,2]*label_mask[:,:-100]
        newimg[:,-100:,:]=rawimg[:,-100:,:]
        # newimg[:,:,3]=label_mask
        cv2.imwrite(os.path.join(saveroot,imgname[:-4]+'.png'),newimg)
        print(imgname)





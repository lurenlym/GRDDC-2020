import cv2
import numpy as np
import os

res1 = {}
res2 = {}

file1 = '/lym/PaddleDetection/GRDDC_test2_result_faster.txt'
file2 = '/lym/DATASET/GRDDC/test2/GRDDC_test2_result_yolo_4cha.txt'
emsemblefile = '/lym/PaddleDetection/GRDDC_test2_result_emsenble.txt'
def readres(file):
    res ={}
    with open(file,'r') as F:
        for line in F:
            imgres = [[],[],[],[]]
            img_name,resline=line.strip().split(',')
            if len(resline)==1:
                res[img_name] = imgres
            else:
                resdata = resline.split(' ')
                rescount = len(resdata)//6
                for i in range(rescount):
                    label = int(resdata[i*6+0])
                    left,top,right,bottom,score =map(float,(resdata[i*6+1],resdata[i*6+2],resdata[i*6+3],resdata[i*6+4],resdata[i*6+5]))
                    imgres[label-1].append([left,top,right,bottom,score])
                res[img_name] = imgres
    return res
                  
def py_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    #每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order是按照score降序排序的
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        #计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        #找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep


def vote(f1,f2,iouthresh):
    emsenble_bbox ={}
    
    for img_name in f1.keys():
        img_bbox=[[],[],[],[]]
        for i in range(4):
            bboxs1 = f1[img_name][i]
            bboxs2 = f2[img_name][i]
            if bboxs1 == [] or bboxs2 == []:
                img_bbox[i]=[]
            elif bboxs1 == []:
                img_bbox[i]=bboxs2
            elif bboxs2 == []:
                img_bbox[i]=bboxs1
            else:
                bboxs = np.vstack([bboxs1,bboxs2])
                newindex = py_nms(bboxs,iouthresh)
                newbboxs = bboxs[newindex]
                img_bbox[i]=newbboxs
        emsenble_bbox[img_name]=img_bbox
    return emsenble_bbox
def writeres(bboxdic,writefile):
    with open(writefile,'w+') as fresult:
        for key in bboxdic.keys():
            fresult.write(key)
            fresult.write(',')
            for i in range(4):
                for j, data in enumerate(bboxdic[key][i]):
                    left, top, right, bottom = map(int,(data[0],data[1],data[2],data[3]))
                    label = i+1
                    bbox_mess = ' '.join([str(label), str(left), str(top), str(right), str(bottom),''])
                    fresult.write(bbox_mess)
            fresult.write('\n')
        
if __name__ == "__main__":
    file1res = readres(file1)
    file2res = readres(file2)
    votebbox = vote(file1res,file2res,1)
    writeres(votebbox,emsemblefile)
    print(votebbox)
                
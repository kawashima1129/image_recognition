import numpy as np
import os
import glob
import pandas as pd

#precision, recall計算
def calc_precision_recall(tp, tp_fp, tp_fn):
    if tp_fp == 0:
        precision = 0
    else:
        precision = tp /  tp_fp
    
    if tp_fn == 0:
        recall = 0
    else:
        recall = tp / tp_fn
    
    return precision, recall

#推定結果とground truthのマッチング
def adjust(iou_list, result):

    dfd = pd.DataFrame(iou_list)
    dfd_array = dfd.values
    
    while dfd_array.shape[0] != 0:
        iou_max_index = dfd.idxmax()[2]
        tmp = dfd.ix[iou_max_index].values.tolist()
    
        flag = False
        if len(result) > 0:
            for r in result:
                if tmp[0] == r[0] or tmp[1] == r[1]:
                    flag = True
        if not flag:
            result.append(tmp)
        dfd = dfd.drop(iou_max_index)
        dfd_array = dfd.values
            
    return result


#IOU計算
def calc_iou(r1, r2):
    xmin1, ymin1, xmax1, ymax1 = r1
    xmin2, ymin2, xmax2, ymax2 = r2
    
    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
    
    and_x1, and_y1 = max(xmin1, xmin2), max(ymin1, ymin2)
    and_x2, and_y2 = min(xmax1, xmax2), min(ymax1, ymax2)
    
    and_w = and_x2 - and_x1 + 1
    and_h = and_y2 - and_y1 + 1
    
    if and_w <= 0 or and_h <= 0:
        return 0
    
    and_area = and_w * and_h
    or_area = area1 + area2 - and_area
    
    return and_area / or_area


def main():
    pred_path = "./person/"
    gt_path = "./correct_data/"
    preds = os.listdir(pred_path)
    class_num = 1
    classes = [i for i in range(class_num)]

    for _class in classes:
        tp_list = []
        tp_fp_list = []
        tp_fn_list = []
        
        tp = 0
        tp_fp = 0
        tp_fn = 0
        for pred in preds:
            p = pred_path + pred
            t = gt_path + pred
            dfp = pd.read_csv(p, names = ['c', 'xmin', 'ymin', 'xmax', 'ymax'])
            dft = pd.read_csv(t, names = ['c', 'xmin', 'ymin', 'xmax', 'ymax'])
            
            pred_array = dfp[(dfp.c == _class)][['xmin', 'ymin', 'xmax', 'ymax']].values
            true_array = dft[(dft.c == _class)][['xmin', 'ymin', 'xmax', 'ymax']].values
            
            pred_num = pred_array.shape[0]#画像１枚あたりの推定矩形の数
            true_num = true_array.shape[0]#画像１枚あたりのground truthの数
            
            if pred_num != 0:
                iou_list = []
                result = []
                for i in range(pred_num):
                    for j in range(true_num):
                        iou = calc_iou(pred_array[i], true_array[j])
                        if iou >= 0.5:  
                            iou_list.append([i, j, iou])
                result = adjust(iou_list, result)

            tp_list.append(len(result))
            tp_fp_list.append(pred_num)
            tp_fn_list.append(true_num)
            if len(result) != true_num or pred_num != true_num:
                print("誤検出もしくは検出漏れをしたファイル名：", pred)
             
        tp = sum(tp_list)
        tp_fp = sum(tp_fp_list)
        tp_fn = sum(tp_fn_list)
        p, r = calc_precision_recall(tp, tp_fp, tp_fn)
        #print(tp)        
        print("class = {}, precision = {}, recall = {}".format(_class, p, r))

            
    
if __name__ == "__main__":
    main()


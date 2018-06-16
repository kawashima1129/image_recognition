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

#複数の推定結果がある一つのground truthに対してIOU > 0.5となった場合に
#IOUが大きい組み合わせのみを抽出
def method2(iou_list):
    if len(iou_list) > 1: 
        
        dfd = pd.DataFrame(iou_list)
        dfd_tmp = dfd[dfd.duplicated(subset=1, keep = False)]#ground truthが被ったところ抽出
        if len(dfd_tmp) == 0:#被っているところがなかったらそのまま元のlistを返す
            return dfd.values.tolist()
        
        iou_max_index = dfd_tmp.idxmax()[2]#列ごとに最大値をとった行をとってくる かつ　IOUの列をとってくる
        index = set(dfd.index) -  (set(dfd_tmp.index) - set(  list([ iou_max_index ])) )   

        return dfd.ix[index,:].values.tolist()
    
    return iou_list

#推定結果が複数のground truthに対してIOU > 0.5になった場合に
#IOUが大きい組み合わせのみ抽出する処理
def method1(iou_list):
    #IOU最大値の行だけ取得
    if len(iou_list) > 1:
        dfd = pd.DataFrame(iou_list)
        iou_max_index = dfd.idxmax()[2]#IOUが最大だったindex抽出
        tmp = dfd.iloc[iou_max_index].values.tolist()#IOUが最大だったレコード抽出
        return tmp
    
    return iou_list

#IOU計算
def calc_iou(r1, r2):
    xmin1, ymin1, xmax1, ymax1 = r1
    xmin2, ymin2, xmax2, ymax2 = r2
    
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    
    and_x1, and_y1 = max(xmin1, xmin2), max(ymin1, ymin2)
    and_x2, and_y2 = min(xmax1, xmax2), min(ymax1, ymax2)
    
    and_w = and_x2 - and_x1
    and_h = and_y2 - and_y1
    
    if and_w <= 0 or and_h <= 0:
        return 0
    
    and_area = and_w * and_h
    or_area = area1 + area2 - and_area
    
    return and_area / or_area


def main():
    pred_path = "./box/"
    gt_path = "./tmp/"
    preds = os.listdir(pred_path)
    class_num = 3
    classes = [i for i in range(class_num)]

    for _class in classes:
        result_list = []
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
            
            hozon = []
            for i in range(pred_num):
                iou_list = []
                for j in range(true_num):
                    iou = calc_iou(pred_array[i], true_array[j])
                    if iou >= 0.5:  
                        iou_list.append([i, j, iou])

                iou_list = method1(iou_list)
                print(iou_list)
                for i in range(len(iou_list)):
                    hozon.append(iou_list[i])

            result_list = method2(hozon)
            
            tp_list.append(len(result_list))
            tp_fp_list.append(tp_fp + pred_num)
            tp_fn_list.append(tp_fn + true_num)
             
        tp = sum(tp_list)
        tp_fp = sum(tp_fp_list)
        tp_fn = sum(tp_fn_list)

        p, r = calc_precision_recall(tp, tp_fp, tp_fn)
        
        print("class = {}, precision = {}, recall = {}".format(_class, p, r))

            
    
if __name__ == "__main__":
    main()
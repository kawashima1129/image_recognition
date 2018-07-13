# image_recognition
## precision_recall.pyの説明
* 物体検出における各クラスのprecisonとrecallを算出するプログラム  
  VOCの算出方法に準拠しています。  
  テスト済み。
使い方  
pred_pathに推定したバウンディングボックスの座標ファイル
gt_pathにground truthの座標ファイルを保存したディレクトリのパスを記載する  
csvファイル形式  
========================  
claass,xmin,ymin,xmax,ymax  
例  
0,12,25,25,48  
1,10,16,37,89  
    .  
    .  
=======================  
class_numにはクラス数  
暇だったら使いやすくする、mAP算出までやる予定  


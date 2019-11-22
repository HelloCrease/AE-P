# AE-P
这是北京pm2.5数据集在autoencoder模型下的结果，主要做的是预测下一个小时的天气污染情况，输入为当前小时的8个天气特征和污染情况，输出为下一小时的天气污染情况.
实验的模型和下图比较类似，采用三层LSTM作为Encoder和Decoder,一个RepeatVector合成Encoder feature
![image](https://github.com/HelloCrease/AE-P/blob/master/image/3.png)
实验结果如下图，打印出了测试数据集中前300个小时的预测值和真实值的对比，平均RMSE大概在0.020左右:
![image](https://github.com/HelloCrease/AE-P/blob/master/image/1.png)
train loss和val_loss如下图
![image](https://github.com/HelloCrease/AE-P/blob/master/image/2.png)



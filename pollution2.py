from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras import optimizers


# 转换序列成监督学习问题
# 该函数有四个参数：数据、输入滞后步数，输出移动步数，是否删除具有NAN值的行
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# 加载数据集
dataset = read_csv('pollution2.csv', header=0, index_col=0)
values = dataset.values
# 整数编码,对风向特征整数标签化
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# 归一化特征
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 构建监督学习问题
reframed = series_to_supervised(scaled, 1, 1)
# 丢弃我们并不想预测的列
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
# 打印出前五行已转换的数据集，可以看到８个输入变量，这是前一小时天气情况和污染情况，还有一个输出变量是当前小时的污染情况
print(reframed.head())

# 分割为训练集和测试集
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# 分为输入输出
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# 重塑成3D形状 [样例, 时间步, 特征]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# 设计网络
model = Sequential()
# 第一个隐藏层有64个神经元，输出层中有１个神经元用于预测污染情况，输入变量为当前小时里的8个天气和污染指数
# Encoder
model.add(LSTM(64, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(LSTM(32,activation='relu',return_sequences=True))
model.add(LSTM(16,activation='relu',return_sequences=False))
model.add(RepeatVector(1))
# Decoder
model.add(LSTM(16,activation='relu',return_sequences=True))
model.add(LSTM(32,activation='relu',return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
model.summary()
# 拟合神经网络模型
history = model.fit(train_X, train_y, epochs=150, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                   shuffle=False)
y = model.predict(test_X)
# 取消时间步
test_X = test_X.reshape((test_X.shape[0],test_X.shape[2]))
pyplot.plot(y[:300],label='forecast')
pyplot.plot(test_y[1:301], label='true')
pyplot.legend()
pyplot.show()

rmse = sqrt(mean_squared_error(y[:100], test_y[:100]))
print('Test RMSE: %.3f' % rmse)




# 绘制loss和验证集loss
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()

#

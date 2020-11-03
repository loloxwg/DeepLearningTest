import numpy
import pandas as pd
import math
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


###把原始数据变为输入输出的数据矩阵
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])

    return numpy.array(dataX), numpy.array(dataY)


###预测未来n期的数据
def prehead(model, num, data1, look_back, scaler):
    x1 = data1[len(data1) - look_back:, 0]
    x1 = numpy.reshape(x1, (len(x1), 1))
    x1 = scaler.fit_transform(x1)
    x1 = numpy.reshape(x1, (1, 1, look_back))

    preh = []
    for i in range(num):
        y1 = model.predict(x1)
        y2 = scaler.inverse_transform(y1)
        preh.append(y2[0][0])
        x1[:, :, :look_back - 1] = x1[:, :, 1:]
        x1[:, :, look_back - 1] = y1[0]

    return preh

###可视化
def plotfun(dataset,trainPredict,look_back,testPredict,predicthead):
    ###训练集画图所需数据
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    ###测试集画图所需数据
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict

    preh = prehead(model, predicthead, data1, look_back, scaler)
    ###预测期数的对应x轴数据
    xx = list(range(len(data1)+1,len(data1)+predicthead+1))

    ###可视化
    plt.plot(data1)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.plot(xx,preh)
    plt.show()


###主函数
if __name__ == '__main__':
    ###固定随机数种子
    numpy.random.seed(7)

    ###读取数据
    dataframe = pd.read_csv('D:/airline-passengers.csv', usecols=[1], engine='python')
    dataset = dataframe.values
    ###把数据转为float型
    dataset = dataset.astype('float32')
    data1 = dataset

    ###标准化数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    ###把数据集拆分为训练集和测试集 0.7为训练集的比例
    train_size = int(len(dataset) * 0.431)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    ###将训练集和测试集转为输入输出格式
    ###look_back意思就是用前12个数据来预测下一个
    look_back = 12
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    ###把输入数据转为LSTM的输入维度 [samples, time steps, features]
    ###samples表示样本个数，time steps表示每个样本的行数，features表示每个样本列数
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, look_back))
    testX = numpy.reshape(testX, (testX.shape[0], 1, look_back))

    ###创建LSTM网络模型
    model = Sequential()
    model.add(LSTM(12, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=530, batch_size=1, verbose=0.1)

    ###对训练集和测试集进行预测
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    ###把预测值还原（前面对原始数据进行了标准化）
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    ###计算训练集和测试集的均方误差
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    ###可视化
    predicthead = 36
    plotfun(dataset, trainPredict, look_back, testPredict, predicthead)
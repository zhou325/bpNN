import numpy as np

# 读取数据
train_X = np.loadtxt("train_X.txt")
train_t = np.loadtxt("train_y_onehot.txt")
test_X = np.loadtxt("test_X.txt")
test_t = np.loadtxt("test_y_onehot.txt")

def train_bpNN(trainX, trainTarget, learning_rate = 0.5):
    """ 用逻辑函数作为激励函数编写bpNN模型
    """
    # 初始化网络权值
    input_number = trainX.shape[1]
    W1 = 2*np.random.random((input_number,100)) - 1
    W2 = 2*np.random.random((100,50)) - 1
    W3 = 2*np.random.random((50,10)) - 1
    # 迭代20次
    for i in range(20):
        # 对每一个样本进行计算
        for row_num in range(trainX.shape[0]):
            sample_X = trainX[row_num,]
            sample_t = trainTarget[row_num,]
            # 正向计算输出结果
            hidden1 = 1/(1+np.exp(-sample_X.dot(W1)))
            hidden2 = 1/(1+np.exp(-hidden1.dot(W2)))
            output = 1/(1+np.exp(-hidden2.dot(W3)))
            # 反向计算误差
            # 1. j在输出层
            delta_output = (output - sample_t) * output * (1 - output)
            # 更新输出层权重
            W3 = W3 - learning_rate * np.asarray((np.asmatrix(hidden2).T)*delta_output)
            # 2. j在hidden2
            delta_hidden2 = (W3.dot(delta_output.T)) * hidden2 * (1 - hidden2)
            # 更新hidden2权重
            W2 = W2 - learning_rate * np.asarray((np.asmatrix(hidden1).T)*delta_hidden2)
            # 3. j在hidden1
            delta_hidden1 = (W2.dot(delta_hidden2.T)) * hidden1 * (1 - hidden1)
            # 更新hidden1权重
            W1 = W1 - learning_rate * np.asarray((np.asmatrix(sample_X).T)*delta_hidden1)
    W = np.array([W1,W2,W3])
    # 计算最终输出结果
    hidden1 = 1/(1+np.exp(-trainX.dot(W[0,])))
    hidden2 = 1/(1+np.exp(-hidden1.dot(W[1,])))
    output = 1/(1+np.exp(-hidden2.dot(W[2,])))
    # 对于一个样本，认为评分最大的那个分类就是它的预测分类，由此计算错误分类率
    # 获取输出层分类结果
    output_class = output.argmax(axis = 1)
    # 获取目标数据分类结果
    target_class = trainTarget.argmax(axis = 1)
    # 计算错误分类率
    err = 0
    for i in range(output.shape[0]):
        if output_class[i] != target_class[i]:
            err += 1
    error_rate = err/output.shape[0]
    print("Under log-sigmoid function, the training error rate for training set is ", error_rate*100, "%.")
    return W

def test_bpNN(testX, testTarget, Weights):
    hidden1 = 1/(1+np.exp(-testX.dot(Weights[0,])))
    hidden2 = 1/(1+np.exp(-hidden1.dot(Weights[1,])))
    output = 1/(1+np.exp(-hidden2.dot(Weights[2,])))
    # 计算错误分类率
    output_class = output.argmax(axis = 1)
    target_class = testTarget.argmax(axis = 1)
    err = 0
    for i in range(output.shape[0]):
        if output_class[i] != target_class[i]:
            err += 1
    error_rate = err/output.shape[0]
    print("Under log-sigmoid function, the training error rate for testing set is ", error_rate*100, "%.")
    return

def train_bpNN_Tan(trainX, trainTarget, learning_rate = 0.5):
    """ 用tan函数作为激励函数编写bpNN模型
    """
    # 初始化网络权值
    input_number = trainX.shape[1]
    W1 = 2*np.random.random((input_number,100)) - 1
    W2 = 2*np.random.random((100,50)) - 1
    W3 = 2*np.random.random((50,10)) - 1
    # 迭代20次
    for i in range(20):
        # 对每一个样本进行计算
        for row_num in range(trainX.shape[0]):
            sample_X = trainX[row_num,]
            sample_t = trainTarget[row_num,]
            # 正向计算输出结果
            hidden1 = 2/(1+np.exp(-2*sample_X.dot(W1)))-1
            hidden2 = 2/(1+np.exp(-2*hidden1.dot(W2)))-1
            output = 2/(1+np.exp(-2*hidden2.dot(W3)))-1
            # 反向计算误差
            # 1. j在输出层
            delta_output = (output - sample_t) * (1 - output**2)
            # 更新输出层权重
            W3 = W3 - learning_rate * np.asarray((np.asmatrix(hidden2).T)*delta_output)
            # 2. j在hidden2
            delta_hidden2 = (W3.dot(delta_output.T)) * (1 - hidden2**2)
            # 更新hidden2权重
            W2 = W2 - learning_rate * np.asarray((np.asmatrix(hidden1).T)*delta_hidden2)
            # 3. j在hidden1
            delta_hidden1 = (W2.dot(delta_hidden2.T)) * (1 - hidden1**2)
            # 更新hidden1权重
            W1 = W1 - learning_rate * np.asarray((np.asmatrix(sample_X).T)*delta_hidden1)
    W = np.array([W1,W2,W3])
    # 计算最终输出结果
    hidden1 = 2/(1+np.exp(-2*trainX.dot(W[0,])))-1
    hidden2 = 2/(1+np.exp(-2*hidden1.dot(W[1,])))-1
    output = 2/(1+np.exp(-2*hidden2.dot(W[2,])))-1
    # 对于一个样本，认为评分最大的那个分类就是它的预测分类，由此计算错误分类率
    # 获取输出层分类结果
    output_class = output.argmax(axis = 1)
    # 获取目标数据分类结果
    target_class = trainTarget.argmax(axis = 1)
    # 计算错误分类率
    err = 0
    for i in range(output.shape[0]):
        if output_class[i] != target_class[i]:
            err += 1
    error_rate = err/output.shape[0]
    print("Under log-sigmoid function, the training error rate for training set is ", error_rate*100, "%.")
    return W

def test_bpNN_Tan(testX, testTarget, Weights):
    hidden1 = 2/(1+np.exp(-2*testX.dot(Weights[0,])))-1
    hidden2 = 2/(1+np.exp(-2*hidden1.dot(Weights[1,])))-1
    output = 2/(1+np.exp(-2*hidden2.dot(Weights[2,])))-1
    # 计算错误分类率
    output_class = output.argmax(axis = 1)
    target_class = testTarget.argmax(axis = 1)
    err = 0
    for i in range(output.shape[0]):
        if output_class[i] != target_class[i]:
            err += 1
    error_rate = err/output.shape[0]
    print("Under log-sigmoid function, the training error rate for training set is ", error_rate*100, "%.")
    return

def main():
    W1 = train_bpNN(train_X, train_t)
    test_bpNN(test_X, test_t, W1)
    W2 = train_bpNN_Tan(train_X, train_t)
    test_bpNN_Tan(test_X, test_t, W2)
    return

if __name__ == "__main__":
    main()

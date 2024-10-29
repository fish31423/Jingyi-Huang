#这是用 Python 语言实现的深度学习代码，具体是使用了 Keras 库构建神经网络模型
#在这段代码中，定义了一个顺序模型（Sequential），依次添加了卷积层（Conv2D）、激活函数层（Activation）、最大池化层（MaxPooling2D）、展平层（Flatten）和全连接层（Dense）等
#运行这个模型进行训练和预测，还需要准备数据、定义损失函数、优化器等，并进行训练循环
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense

# 定义模型
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(120,120,3)))
model.add(Activation('relu'))
model.add(Conv2D(32,(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

# 打印模型结构
model.summary()

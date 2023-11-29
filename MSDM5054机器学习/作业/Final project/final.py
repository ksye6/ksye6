import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math

#(3)
#1
def getEuclidean(point1, point2):
    dimension = len(point1)
    dist = 0.0
    for i in range(dimension):
        dist += (point1[i] - point2[i]) ** 2
    return math.sqrt(dist)

def k_means(df, k, iteration):
    #初始化簇心向量
    index = random.sample(list(range(len(df))), k)
    vectors = []
    for i in index:
        vectors.append(list(df.loc[i,].values))
    
    #初始化类别
    labels = []
    for i in range(len(df)):
        labels.append(-1)
    
    while(iteration > 0):
        #初始化簇
        C = []
        for i in range(k):
            C.append([])
        for labelIndex, item in enumerate(df.to_numpy()):
            classIndex = -1
            minDist = 1e6
            for i, point in enumerate(vectors):
                dist = getEuclidean(item, point)
                if(dist < minDist):
                    classIndex = i
                    minDist = dist
            C[classIndex].append(item)
            labels[labelIndex] = classIndex
        
        for i, cluster in enumerate(C):
            clusterHeart = []
            dimension = df.shape[1]
            for j in range(dimension):
                clusterHeart.append(0)
            for item in cluster:
                for j, coordinate in enumerate(item):
                    clusterHeart[j] += coordinate / len(cluster)
            vectors[i] = clusterHeart
        
        iteration -= 1
    return C, labels

#2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import time

df0 = pd.read_csv("C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5054机器学习\\作业\\Final project\\group_data.csv")
df = df0.iloc[:,0:100]

scaler = StandardScaler()

# 对数据进行标准化
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

start_time = time.time()

# 创建 PCA 模型并进行主成分分析
pca = PCA(n_components=4)
principal_components = pca.fit_transform(scaled_df)

# 将主成分数据转换为数据框
pc_df = pd.DataFrame(data=principal_components, columns=['PC1','PC2','PC3','PC4'])

random.seed(123)
C, labels = k_means(pc_df, 4, 20)

pc_df.loc[:,'grouptype']=labels

df0['grouptype'] = df0['grouptype'].replace('comp.*', 2)
df0['grouptype'] = df0['grouptype'].replace('talk.*', 3)
df0['grouptype'] = df0['grouptype'].replace('sci.*', 0)
df0['grouptype'] = df0['grouptype'].replace('rec.*', 1)

# 计算混淆矩阵
cm = confusion_matrix(df0.loc[:,'grouptype'], labels)
print(cm)

# 计算误判率
misclassification_rate = (np.sum(cm) - np.trace(cm)) / np.sum(cm)

print("misclassification_rate: "+ str(misclassification_rate))

end_time = time.time()
execution_time = end_time - start_time
print(f"Total cost time：{execution_time:.4f} seconds")


#3
df0 = pd.read_csv("C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5054机器学习\\作业\\Final project\\group_data.csv")
df = df0.iloc[:,0:100]

scaler = StandardScaler()

# 对数据进行标准化
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

start_time = time.time()

# 创建 PCA 模型并进行主成分分析
pca = PCA(n_components=5)
principal_components = pca.fit_transform(scaled_df)

# 将主成分数据转换为数据框
pc_df = pd.DataFrame(data=principal_components, columns=['PC1','PC2','PC3','PC4','PC5'])

random.seed(123)
C, labels = k_means(pc_df, 4, 20)

pc_df.loc[:,'grouptype']=labels

df0['grouptype'] = df0['grouptype'].replace('comp.*', 2)
df0['grouptype'] = df0['grouptype'].replace('talk.*', 3)
df0['grouptype'] = df0['grouptype'].replace('sci.*', 0)
df0['grouptype'] = df0['grouptype'].replace('rec.*', 1)

# 计算混淆矩阵
cm = confusion_matrix(df0.loc[:,'grouptype'], labels)
print(cm)

# 计算误判率
misclassification_rate = (np.sum(cm) - np.trace(cm)) / np.sum(cm)

print("misclassification_rate: "+ str(misclassification_rate))

end_time = time.time()
execution_time = end_time - start_time
print(f"Total cost time：{execution_time:.4f} seconds")


#4
#      MODEL        Accuracy   Time cost to train models
#
#  Random Forest   0.8148625     Middle
#       GBM        0.8228666     Large
#       LDA        0.7974388     Small
#       QDA        0.7710264     Small
#       SVM        0.8088906     Large
#     K-means      0.4881788     Small


#5
# 使用 PCA 将数据投影到前三个主成分上
pca = PCA(n_components=3)
df_pca = pca.fit_transform(scaled_df)

# 绘制投影图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
ax.scatter(df_pca[:, 0], df_pca[:, 1], df_pca[:, 2], marker='o')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('Projection of Data onto First Three Principal Components')

ax.view_init(elev=10, azim=-15)

plt.show()

# We can see the 4 clusters' structure.



#(5)
#1
import numpy as np
import pandas as pd
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

test=pd.read_csv("C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5054机器学习\\作业\\Final project\\MNIST\\test_resized.csv")
train=pd.read_csv("C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5054机器学习\\作业\\Final project\\MNIST\\train_resized.csv")

trainy=train.loc[:,"label"].values
testy=test.loc[:,"label"].values

tent1 = train.iloc[:, 1:].values
tent1 = (tent1 - np.mean(tent1, axis=1)[:, np.newaxis]) / np.std(tent1, axis=1)[:, np.newaxis]
tent2 = test.iloc[:, 1:].values
tent2 = (tent2 - np.mean(tent2, axis=1)[:, np.newaxis]) / np.std(tent2, axis=1)[:, np.newaxis]

trainx=np.array(tent1.reshape(30000, 12, 12))
testx=np.array(tent2.reshape(12000, 12, 12))

featuresTrain = torch.from_numpy(trainx)
targetsTrain = torch.from_numpy(trainy).type(torch.LongTensor) # data type is long

featuresTest = torch.from_numpy(testx)
targetsTest = torch.from_numpy(testy).type(torch.LongTensor) # data type is long

# Pytorch train and test TensorDataset
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

###########################################################################################

# Hyper Parameters
# batch_size, epoch and iteration
LR = 0.01
batch_size = 100
n_iters = 20000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch DataLoader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)

###########################################################################################

# Create CNN Model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(1,12,12)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0) #output_shape=(32,8,8)
        self.relu1 = nn.ReLU() # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,4,4)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0) #output_shape=(64,2,2)
        self.relu2 = nn.ReLU() # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) #output_shape=(64,1,1)
        # Fully connected 1 ,#input_shape=(64*1*1)
        self.fc1 = nn.Linear(64 * 1 * 1, 10) #output 0-9
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        # Max pool 1
        out = self.maxpool1(out)
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        # Max pool 2 
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        return out

###########################################################################################

model = CNN_Model()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
input_shape = (-1,1,12,12)

###########################################################################################

def fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, test_loader):
    # Traning the Model
    #history-like list for store loss & acc value
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    for epoch in range(num_epochs):
        #training model & store loss & acc / epoch
        correct_train = 0
        total_train = 0
        for i, (images, labels) in enumerate(train_loader):
            # 1.Define variables
            train = Variable(images.view(input_shape)).float()
            labels = Variable(labels)
            # 2.Clear gradients
            optimizer.zero_grad()
            # 3.Forward propagation
            outputs = model(train)
            # 4.Calculate softmax and cross entropy loss
            train_loss = loss_func(outputs, labels)
            # 5.Calculate gradients
            train_loss.backward()
            # 6.Update parameters
            optimizer.step()
            # 7.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 8.Total number of labels
            total_train += len(labels)
            # 9.Total correct predictions
            correct_train += (predicted == labels).float().sum()
        #10.store val_acc / epoch
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)
        # 11.store loss / epoch
        training_loss.append(train_loss.data)

        #evaluate model & store loss & acc / epoch
        correct_test = 0
        total_test = 0
        for images, labels in test_loader:
            # 1.Define variables
            test = Variable(images.view(input_shape)).float()
            # 2.Forward propagation
            outputs = model(test)
            # 3.Calculate softmax and cross entropy loss
            val_loss = loss_func(outputs, labels)
            # 4.Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # 5.Total number of labels
            total_test += len(labels)
            # 6.Total correct predictions
            correct_test += (predicted == labels).float().sum()
        #6.store val_acc / epoch
        val_accuracy = 100 * correct_test / float(total_test)
        validation_accuracy.append(val_accuracy)
        # 11.store val_loss / epoch
        validation_loss.append(val_loss.data)
        print('Train Epoch: {}/{} Traing_Loss: {} Traing_acc: {:.6f}% Val_Loss: {} Val_accuracy: {:.6f}%'.format(epoch+1, num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy))
    return training_loss, training_accuracy, validation_loss, validation_accuracy

###########################################################################################

start_time = time.time()

training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, test_loader)

end_time = time.time()
runtime = end_time - start_time
print("程序运行时间：", runtime, "秒")




#2
# VAE to visulization:

import keras
from keras import layers

from keras.datasets import mnist
import numpy as np

original_dim = 12 * 12
intermediate_dim = 64
latent_dim = 2

inputs = keras.Input(shape=(original_dim,))
h = layers.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(h)
z_log_sigma = layers.Dense(latent_dim)(h)

from keras import backend as K

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_sigma])

# Create encoder
encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

# Create decoder
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = layers.Dense(original_dim, activation='sigmoid')(x)
decoder = keras.Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae_mlp')
vae.summary()


reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

def change(input_arr):
    
    # 缩放后每行的长度
    scaled_length = 144
    
    # 生成插值的位置
    interpolation_indices = np.linspace(0, input_arr.shape[1] - 1, scaled_length)
    
    # 进行线性插值
    scaled_arr = np.zeros((input_arr.shape[0], scaled_length))
    for i, row in enumerate(input_arr):
        scaled_arr[i] = np.interp(interpolation_indices, np.arange(row.shape[0]), row)
    
    # 输出结果
    return scaled_arr

train=pd.read_csv("C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5054机器学习\\作业\\Final project\\MNIST\\train_resized.csv")
test=pd.read_csv("C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5054机器学习\\作业\\Final project\\MNIST\\test_resized.csv")
trainy=train.loc[:,"label"].values
testy=test.loc[:,"label"].values
tent1 = train.iloc[:, 1:].values
tent1 = (tent1 - np.mean(tent1, axis=1)[:, np.newaxis]) / np.std(tent1, axis=1)[:, np.newaxis]
tent2 = test.iloc[:, 1:].values
tent2 = (tent2 - np.mean(tent2, axis=1)[:, np.newaxis]) / np.std(tent2, axis=1)[:, np.newaxis]

x1 = change(x_train)
x2 = change(x_test)
total = np.vstack((x1, x2))
totallabel = np.concatenate((y_train, y_test))

vae.fit(total, total,
        epochs=100,
        batch_size=32,
        validation_data=(total, total),
        verbose=2)

x_test_encoded = encoder.predict(total)

plt.figure(figsize=(16, 16))
plt.scatter(x_test_encoded[0][:, 0], x_test_encoded[0][:, 1], c=totallabel, cmap='Set1',s=6)
plt.colorbar()
plt.show()

#########################################################################################
#######                        CAE example:

# import keras
# from keras import layers
# 
# input_img = keras.Input(shape=(28, 28, 1))
# 
# x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
# x = layers.MaxPooling2D((2, 2), padding='same')(x)
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = layers.MaxPooling2D((2, 2), padding='same')(x)
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
# 
# # at this point the representation is (4, 4, 8) i.e. 128-dimensional
# 
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
# x = layers.UpSampling2D((2, 2))(x)
# x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
# x = layers.UpSampling2D((2, 2))(x)
# x = layers.Conv2D(16, (3, 3), activation='relu')(x)
# x = layers.UpSampling2D((2, 2))(x)
# decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
# 
# autoencoder = keras.Model(input_img, decoded)
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# 
# from keras.datasets import mnist
# import numpy as np
# 
# (x_train, _), (x_test, _) = mnist.load_data()
# 
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
# x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
# 
# from keras.callbacks import TensorBoard
# 
# autoencoder.fit(x_train, x_train,
#                 epochs=50,
#                 batch_size=128,
#                 shuffle=True,
#                 validation_data=(x_test, x_test),
#                 verbose=2,
#                 callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
# 
# decoded_imgs = autoencoder.predict(x_test)
# 
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(1, n + 1):
#     # Display original
#     ax = plt.subplot(2, n, i)
#     plt.imshow(x_test[i+100].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# 
#     # Display reconstruction
#     ax = plt.subplot(2, n, i + n)
#     plt.imshow(decoded_imgs[i++100].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()



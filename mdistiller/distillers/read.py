# 两种方法都能打开
import pickle
import numpy as np

# f = open('./data.checkpoints', 'rb')
# data = pickle.load(f)
# print(data)

img_path = './data.pkl'
img_data = np.load(img_path)
print(img_data)

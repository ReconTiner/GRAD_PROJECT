import os
from PIL import Image
from PIL import ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# 训练集图片存储路径
image_dir = "./images/train/"
# 将所有的图片resize
width = 100
height = 100
c = 3


def read_file(image_dir):
    imgs = []
    labels = []
    for class_name in os.listdir(image_dir):
        # 标签赋值
        label = int(class_name)
        # 进入分类后的文件夹
        img_path = os.path.join(image_dir, class_name)
        # 获取分类后文件夹中的所有图像信息
        for image in tqdm(os.listdir(img_path)):
            img = Image.open(os.path.join(img_path, image))
            # 将图片重新设置尺寸
            img = img.resize((width, height), Image.ANTIALIAS)
            # 保存为数组形式
            img = img.convert("RGB")
            img_array = np.asarray(img)  # 原图片
            # 将图片数据以及标签加入数组中
            imgs.append(img_array)
            labels.append(label)
            # 对比度增强，增强因子为1.0是原始图片
            enh_con = ImageEnhance.Contrast(img)
            contrast = 1.5
            img_contrasted = enh_con.enhance(contrast)  # 对比度增强后的图片
            img_contrasted_array = np.asarray(img_contrasted)
            imgs.append(img_contrasted_array)
            labels.append(label)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


def data_analyst(label):
    # 数据可视化展示
    plt.figure(figsize=(20, 10), dpi=100)
    pd.value_counts(label).plot.bar()
    plt.show()


if __name__ == "__main__":
    # 加载数据
    data, label = read_file(image_dir)
    print(data.shape)
    print(label.shape)

    # # 保存数据
    np.save("./log/data.npy", data)
    np.save("./log/label.npy", label)

    data_analyst(label)

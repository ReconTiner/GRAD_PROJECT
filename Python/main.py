"""
传入分辨率为4000 * 4000的无人机影像, 依次执行以下步骤
1. 图片分割，分成200 * 200分辨率的图片共300张, 并压缩为100 * 100分辨率
2. 对每张子图像进行分类
3. 按照分类获得的标签信息, 为子图片用不同颜色进行填充
"""
from PIL import Image
import tensorflow.compat.v1 as tf
import numpy as np
from tqdm import tqdm


img_path = "G:\\grad_data\\908\\DJI_0204.JPG"  # 原始图片具体路径
# 调整压缩后的图像大小
width = 100
height = 100
# RGB三个通道数
channel = 3

label_name_dict = {
    0: "健康",
    1: "一般",
    2: "严重",
    }


def image_cutting(img_path):
    '''
    该函数用于将原始无人机影响分割为300等份, 并且每张图像分辨率调整为为100 * 100
    参数:
        img_path: 图片路径
    返回:
        img-list: 存放300张分割后子图像的数组
    '''
    img_list = []
    img = Image.open(img_path)
    # 将一张图片裁剪为20 * 15张, 每张像素大小为200 * 200
    split_width = 200
    split_height = 200
    for i in range(15):  # 垂直方向上分为15份
        for j in range(20):  # 水平方向上分为20份
            imgs = []  # 存放分割后的子图片
            box = (j*split_width, i*split_height,
                   (j+1)*split_width, (i+1)*split_height)
            child_img = img.crop(box)
            # 调整图像分辨率大小
            child_img = child_img.resize((width, height), Image.ANTIALIAS)
            # 将图片变为数组形式
            child_array = np.asarray(child_img)
            # 存储图像数据
            imgs.append(child_array)
            # 添加到列表中
            img_list.append(np.asarray(imgs, np.float32))

    return img_list


def image_classify(img_list):
    '''
    该函数用于图像分类
    参数:
        data_list: 某张子图片的数组数据
    返回:
        result: 分类结果
    '''
    result = []
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            "./model/fc_model.ckpt-135.meta")
        saver.restore(sess, tf.train.latest_checkpoint("./model/"))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        for data in tqdm(img_list):
            feed_dict = {x: data}

            logits = graph.get_tensor_by_name("logits_eval:0")
            classification_result = sess.run(logits, feed_dict)

            # 打印出预测矩阵
            # print(classification_result)
            # 预测矩阵每一行最大值的索引
            # tf.argmax(classification_result, 1).eval()
            # 根据索引通过字典对应分类结果
            output = tf.argmax(classification_result, 1).eval()
            # print("该图像识别为:" + label_name_dict[output[0]])
            result.append(output[0])

    return result


def image_synthesis(img_path, classify_result):
    '''
    该函数根据分类结果, 生成合成后的图片
    参数:
        img_path: 原始图像路径
        classify_result: 分类结果
    返回:
        result_img: 合成后的图片
    '''
    img = Image.open(img_path)
    # 将图像转成数组格式
    img_array = np.asarray(img)
    count = 0
    # 判断分割区域是否需要修改颜色
    for i in range(15):  # 垂直方向上分为15份
        for j in range(20):  # 水平方向上分为20份
            # 该区域左上角坐标
            left_top_x = j * 200
            left_top_y = i * 200
            if classify_result[count] == 0:
                # 该区域分类结果为健康
                pass
            elif classify_result[count] == 1:
                # 该区域分类结果为一般(蓝色)
                for x in range(left_top_x, left_top_x+200):
                    for y in range(left_top_y, left_top_y+200):
                        img_array[y, x] = (0, 0, 255)
            elif classify_result[count] == 2:
                # 该区域分类结果为严重
                for x in range(left_top_x, left_top_x+200):
                    for y in range(left_top_y, left_top_y+200):
                        img_array[y, x] = (255, 0, 0)
            count += 1

    return Image.fromarray(np.uint8(img_array))


if __name__ == "__main__":
    image_list = image_cutting(img_path)
    result = image_classify(image_list)
    result_image = image_synthesis(img_path, result)
    result_image.save("result.png", "png")

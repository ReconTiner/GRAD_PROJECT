import tensorflow.compat.v1 as tf
import numpy as np
import time
from tqdm import tqdm

tf.disable_v2_behavior()

# 读取数据
data = np.load("./log/data.npy")
label = np.load("./log/label.npy")

# 数据信息
height = 100  # 图片高度
width = 100  # 图片宽度
channel = 3  # RGB三个通道

# 模型保存地址
model_path = "./model/new/fc_model_pro.ckpt"

# 打乱顺序
# 读取data矩阵的第一维数（图片的个数）
num_example = data.shape[0]
# 产生一个num_example范围，步长为1的序列
arr = np.arange(num_example)
# 调用函数，打乱顺序
np.random.shuffle(arr)
# 按照打乱的顺序，重新排序
data = data[arr]
label = label[arr]

# 将所有数据分为训练集和验证集
ratio = 0.8
s = np.int(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]

# 构建网络
"""
搭建CNN类型网络
本网络共包含7层，前三层为卷积层（包含卷积、激活、池化层），后三层为全连接层
"""
# 设置占位符
x = tf.placeholder(tf.float32, shape=[None, width, height, channel], name="x")
y_ = tf.placeholder(tf.int32, shape=[None, ], name="y_")


def inference(input_tensor, train, regularizer):
    # 第一层网络
    # 卷积层
    with tf.compat.v1.variable_scope("layer1-conv"):
        # 初始化权重conv1_weights，大小为5×5,，3个通道，数量为32个
        conv1_weights = tf.get_variable(
            "weight",
            [5, 5, 3, 32],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        # 初始化偏执conv1——bias，数量为32个
        conv1_biases = tf.get_variable(
            "bias",
            [32],
            initializer=tf.constant_initializer(0.0)
        )
        # 卷积计算，tf.nn.conv2d为tensorflow自带2维卷积函数，input_tensor为输入数据
        conv1 = tf.nn.conv2d(
            input_tensor,
            conv1_weights,
            strides=[1, 1, 1, 1],
            padding="SAME"
        )
        # 激励计算
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    # 池化层
    with tf.compat.v1.name_scope("layer1-pool"):
        # 池化计算
        pool1 = tf.nn.max_pool(
            relu1,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="VALID"
        )
    # 第二层网络
    # 卷积层
    with tf.compat.v1.variable_scope("layer2-conv"):
        conv2_weights = tf.get_variable(
            "weight",
            [5, 5, 32, 64],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases = tf.get_variable(
            "bias",
            [64],
            initializer=tf.constant_initializer(0.0)
        )
        conv2 = tf.nn.conv2d(
            pool1,
            conv2_weights,
            strides=[1, 1, 1, 1],
            padding="SAME"
        )
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    # 池化层
    with tf.compat.v1.name_scope("layer2-pool"):
        pool2 = tf.nn.max_pool(
            relu2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="VALID"
        )
    # 第三层网络
    # 卷积层
    with tf.compat.v1.variable_scope("layer3-conv"):
        conv3_weights = tf.get_variable(
            "weight",
            [3, 3, 64, 128],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv3_biases = tf.get_variable(
            "bias",
            [128],
            initializer=tf.constant_initializer(0.0)
        )
        conv3 = tf.nn.conv2d(
            pool2,
            conv3_weights,
            strides=[1, 1, 1, 1],
            padding="SAME"
        )
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
    # 池化层
    with tf.compat.v1.name_scope("layer3-pool"):
        pool3 = tf.nn.max_pool(
            relu3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="VALID"
        )
    # 第四层网络
    # 卷积层
    with tf.compat.v1.variable_scope("layer4-conv"):
        conv4_weights = tf.get_variable(
            "weight",
            [3, 3, 128, 128],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv4_biases = tf.get_variable(
            "bias",
            [128],
            initializer=tf.constant_initializer(0.0)
        )
        conv4 = tf.nn.conv2d(
            pool3,
            conv4_weights,
            strides=[1, 1, 1, 1],
            padding="SAME"
        )
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
    # 池化层
    with tf.compat.v1.name_scope("layer4-pool"):
        pool4 = tf.nn.max_pool(
            relu4,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="VALID"
        )
        nodes = 6*6*128
        reshaped = tf.reshape(pool4, [-1, nodes])
    # 第五层网络
    with tf.compat.v1.variable_scope("layer5-fc1"):
        # 初始化全连接层的参数，隐含节点为1024个
        fc1_weights = tf.get_variable(
            "weight",
            [nodes, 1024],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        # 正则化参数
        if regularizer is not None:
            tf.add_to_collection(
                "losses",
                regularizer * tf.nn.l2_loss(fc1_weights)
            )
        fc1_biases = tf.get_variable(
            "bias",
            [1024],
            initializer=tf.constant_initializer(0.1)
        )
        # 使用RELU作为激活函数
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        # 采用dropout层，减少过拟合和欠拟合的程度
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
    # 第六层网络
    with tf.compat.v1.variable_scope("layer6-fc2"):
        # 初始化全连接层的参数，隐含节点为1024个
        fc2_weights = tf.get_variable(
            "weight",
            [1024, 512],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        # 正则化参数
        if regularizer is not None:
            tf.add_to_collection(
                "losses",
                regularizer * tf.nn.l2_loss(fc2_weights)
            )
        fc2_biases = tf.get_variable(
            "bias",
            [512],
            initializer=tf.constant_initializer(0.1)
        )
        # 使用RELU作为激活函数
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        # 采用dropout层，减少过拟合和欠拟合的程度
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)
    # 第七层网络
    with tf.compat.v1.variable_scope("layer7-fc3"):
        # 初始化全连接层的参数，隐含节点为1024个
        fc3_weights = tf.get_variable(
            "weight",
            [512, 3],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        # 正则化参数
        if regularizer is not None:
            tf.add_to_collection(
                "losses",
                regularizer * tf.nn.l2_loss(fc3_weights)
            )

        fc3_biases = tf.get_variable(
            "bias",
            [3],
            initializer=tf.constant_initializer(0.1)
        )
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases
    # 返回最终结果
    return logit


# 设置正则化参数
regularizer = 0.0001

# 将上述构建网络结构引入
logits = inference(x, False, regularizer)

# (小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1, dtype=tf.float32)
logits_eval = tf.multiply(logits, b, name='logits_eval')

# 设置损失函数，作为模型训练优化的参考标准，loss越小，模型越优
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
# 设置整体学习率
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# 设置预测精度
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# 迭代次数
n_epoch = 150
# 每次迭代送入的图片数据
batch_size = 64
# 指定保存的模型个数，利用max_to_keep=4，则最终会保存4个模型
saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    # 初始化全局参数
    sess.run(tf.global_variables_initializer())
    # 开始迭代训练
    for epoch in tqdm(range(n_epoch)):
        start_time = time.time()

        # 训练集
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(
                x_train, y_train, batch_size, shuffle=True):
            non, err, ac = sess.run(
                [train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err
            train_acc += ac
            n_batch += 1
            # print("train loss: %f" % (np.sum(train_loss) / n_batch))
            # print("train acc: %f" % (np.sum(train_acc) / n_batch))

        # 验证集
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(
                x_val, y_val, batch_size, shuffle=False):
            err, ac = sess.run(
                [loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err
            val_acc += ac
            n_batch += 1
            # print("validation loss: %f" % (np.sum(val_loss) / n_batch))
            # print("validation acc: %f" % (np.sum(val_acc) / n_batch))

        # 保存模型及模型参数
        if epoch % 15 == 0:
            saver.save(sess, model_path, global_step=epoch)
            print("train loss: %f" % (np.sum(train_loss) / n_batch))
            print("train acc: %f" % (np.sum(train_acc) / n_batch))
            print("validation loss: %f" % (np.sum(val_loss) / n_batch))
            print("validation acc: %f" % (np.sum(val_acc) / n_batch))

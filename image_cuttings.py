from PIL import Image
import os


# 原始图片存储目录
path_list = [
    "G:\\grad_data\\908",
    "G:\\grad_data\\919",
    "G:\\grad_data\\924"
]

# 裁剪后图片的存储目录
save_dir = ".\\split_image\\"


def image_cutting(image_path, image_name, save_path):
    '''
    该函数用于将(4000 * 3000)大小的图片裁剪为20 * 15等份
    参数:
        iamge_path: 图片所在目录
        image_name: 图片名称
        save_paht: 图片保存路径
    返回:
        None
    '''
    img = Image.open(image_path)
    # 将一张图片裁剪为20 * 15张, 每张像素大小为200 * 200
    split_width = 200
    split_height = 200
    for i in range(15):  # 垂直方向上分为15份
        for j in range(20):  # 水平方向上分为20份
            box = (j*split_width, i*split_height,
                   (j+1)*split_width, (i+1)*split_height)
            child_img = img.crop(box)
            child_img.save(save_path + image_name + f"_({j}_{i}).jpg")
    return


if __name__ == "__main__":
    # 逐个遍历原始图片目录
    for path in path_list:
        # 908, 919, 924
        dir = path.split("\\")[-1]
        # 获取原始图片目录下的所有图片名称
        image_list = os.listdir(path)
        # 创建对应日期的文件夹
        os.mkdir(save_dir + dir)

        # 为每一张图片创建一个保存目录并进行裁剪
        for image in image_list:
            save_path = os.path.join(save_dir, dir, image).replace(
                ".JPG", "\\")
            os.mkdir(save_path)
            # 裁剪图片
            image_cutting(os.path.join(path, image), image, save_path)
            print(os.path.join(path, image), image + " 裁剪完成!")

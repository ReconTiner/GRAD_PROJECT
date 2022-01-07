import exifread
import csv
import os
from tqdm import tqdm


dir_path = "./images/ImageBase/"  # 图片路径


def get_image_info(img_path, img_name):
    '''
    该函数用来处理图片 并返回图片的拍摄时间以及经纬度信息
    参数:
        img_path - 传入图片的路径
        img_name - 传入图片的名称
    返回:
        info_list - 存储照片信息的列表
    '''
    info_list = []
    with open(img_path, 'rb') as f:
        tags = exifread.process_file(f)
        # 照片名称
        info_list.append(img_name)
        # 拍摄时间
        image_datatime = tags.get('Image DateTime', '0').values
        info_list.append(image_datatime)
        # 维度
        lat = tags.get('GPS GPSLatitude', '0').values
        info_list.append(process_lonlat(lat))
        # 经度
        lon = tags.get('GPS GPSLongitude', '0').values
        info_list.append(process_lonlat(lon))
    return info_list


def process_lonlat(x):
    '''
    处理经纬度 将其转化为 xx.xxxxxx格式
    注意列表中的每一个元素 是 <class 'exifread.utils.Ratio'>
    由于最后一个是 10243/2000 这样的格式 需要手动将其处理 其余的使用 .num 方法就能获得到值
    参数:
        x - 传入的经度和纬度
    返回:
        处理好了经纬度
    '''
    # 处理列表中最后一个元素
    x_last = eval(str(x[-1]))
    #  转化
    new_x = x[0].num + x[1].num / 60 + x_last / 3600

    return '{:.13f}'.format(new_x)


if __name__ == "__main__":
    # 将照片信息写入到CSV文件中
    with open('image_info.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ImageName", "ImageDataTime", "LAT", "LON"])
        image_list = os.listdir(dir_path)
        for image in tqdm(image_list):
            img_path = os.path.join(dir_path, image)
            info_list = get_image_info(img_path, image)
            writer.writerow(info_list)

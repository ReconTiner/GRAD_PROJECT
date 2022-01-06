## 基于Python实现各项后端功能
### 文件说明
**1. 图像分割：image_cutting.py**  
将分辨率为4000×4000的无人机影像分割为200×200的300等份，用于分类生成训练集    
**2. 数据加载：data_loading.py**  
将训练集样本加载为np文件，之后仅需调用生成的np文件即可进行模型训练，避免多次进行图像加载操作  
**3. 深度学习分类模型: model.py**  
使用CNN进行训练，生成分类模型  
**4. 主程序：main.py**  
通过输入需要识别的原始图片路径，通过分割操作，对子图片逐个分类，最后根据分类结果生成合成后的图片
![Image text](https://github.com/ReconTiner/GRAD_PROJECT/blob/main/Python/img_result.jpg)

# GRAD_PROJECT
本科毕业设计

主题为WEB的智慧农业管理系统

## 主要功能目前包括
### 在线虫害识别（重点）
  模型建立：通过将一张无人机影像裁剪为300张，分为0，1，2三类，分别对应不同程度的虫害影响，使用CNN模型进行训练，建立初步模型。
  模型使用：前端传入图片，在后端进行模型调用，同样将传来的图片进行裁剪分割，返回分类结果，对受到虫害影响的部分进行标识，并将标识结果与原图像进行叠加。

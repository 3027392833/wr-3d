import numpy as np
import matplotlib.pyplot as plt
from CBCTReconstructor import load_nii

# sinogram=np.load("sinogram.npy")
# print(sinogram.shape)
# for i in range(sinogram.shape[0]): #(height,angles,width)
#     plt.imshow(sinogram[i,:,:],cmap='gray')
#     plt.show()

input_nii_path = r"C:\Users\yan\Desktop\X2313838.nii"  # 输入的 .nii 文件路径
output_nii_path = 'output/reconstruction_angle_90.0_image.nii'  # 输出的 .nii 文件路径
sinogram_npy_path = 'sinogram.npy'  # 保存投影数据的路径
detector_width = 430  # 探测器宽度
detector_height = 430  # 探测器高度
source_origin = 1233  # 光源到旋转中心的距离
origin_detector = 267  # 旋转中心到探测器的距离
angle_range = (0, 180)  # 有限角度范围（0 到 180 度）
angle_step = 1.0  # 角度步长（度）

# 加载 .nii 文件
data, affine = load_nii(output_nii_path)
for i in range(data.shape[2]):
    plt.imshow(data[:,:,i],cmap='gray')
    plt.axis('off')
    plt.show()
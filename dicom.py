import numpy as np
import astra
import matplotlib.pyplot as plt
import nibabel as nib
import os
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import generate_uid
from skimage.metrics import peak_signal_noise_ratio as psnr
# --------------------------
# 工具函数：加载和保存数据
# --------------------------

#CBCTReconstruction write by dicom



def load_dicom(dicom_dir):
    """
    加载 DICOM 文件并返回图像数据和元数据
    """
    if not os.path.exists(dicom_dir):
        raise FileNotFoundError(f"DICOM directory {dicom_dir} not found.")

    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')or f.endswith('.IMA')]
    dicom_files.sort()  # 确保文件按顺序加载

    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))  # 按 Z 轴排序

    # 获取像素数据
    data = np.stack([s.pixel_array for s in slices])
    data = data.astype(np.float32)  # 转换为浮点数

    # 获取元数据（例如像素间距、切片厚度等）
    metadata = {
        'pixel_spacing': slices[0].PixelSpacing[0],
        'slice_thickness': slices[0].SliceThickness,
        'image_position': slices[0].ImagePositionPatient,
        'image_orientation': slices[0].ImageOrientationPatient,
    }
    # 保存每个切片的头文

    return data, metadata,slices

def save_dicom(data, output_dir, slices):
    """
    将数据保存为 DICOM 文件，直接修改 slices 中的 pixel_array
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 为每个切片创建 DICOM 文件
    for i in range(data.shape[0]):
        # 直接修改原始 DICOM 文件的像素数据
        pixel_data_int16 = data[i, :, :].astype(np.int16)

        # 直接修改原始 DICOM 文件的像素数据
        slices[i].PixelData = pixel_data_int16.tobytes()

        # 更新 SOP 实例 UID（确保唯一性）
        slices[i].SOPInstanceUID = generate_uid()


        # 保存文件
        output_path = os.path.join(output_dir, f'{output_dir[-4:]}_{i + 1:04d}.dcm')
        slices[i].save_as(output_path)
        # print(f"Saved {output_path}")


def min_max_normalization(data, feature_range=(0, 1)):
    """
    对数据进行最大最小归一化。

    参数：
        data (array-like): 原始数据，numpy 数组或列表。
        feature_range (tuple): 目标归一化范围，默认 (0, 1)。

    返回：
        numpy.ndarray: 归一化后的数据。
    """
    data = np.array(data)
    min_val, max_val = feature_range
    data_min = data.min()
    data_max = data.max()

    # 避免分母为 0
    if data_max - data_min == 0:
        return np.full(data.shape, min_val)

    normalized_data = min_val + (data - data_min) * (max_val - min_val) / (data_max - data_min)
    return normalized_data

class CBCTReconstructor:
    def __init__(self, data, metadata,slices, detector_width, detector_height,
                 source_origin, origin_detector, angle_range=(0, 360), angle_step=1.0):
        """
        初始化 CBCT 重建器

        参数:
        - data: 输入的 3D 图像数据
        - metadata: DICOM 元数据
        - detector_width: 探测器宽度（像素数）
        - detector_height: 探测器高度（像素数）
        - source_origin: 光源到旋转中心的距离（毫米）
        - origin_detector: 旋转中心到探测器的距离（毫米）
        - angle_range: 角度范围（起始角度, 结束角度），默认为 (0, 360)
        - angle_step: 角度步长（度），控制投影角度的密度
        """
        self.data = data
        self.metadata = metadata
        self.slices = slices
        self.detector_width = detector_width
        self.detector_height = detector_height
        self.source_origin = source_origin
        self.origin_detector = origin_detector
        self.angle_range = angle_range
        self.angle_step = angle_step

        # 检查数据维度
        if self.data.ndim != 3:
            raise ValueError("Input data must be 3D (slices, rows, columns).")

    def reconstruct(self, sinogram_npy_path, output_dicom_dir):
        """
        执行 CBCT 投影和重建

        参数:
        - sinogram_npy_path: 保存投影数据的路径
        - output_dicom_dir: 输出的 DICOM 文件目录

        返回:
        - reconstruction: 重建后的 3D 图像数据
        """
        # 计算投影角度
        start_angle, end_angle = self.angle_range
        angles = np.arange(start_angle, end_angle, self.angle_step)  # 根据 angle_range 和 angle_step 生成角度
        angles_rad = np.deg2rad(angles)  # 转换为弧度

        # 创建 CBCT 投影几何（锥形束）
        proj_geom = astra.create_proj_geom('cone',  # 锥形束几何
                                           0.5, 0.5,  # 探测器像素尺寸
                                           self.detector_height, self.detector_width,  # 探测器尺寸
                                           angles_rad,  # 投影角度
                                           self.source_origin, self.origin_detector)  # 几何参数

        # 创建体几何
        # vol_geom = astra.create_vol_geom(self.data.shape[1], self.data.shape[2], self.data.shape[0],
        #                                  -self.data.shape[1] * self.metadata['pixel_spacing'] / 2.0,
        #                                  self.data.shape[1] * self.metadata['pixel_spacing'] / 2.0,
        #                                  -self.data.shape[2] * self.metadata['pixel_spacing'] / 2.0,
        #                                  self.data.shape[2] * self.metadata['pixel_spacing'] / 2.0,
        #                                  -self.data.shape[0] * self.metadata['slice_thickness'] / 2.0,
        #                                  self.data.shape[0] * self.metadata['slice_thickness'] / 2.0
        #                                  )  # 3D 体数据
        vol_geom = astra.create_vol_geom(self.data.shape[1], self.data.shape[2], self.data.shape[0],
                                         -self.data.shape[1] * self.metadata['pixel_spacing'] / 2.0,
                                         self.data.shape[1] * self.metadata['pixel_spacing'] / 2.0,
                                         -self.data.shape[2] * self.metadata['pixel_spacing'] / 2.0,
                                         self.data.shape[2] * self.metadata['pixel_spacing'] / 2.0,
                                         -self.data.shape[0] * self.metadata['pixel_spacing'] / 2.0,
                                         self.data.shape[0] * self.metadata['pixel_spacing'] / 2.0
                                         )  # 3D 体数据
        # 创建投影器
        projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)  # 使用 CUDA 加速

        # 创建投影数据对象
        proj_id = astra.data3d.create('-sino', proj_geom)

        # 创建体数据对象
        vol_id = astra.data3d.create('-vol', vol_geom, self.data)

        # 配置正向投影算法
        fp_cfg = astra.astra_dict('FP3D_CUDA')
        fp_cfg['ProjectionDataId'] = proj_id
        fp_cfg['VolumeDataId'] = vol_id
        fp_cfg['ProjectorId'] = projector_id

        # 创建并运行正向投影算法
        fp_id = astra.algorithm.create(fp_cfg)
        astra.algorithm.run(fp_id)

        # 获取投影数据（正弦图）
        sinogram = astra.data3d.get(proj_id)  # (height, angles, width)

        # 保存投影数据
        # np.save(sinogram_npy_path, sinogram)
        # print(f"Projection data saved to {sinogram_npy_path}")

        vol_id_recon = astra.data3d.create('-vol', vol_geom)

        # 配置滤波反投影（FBP）算法
        fbp_cfg = astra.astra_dict('FDK_CUDA')
        # fbp_cfg = astra.astra_dict('SIRT_CUDA')
        fbp_cfg['ProjectionDataId'] = proj_id
        fbp_cfg['ReconstructionDataId'] = vol_id_recon
        fbp_cfg['ProjectorId'] = projector_id

        # 创建并运行 FBP 算法
        fbp_id = astra.algorithm.create(fbp_cfg)
        astra.algorithm.run(fbp_id,20)

        # 获取重建图像
        reconstruction = astra.data3d.get(vol_id_recon)
        reconstruction[reconstruction<0] = 0
        reconstruction[reconstruction>4095] = 4095
        print(reconstruction.shape)
        print(reconstruction.max())
        print(reconstruction.min())

        # 将重建结果保存为 DICOM 文件
        save_dicom(reconstruction, output_dicom_dir, self.slices)
        print(f"Reconstructed DICOM files saved to {output_dicom_dir}")

        # 清理 ASTRA 对象
        astra.algorithm.delete(fp_id)
        astra.algorithm.delete(fbp_id)
        astra.data3d.delete(proj_id)
        astra.data3d.delete(vol_id)
        astra.projector.delete(projector_id)

        # 返回重建结果
        return reconstruction

    def visualize(self, reconstruction, sinogram_npy_path):
        """
        可视化原始图像、正弦图和重建图像

        参数:
        - reconstruction: 重建后的 3D 图像数据
        - sinogram_npy_path: 投影数据的路径
        """
        slice_idx = self.data.shape[0] // 2  # 中间切片
        reconstruction = min_max_normalization(reconstruction)
        self.data=min_max_normalization(self.data)

        plt.figure(figsize=(12, 6))

        # 显示原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(self.data[slice_idx, :, :], cmap='gray')
        plt.title('Original Slice')

        # 显示重建图像
        plt.subplot(1, 3, 2)
        plt.imshow(reconstruction[slice_idx, :, :], cmap='gray')
        plt.title('Reconstructed Slice (FBP)')

        # 显示重建图像
        plt.subplot(1, 3, 3)
        plt.imshow(reconstruction[slice_idx, :, :]-self.data[slice_idx, :, :], cmap='gray')
        plt.title('Reconstructed Slice (FBP)')

        plt.tight_layout()
        plt.show()
        print(reconstruction[slice_idx, :, :].max())
        print(reconstruction[slice_idx, :, :].min())
        print(psnr(reconstruction[slice_idx, :, :],self.data[slice_idx, :, :]))
        print(psnr(self.data[slice_idx, :, :], self.data[slice_idx, :, :]))

if '__main__' == __name__:
    dcm_dir = r'D:\dataset\FD_1mm\full_1mm\L506\full_1mm'
    data, metadata,slices= load_dicom(dcm_dir)
    print(data.shape)
    print(metadata)
    print(data.max())
    print(data.min())
    output_dir = r'D:\dataset\1mm_120'  # 输出目录
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    detector_width = 1000  # 探测器宽度
    detector_height = 1600 # 探测器高度
    source_origin = 1233  # 光源到旋转中心的距离
    origin_detector = 267  # 旋转中心到探测器的距离
    angle_range = (0, 120)  # 有限角度范围（0 到 180 度）
    angle_step = 1 # 角度步长（度）
    angle=angle_range[1]-angle_range[0] # 选择要可视化的角度
    output_dicom_path = os.path.join(output_dir, f'L506')
    print(output_dicom_path)
    sinogram_npy_path = os.path.join(output_dir, f'sinogram_angle_{angle:.1f}.npy')

    # 创建 CBCT 重建器
    reconstructor = CBCTReconstructor(data, metadata,slices,detector_width, detector_height,
                                      source_origin, origin_detector, angle_range, angle_step)

    # 执行重建
    reconstruction = reconstructor.reconstruct(sinogram_npy_path, output_dicom_path)

    # 可视化结果
    reconstructor.visualize(reconstruction, sinogram_npy_path)


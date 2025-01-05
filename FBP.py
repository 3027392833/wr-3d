import numpy as np
import astra
import matplotlib.pyplot as plt
import nibabel as nib
import os

# --------------------------
# 工具函数：加载和保存数据
# --------------------------

def load_nii(file_path):
    """
    加载 .nii 文件并返回图像数据和 affine 矩阵
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file {file_path} not found.")
    nii_image = nib.load(file_path)
    data = nii_image.get_fdata()  # 获取图像数据
    affine = nii_image.affine  # 获取 affine 矩阵
    return data, affine

def save_nii(data, affine, output_path):
    """
    将数据保存为 .nii 文件
    """
    nii_image = nib.Nifti1Image(data, affine)
    nib.save(nii_image, output_path)

def save_npy(data, output_path):
    """
    将数据保存为 .npy 文件
    """
    np.save(output_path, data)

# --------------------------
# CBCT 重建类
# --------------------------

class CBCTReconstructor:
    def __init__(self, data, affine, detector_width, detector_height,
                 source_origin, origin_detector, angle_range=(0, 360), angle_step=1.0):
        """
        初始化 CBCT 重建器

        参数:
        - data: 输入的 3D 图像数据
        - affine: 图像的 affine 矩阵
        - detector_width: 探测器宽度（像素数）
        - detector_height: 探测器高度（像素数）
        - source_origin: 光源到旋转中心的距离（毫米）
        - origin_detector: 旋转中心到探测器的距离（毫米）
        - angle_range: 角度范围（起始角度, 结束角度），默认为 (0, 360)
        - angle_step: 角度步长（度），控制投影角度的密度
        """
        self.data = data
        self.affine = affine
        self.detector_width = detector_width
        self.detector_height = detector_height
        self.source_origin = source_origin
        self.origin_detector = origin_detector
        self.angle_range = angle_range
        self.angle_step = angle_step

        # 检查数据维度
        if self.data.ndim != 3:
            raise ValueError("Input data must be 3D (slices, rows, columns).")

    def reconstruct_single_angle(self, angle, output_nii_path):
        """
        对单个角度进行投影和重建，并保存结果

        参数:
        - angle: 当前角度（度）
        - output_nii_path: 输出的 .nii 文件路径

        返回:
        - reconstruction: 重建后的 3D 图像数据
        """
        # 将角度转换为弧度
        angle_rad = np.deg2rad(angle)

        # 创建 CBCT 投影几何（锥形束）
        proj_geom = astra.create_proj_geom('cone',  # 锥形束几何
                                           1.0, 1.0,  # 探测器像素尺寸
                                           self.detector_height, self.detector_width,  # 探测器尺寸
                                           [angle_rad],  # 当前角度
                                           self.source_origin, self.origin_detector)  # 几何参数

        # 创建体几何
        vol_geom = astra.create_vol_geom(self.data.shape[1], self.data.shape[2], self.data.shape[0])  # 3D 体数据

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

        # 配置滤波反投影（FBP）算法
        fbp_cfg = astra.astra_dict('FDK_CUDA')
        fbp_cfg['ProjectionDataId'] = proj_id
        fbp_cfg['ReconstructionDataId'] = vol_id
        fbp_cfg['ProjectorId'] = projector_id

        # 创建并运行 FBP 算法
        fbp_id = astra.algorithm.create(fbp_cfg)
        astra.algorithm.run(fbp_id)

        # 获取重建图像
        reconstruction = astra.data3d.get(vol_id)

        # 将重建结果保存为 .nii 文件
        save_nii(reconstruction, self.affine, output_nii_path)
        print(f"Reconstructed image for angle {angle}° saved to {output_nii_path}")

        # 清理 ASTRA 对象
        astra.algorithm.delete(fp_id)
        astra.algorithm.delete(fbp_id)
        astra.data3d.delete(proj_id)
        astra.data3d.delete(vol_id)
        astra.projector.delete(projector_id)

        # 返回重建结果
        return reconstruction

    def reconstruct_all_angles(self, output_dir):
        """
        对所有角度进行投影和重建，并保存每个角度的结果

        参数:
        - output_dir: 输出目录，用于保存每个角度的重建结果
        """
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 计算投影角度
        start_angle, end_angle = self.angle_range
        angles = np.arange(start_angle, end_angle, self.angle_step)  # 根据 angle_range 和 angle_step 生成角度

        # 对每个角度进行重建
        for angle in angles:
            output_nii_path = os.path.join(output_dir, f'reconstruction_angle_{angle:.1f}.nii')
            self.reconstruct_single_angle(angle, output_nii_path)

    def visualize(self, reconstruction, angle):
        """
        可视化原始图像和重建图像

        参数:
        - reconstruction: 重建后的 3D 图像数据
        - angle: 当前角度（度）
        """
        slice_idx = self.data.shape[0] // 2  # 中间切片

        plt.figure(figsize=(10, 5))

        # 显示原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(self.data[slice_idx, :, :], cmap='gray')
        plt.title('Original Slice')

        # 显示重建图像
        plt.subplot(1, 2, 2)
        plt.imshow(reconstruction[slice_idx, :, :], cmap='gray')
        plt.title(f'Reconstructed Slice (Angle {angle}°)')

        plt.tight_layout()
        plt.show()


# --------------------------
# 示例用法
# --------------------------

if __name__ == "__main__":
    # 定义参数
    input_nii_path = r"C:\Users\yan\Desktop\X2313838.nii"  # 输入的 .nii 文件路径
    output_dir = 'reconstructed_images'  # 输出目录
    detector_width = 430  # 探测器宽度
    detector_height = 100  # 探测器高度
    source_origin = 1233  # 光源到旋转中心的距离
    origin_detector = 267  # 旋转中心到探测器的距离
    angle_range = (0, 180)  # 有限角度范围（0 到 180 度）
    angle_step = 30.0  # 角度步长（度）

    # 加载 .nii 文件
    data, affine = load_nii(input_nii_path)

    # 创建 CBCT 重建器
    reconstructor = CBCTReconstructor(data, affine, detector_width, detector_height,
                                      source_origin, origin_detector, angle_range, angle_step)

    # 对所有角度进行重建
    reconstructor.reconstruct_all_angles(output_dir)

    # 可视化某个角度的重建结果
    angle_to_visualize = 45  # 选择要可视化的角度
    output_nii_path = os.path.join(output_dir, f'reconstruction_angle_{angle_to_visualize:.1f}.nii')
    reconstruction = load_nii(output_nii_path)[0]  # 加载重建结果
    reconstructor.visualize(reconstruction, angle_to_visualize)
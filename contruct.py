import numpy as np
import astra
import os
import matplotlib.pyplot as plt

# 读取保存的投影数据
output_dir = 'projections'
num_projections = 360
det_row_count = 256
det_col_count = 256

# 从jpg获取投影数据
# projections = np.zeros((det_row_count, num_projections, det_col_count), dtype=np.float32)
#
# for i in range(det_row_count):
#     input_file = os.path.join(output_dir, f'projection_{i:03d}.jpg')
#     projection = imread(input_file, as_gray=True)
#     projections[i] = projection
#
# # 将投影数据恢复到原始范围
# projections *= np.max(projections)

# 获取二进制投影数据
input_file = 'projections.npy'
projections = np.load(input_file)
print(projections.shape)
# 投影几何
angles = np.linspace(0, 2 * np.pi, num_projections, endpoint=False)
source_origin = 200
origin_det = 200
proj_geom = astra.create_proj_geom('cone', 1, 1, det_row_count, det_col_count, angles, source_origin, origin_det)

# 体积几何
vol_geom = astra.create_vol_geom(128, 128, 128)

# 创建投影数据和重建数据对象
proj_id = astra.data3d.create('-proj3d', proj_geom, projections)
recon_id = astra.data3d.create('-vol', vol_geom)

# 配置并执行SIRT算法
# cfg = astra.astra_dict('SIRT3D_CUDA')
cfg = astra.astra_dict('FDK_CUDA')

cfg['ProjectionDataId'] = proj_id
cfg['ReconstructionDataId'] = recon_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 1)  # 迭代次数

# 获取并显示重建结果
reconstruction = astra.data3d.get(recon_id)
for i in range(48,81):
    plt.imshow(reconstruction[i], cmap='gray')  # 显示中间切片
    plt.colorbar()
    plt.show()

# 清理
astra.algorithm.delete(alg_id)
astra.data3d.delete(proj_id)
astra.data3d.delete(recon_id)


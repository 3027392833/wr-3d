import numpy as np
import astra
import matplotlib.pyplot as plt
import os
# from skimage.io import imread

# 创建一个简单的3D幻灯片（立方体）
vol_geom = astra.create_vol_geom(128, 128, 128)
phantom = np.zeros(vol_geom['GridRowCount'] * vol_geom['GridColCount'] * vol_geom['GridSliceCount'], dtype='float32')
phantom = phantom.reshape(vol_geom['GridRowCount'], vol_geom['GridColCount'], vol_geom['GridSliceCount'])
phantom[50:78, 50:78, 50:78] = 1

# 创建一个圆锥光投影几何
angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
det_col_count = 256  # 列数
det_row_count = 256 # 行数---->最外层
source_origin = 200
origin_det = 200
proj_geom = astra.create_proj_geom('cone', 1, 1, det_row_count, det_col_count, angles, source_origin, origin_det)

# 创建一个投影数据对象
proj_id = astra.data3d.create('-proj3d', proj_geom)
# proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)


# 计算投影数据
vol_id = astra.data3d.create('-vol', vol_geom, phantom)


alg_cfg = astra.astra_dict('FP3D_CUDA')
alg_cfg['ProjectionDataId'] = proj_id
alg_cfg['VolumeDataId'] = vol_id
alg_id = astra.algorithm.create(alg_cfg)
astra.algorithm.run(alg_id)

# 获取投影数据
projections = astra.data3d.get(proj_id)
print(projections.shape)
# 可视化投影数据
plt.imshow(projections[50], cmap='gray')
plt.colorbar()
plt.show()
# *********************************************保存为jpg图片数据
# output_dir = 'projections'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # 归一化投影数据
# normalized_projections = projections / np.max(projections)
#
# # 将每个投影保存为JPG文件
# for i, projection in enumerate(normalized_projections):
#     output_file = os.path.join(output_dir, f'projection_{i:03d}.jpg')
#     plt.imsave(output_file, projection, cmap='gray')

#  ***************************************保存为二进制文件
output_file = 'projections.npy'
np.save(output_file, projections)


# 清理
astra.algorithm.delete(alg_id)
astra.data3d.delete(proj_id)
astra.data3d.delete(vol_id)


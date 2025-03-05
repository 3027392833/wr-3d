def gaussian_kernel(kernelsize=3, sigma=1.0):
    """生成 2D 高斯核"""
    # 创建坐标网格
    x = torch.arange(-(kernelsize // 2), kernelsize // 2 + 1, dtype=torch.float32)
    y = torch.arange(-(kernelsize // 2), kernelsize // 2 + 1, dtype=torch.float32)
    x, y = torch.meshgrid(x, y, indexing='ij')

    # 计算高斯分布
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()  # 归一化
    return kernel


def GaussianSmooth(img, kernelsize=9, sigma=9):
    """高斯平滑"""
    # 生成高斯核
    kernel = gaussian_kernel(kernelsize, sigma)
    kernel = kernel.view(1, 1, kernelsize, kernelsize)  # 重塑为 [1, 1, kernelsize, kernelsize]

    # 扩展卷积核以匹配输入图像的通道数
    # in_channels = img.size(1)  # 获取输入图像的通道数
    # kernel = kernel.expand(in_channels, 1, kernelsize, kernelsize)  # 扩展为 [in_channels, 1, kernelsize, kernelsize]

    # 确保核张量与输入图像在相同的设备上
    kernel = kernel.to(img.device)

    # 对每个通道进行卷积
    low = F.conv2d(img, kernel,  padding=kernelsize // 2)
    high = img - low  # 高频部分
    return high

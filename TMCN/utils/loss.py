import torch
import torch.nn as nn
import torch.nn.functional as F



class Fusionloss(nn.Module):
    #这个 Fusionloss 类定义了一个联合的损失函数，该损失函数同时考虑了图像的亮度值和边缘信息，
    # 用于评估生成图像与输入图像之间的差异。
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        # mage_y = image_vis[:,:1,:,:]：从可见光图像中提取亮度信息。:,:1,:,: 用于获取通道维度的第一个通道（通常是灰度图像）
        x_in_max=torch.max(image_y,image_ir)
        # x_in_max = torch.max(image_y, image_ir)：计算了可见光图像亮度和红外图像亮度的最大值，
        # 用于计算输入图像的 L1 损失。即对于每个像素位置，选择两个输入图像中亮度更大的值。
        loss_in=F.l1_loss(x_in_max,generate_img)
        # 计算了基于最大亮度值的 L1 损失。这个损失衡量了生成的图像与输入图像中亮度值较大的部分之间的差异。
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        # 分别对可见光图像、红外图像和生成的图像进行 Sobel 边缘检测。
        x_grad_joint=torch.max(y_grad,ir_grad)
        # 计算了可见光图像和红外图像的 Sobel 梯度中的较大值。
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        # 基于边缘检测梯度的 L1 损失。衡量了生成的图像与输入图像中边缘信息（较大梯度值）之间的差异。
        loss_total=loss_in+10*loss_grad
        # 将亮度值损失和边缘梯度损失结合为总的损失。这里通过加权求和将两种损失相结合，权重为 10。
        return loss_total,loss_in,loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        # 这是一个包含了 Sobel 算子的垂直边缘检测的 3x3 卷积核。Sobel 算子是一种常用的图像处理算子，用于检测图像中的边缘。
        kernely = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        # 创建了一个 PyTorch 浮点型张量（FloatTensor），将 kernelx 列表转换为张量。这个操作将二维列表转换为 PyTorch 的浮点型张量，便于后续的张量操作。
        # unsqueeze(0).unsqueeze(0): 这是两次使用 unsqueeze 方法，将维度扩展到指定的位置。
        # unsqueeze(0) 将维度在索引 0 的位置上进行扩展，将原本的二维张量转换为一个维度为 1 的三维张量。
        # 第二次 unsqueeze(0) 再次在索引 0 处扩展了一个维度，
        # 使得现在的四维张量中第一个维度为 1（代表 batch 维度），第二个维度为 1（代表输入通道维度，通常为灰度图像或单通道图像），
        # 第三个和第四个维度分别是卷积核的高度和宽度。
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        # nn.Parameter 创建了一个可学习的参数（或权重），是模型可训练的一部分。
        # 在这个例子中，kernelx 是之前处理过的 Sobel 算子卷积核的四维张量（已经通过 .unsqueeze(0).unsqueeze(0) 转换为 PyTorch 需要的格式）

        # data=kernelx: 通过指定 data 参数，将 kernelx 设置为 nn.Parameter 的初始数据。
        # 这样做的目的是将 Sobel 算子卷积核作为权重初始化到这个 nn.Parameter 对象中

        # requires_grad=False，表示这个权重参数是不需要梯度更新的，也就是它在训练过程中不会被优化器所更新。
        # 在这个情况下，Sobel 算子作为固定的卷积核，用于对图像进行边缘检测，不需要通过反向传播更新权重
        # cuda(): 这个方法将 self.weightx 移动到 GPU 上进行计算。
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        # def forward(self, x):：这个函数定义了模型的前向传播过程。在 PyTorch 中，
        # 所有的模型都需要实现一个名为 forward 的方法，用于定义数据从输入到输出的传播过程。
        sobelx=F.conv2d(x, self.weightx, padding=1)
        # sobelx = F.conv2d(x, self.weightx, padding=1): 使用 PyTorch 的 F.conv2d 函数进行卷积操作。
        # x 是输入的特征图数据，self.weightx 是之前定义的 Sobel 算子卷积核的权重。
        # 这行代码执行了垂直方向（x 方向）的 Sobel 边缘检测卷积操作，并将结果保存在 sobelx 变量中。
        # padding=1 表示对输入进行了 1 个像素的填充，以保持卷积后特征图的大小不变。
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
        # 总的来说，这段代码定义了一个自定义的 PyTorch 模型，
        # 在 forward 方法中执行了垂直和水平方向的 Sobel 边缘检测卷积操作，
        # 并将两个方向上的边缘强度图像相加，得到最终的边缘检测结果。


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()
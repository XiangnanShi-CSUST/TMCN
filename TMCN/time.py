from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, MDetailFeatureExtraction, M2BaseFeatureExtraction, DetailFeatureExtraction
import torch
import torch.nn as nn
import time
import clip
# 计算 Total Params (M)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 计算 PyTorch Speed (FPS)
def calculate_fps(model, input_tensor, num_runs=100):
    # 预热 GPU（如果有）
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    # 设置模型为评估模式
    model.eval()

    # 推理多次以稳定时间测量
    start_time = time.time()
    with torch.no_grad():  # 禁用梯度计算
        for _ in range(num_runs):
            _ = model(input_tensor)

    # 计算平均推理时间
    elapsed_time = time.time() - start_time
    avg_time_per_run = elapsed_time / num_runs
    fps = 1 / avg_time_per_run

    return fps

# 主逻辑
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model_clip, _ = clip.load("ViT-B/32", device=device)

    # 初始化模型
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder(model_clip = model_clip)).to(device)
    BaseFuseLayer = nn.DataParallel(M2BaseFeatureExtraction(dim=64, hidden_dim=64, model_clip = model_clip)).to(device)
    DetailFuseLayer = nn.DataParallel(MDetailFeatureExtraction(num_layers=1, hidden_dim=64, model_clip = model_clip)).to(device)

    # 计算 Total Params (M)
    total_params = count_parameters(Encoder) + count_parameters(Decoder) + \
                   count_parameters(BaseFuseLayer) + count_parameters(DetailFuseLayer)
    print(f"Total Params: {total_params / 1e6:.2f} M")

    # 创建输入张量
    input_tensor = torch.randn(1, 1, 256, 256).to(device)  # 假设输入尺寸为 (1, 1, 256, 256)

    # 计算 FPS
    num_runs = 100  # 推理次数
    fps = calculate_fps(Encoder, input_tensor, num_runs)
    print(f"FPS: {fps:.2f}")
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image # 用于处理图像文件
import json # 用于加载标签映射

print("--- PyTorch 环境检查 ---")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
else:
    print("当前使用 CPU 进行计算。")

print("\n--- 1. 加载预训练的 ResNet50 模型 ---")
# 加载 ResNet50 预训练模型
# pretrained=True 会下载在 ImageNet 数据集上预训练好的权重
# 如果你的 GPU 可用，模型会被自动加载到 GPU 上
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
print("ResNet50 模型加载成功！")

# 将模型设置为评估模式（这会关闭 Dropout 等，确保预测结果一致）
model.eval()

# 将模型移动到可用的设备 (GPU 或 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"模型已移动到: {device}")

print("\n--- 2. 准备图像预处理转换 ---")
# ImageNet 数据集通常需要的预处理步骤：
# 1. 调整大小到 256x256
# 2. 从中心裁剪到 224x224
# 3. 转换为张量
# 4. 标准化 (根据 ImageNet 的均值和标准差)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("图像预处理转换已定义。")

print("\n--- 3. 加载一个示例图片并进行预测 ---")
# 为了演示，我们先加载一个示例图片。
# 你需要准备一张图片，例如一张猫或狗的图片，放在你的 dog_project 文件夹下，命名为 'test_image.jpg'
# 如果你没有图片，可以暂时跳过此步骤，或者从网上下载一张。

try:
    image_path = "test_image.jpg" # 确保这张图片在你的项目文件夹下
    input_image = Image.open(image_path)
    print(f"成功加载图片: {image_path}")

    # 对图片进行预处理
    input_tensor = preprocess(input_image)
    # 给张量添加一个批次维度（因为模型期望输入是 [批量大小, 通道数, 高, 宽]）
    input_batch = input_tensor.unsqueeze(0) # 形状变为 [1, 3, 224, 224]

    # 将输入张量移动到与模型相同的设备
    input_batch = input_batch.to(device)

    # 进行预测
    print("正在进行模型预测...")
    with torch.no_grad(): # 在预测时，我们不需要计算梯度，这可以节省内存和提高速度
        output = model(input_batch)

    # output 是一个包含 1000 个类别的 Logits 张量
    # 找到最高概率的类别索引
    _, predicted_idx = torch.max(output, 1)
    predicted_class_idx = predicted_idx.item() # 获取预测的类别索引 (0-999)

    print(f"预测的类别索引: {predicted_class_idx}")

    # 加载 ImageNet 1000 个类别的标签映射
    # 你需要下载这个文件：https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
    # 并将其保存为 'imagenet_classes.txt' 在你的 dog_project 文件夹下
    try:
        with open("imagenet_classes.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
        predicted_label = labels[predicted_class_idx]
        print(f"模型预测结果: {predicted_label}")
    except FileNotFoundError:
        print("未找到 'imagenet_classes.txt' 文件，无法显示类别名称。")
        print("请从 https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt 下载此文件并放到项目文件夹下。")

except FileNotFoundError:
    print(f"错误：未找到 {image_path}。请确保图片文件存在并放在当前目录下。")
except Exception as e:
    print(f"处理图片或预测时发生错误: {e}")

print("\n--- 脚本运行完毕 ---")
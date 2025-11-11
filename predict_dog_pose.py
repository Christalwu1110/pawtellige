import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

print("--- 狗狗姿态预测脚本 ---")

# --- 1. 定义设备 ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"模型将运行在: {device}")

# --- 2. 定义图像预处理转换 (与训练时的验证集转换保持一致) ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print("图像预处理转换已定义。")

# --- 3. 加载类别映射 ---
class_mapping_path = 'dog_pose_classes.txt'
class_names = []
if os.path.exists(class_mapping_path):
    with open(class_mapping_path, 'r') as f:
        for line in f:
            class_names.append(line.strip())
    num_classes = len(class_names)
    print(f"加载的类别: {class_names}")
else:
    print(f"错误: 未找到类别映射文件 '{class_mapping_path}'。请确保 'train_dog_pose.py' 已经运行并生成了该文件。")
    exit()

# --- 4. 加载 ResNet50 模型结构 ---
# 注意: 加载模型结构时需要与训练时保持一致
# 这里我们加载预训练权重，但重要的是它的结构，之后会被我们自己的权重覆盖
model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# 修改模型的全连接层以匹配你的类别数
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)

# --- 5. 加载保存的模型权重 (state_dict) ---
model_save_path = 'dog_pose_resnet50.pth'
if os.path.exists(model_save_path):
    # load_state_dict 默认严格匹配键名，如果模型在训练前后结构有微小变化需要调整
    model_ft.load_state_dict(torch.load(model_save_path, map_location=device))
    model_ft = model_ft.to(device) # 将模型移到设备上
    model_ft.eval() # 设置为评估模式 (非常重要，预测时必须设为 eval 模式)
    print(f"模型权重已从 '{model_save_path}' 加载成功！")
else:
    print(f"错误: 未找到模型权重文件 '{model_save_path}'。请确保 'train_dog_pose.py' 已经成功训练并保存了模型。")
    exit()

# --- 6. 定义预测函数 ---
def predict_pose(image_path, model, transform, class_names, device):
    """
    对单张图片进行姿态预测。
    """
    try:
        image = Image.open(image_path).convert('RGB') # 确保图片是RGB格式
        image = transform(image).unsqueeze(0).to(device) # 预处理并添加批次维度，移动到设备

        with torch.no_grad(): # 预测时不需要计算梯度
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1) # 转换为概率
            
            # 获取最高概率的预测结果
            max_prob, predicted_idx = torch.max(probabilities, 1)
            predicted_class = class_names[predicted_idx.item()]
            
            # 获取所有类别的概率
            all_probs = {class_names[i]: prob.item() for i, prob in enumerate(probabilities[0])}

        return predicted_class, max_prob.item(), all_probs

    except FileNotFoundError:
        print(f"错误: 图片文件 '{image_path}' 未找到。")
        return None, None, None
    except Exception as e:
        print(f"处理图片 '{image_path}' 时发生错误: {e}")
        return None, None, None

# --- 7. 进行预测 ---
# 请将 'path/to/your/test_dog_image.jpg' 替换为你要预测的实际图片路径
# 你可以使用之前收集的任何一张图片，或者新下载一张。
# 确保这张图片不在你的训练集或验证集中，这样才能真实反映模型的泛化能力。
test_image_path = 'test_image.jpg' # 假设你的 test_image.jpg 在 dog_project 目录下

print(f"\n正在预测图片: {test_image_path}")
predicted_class, confidence, all_probabilities = predict_pose(test_image_path, model_ft, preprocess, class_names, device)

if predicted_class:
    print(f"\n预测姿态: {predicted_class}")
    print(f"置信度: {confidence:.4f}")
    print("\n所有姿态的预测概率:")
    for cls, prob in all_probabilities.items():
        print(f"  {cls}: {prob:.4f}")
else:
    print("未能完成预测。请检查图片路径和文件。")

print("\n--- 脚本运行完毕 ---")

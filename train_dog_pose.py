import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import time
import copy

print("--- PyTorch 环境检查 ---")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0") # 使用第一个GPU
else:
    print("当前使用 CPU 进行计算。")
    device = torch.device("cpu")

print(f"\n模型将运行在: {device}")

# --- 1. 定义数据转换 (Data Transforms) ---
# 数据增强 (Data Augmentation) 是为了增加训练数据的多样性，防止过拟合
# 训练集：进行随机裁剪、水平翻转等增强操作
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # 随机裁剪并缩放到 224x224
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.RandomRotation(degrees=15), # 随机旋转，最大旋转15度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 随机调整亮度、对比度、饱和度、色相
        transforms.ToTensor(),             # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 标准化
    ]),
    # 验证集：只进行调整大小、中心裁剪和标准化，不进行随机增强
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- 2. 加载数据集 ---
# 数据集根目录，假设你的图片在 dog_project/data/ 目录下
data_dir = './data'

# 使用 ImageFolder 加载整个数据集，它会自动根据子文件夹名称识别类别
# ⚠️ 注意：这里先用训练集的 transforms，因为 ImageFolder 只能接受一个 transform
# 我们会在后续划分数据集时为 train_dataset 和 val_dataset 重新设置 transform
full_dataset = datasets.ImageFolder(data_dir, data_transforms['train'])

# 获取类别名称和类别到索引的映射
class_names = full_dataset.classes
class_to_idx = full_dataset.class_to_idx
num_classes = len(class_names)

print(f"\n识别到的类别: {class_names}")
print(f"类别到索引的映射: {class_to_idx}")
print(f"总共 {num_classes} 个类别。")

# --- 3. 划分训练集和验证集 ---
# 这里我们将整个数据集按 80% 训练集，20% 验证集进行划分
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 重要的修正：为训练集和验证集应用各自的 transform
# random_split 只是划分了索引，Dataset 对象本身共享同一个底层数据源。
# 为了让它们使用不同的 transform (特别是验证集不应该有随机增强)，
# 我们需要创建两个新的 Dataset 对象，或更巧妙地处理。

# 最简洁的方式是，在 DataLoader 中对这些子集应用 transform
# 但由于 random_split 返回的是 Subset 类型，它们不直接支持 transform 属性。
# 让我们使用一个辅助函数来应用正确的 transform。

class DatasetWithTransforms(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# 使用新的辅助类，将正确的 transform 应用到划分后的子集上
train_dataset_transformed = DatasetWithTransforms(train_dataset, data_transforms['train'])
val_dataset_transformed = DatasetWithTransforms(val_dataset, data_transforms['val'])


# 创建数据加载器 (DataLoader)
# DataLoader 会批量加载数据，方便训练
dataloaders = {
    'train': DataLoader(train_dataset_transformed, batch_size=32, shuffle=True, num_workers=4), # num_workers 可以根据你的CPU核心数调整
    'val': DataLoader(val_dataset_transformed, batch_size=32, shuffle=False, num_workers=4)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

print(f"训练集大小: {dataset_sizes['train']}")
print(f"验证集大小: {dataset_sizes['val']}")
print("数据加载器已准备完毕。")

# --- 3. 划分训练集和验证集 ---
# 这里我们将整个数据集按 80% 训练集，20% 验证集进行划分
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 为训练集和验证集应用各自的transform (重要!)
# random_split 只是划分了索引，没有应用transform。
# 需要重新创建Dataset来应用不同的transform
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']


# 创建数据加载器 (DataLoader)
# DataLoader 会批量加载数据，方便训练
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4), # num_workers 可以根据你的CPU核心数调整
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

print(f"训练集大小: {dataset_sizes['train']}")
print(f"验证集大小: {dataset_sizes['val']}")
print("数据加载器已准备完毕。")

# --- 4. 加载并修改预训练模型 ---
print("\n--- 4. 加载并修改预训练模型 ---")
# 加载 ResNet50 预训练模型
model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# 获取 ResNet50 的全连接层 (fc) 的输入特征数量
num_ftrs = model_ft.fc.in_features

# 替换掉原来的全连接层，使其输出我们定义的类别数量
# 这是一个新的全连接层，它的权重是随机初始化的，需要从头开始学习
model_ft.fc = nn.Linear(num_ftrs, num_classes)

# 将模型移动到 GPU 或 CPU
model_ft = model_ft.to(device)
print("ResNet50 模型加载并修改成功！")
print(f"模型最终分类层输出特征数: {num_classes}")

# --- 5. 定义损失函数和优化器 ---
criterion = nn.CrossEntropyLoss()

# 冻结 ResNet50 大部分参数的梯度
for param in model_ft.parameters():
    param.requires_grad = False
model_ft.fc.weight.requires_grad = True
model_ft.fc.bias.requires_grad = True

# 优化器：Adam 优化器
# 调整学习率从 0.001 到 0.0005
optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=0.005) # 学习率调小

# 学习率调度器 (保持不变)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print("损失函数、优化器和学习率调度器已定义。")
# --- 6. 定义训练函数 ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time() # 记录开始时间

    best_model_wts = copy.deepcopy(model.state_dict()) # 存储最佳模型的权重
    best_acc = 0.0 # 记录最佳验证集准确率

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个 epoch 都有一个训练阶段和一个验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历每个批次的数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device) # 将输入数据移动到GPU/CPU
                labels = labels.to(device) # 将标签移动到GPU/CPU

                # 清零梯度 (重要步骤!)
                # 每次反向传播前都需要清零梯度，否则梯度会累加
                optimizer.zero_grad()

                # 前向传播
                # 仅在训练阶段启用梯度计算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) # 模型前向计算
                    _, preds = torch.max(outputs, 1) # 获取预测的类别
                    loss = criterion(outputs, labels) # 计算损失

                    # 后向传播 + 优化 (仅在训练阶段)
                    if phase == 'train':
                        loss.backward() # 反向传播，计算梯度
                        optimizer.step() # 根据梯度更新模型参数

                # 统计损失和正确预测的数量
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 在训练阶段结束后，更新学习率
            if phase == 'train':
                scheduler.step()

            # 计算当前 epoch 的平均损失和准确率
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度复制最佳模型 (基于验证集准确率)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) # 存储当前最佳模型权重

        print() # 每个 epoch 结束后打印空行

    time_elapsed = time.time() - since
    print(f'训练完成，耗时 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最佳验证集准确率: {best_acc:.4f}')

    # 加载最佳模型的权重
    model.load_state_dict(best_model_wts)
    return model

# --- 7. 开始训练 ---
print("\n--- 7. 开始训练模型 ---")
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=35) # 暂定训练10个epoch

# --- 8. 保存训练好的模型 ---
model_save_path = 'dog_pose_resnet50.pth' # 定义模型保存路径和文件名

# 推荐保存模型的 state_dict（只保存学习到的参数，文件更小）
torch.save(model_ft.state_dict(), model_save_path)
print(f"模型已保存到: {model_save_path}")

# --- 9. 将类别到索引的映射保存为文件 ---
# 为了预测时能知道模型预测的数字代表哪个姿态，我们需要保存这个映射
idx_to_class = {v: k for k, v in class_to_idx.items()}
class_mapping_path = 'dog_pose_classes.txt'
with open(class_mapping_path, 'w') as f:
    for idx in range(num_classes):
        f.write(f"{idx_to_class[idx]}\n")
print(f"类别映射已保存到: {class_mapping_path}")


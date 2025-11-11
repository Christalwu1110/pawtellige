import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

def grad_cam_plus_plus(model, image_path, target_class, output_path='gradcam_effnet.jpg'):
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0)

    # 初始化GradCAM++
    target_layers = [model._conv_head]  # EfficientNet-B0的最后一层卷积
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    # 生成热力图
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_class)
    grayscale_cam = grayscale_cam[0, :]  # 取第一个样本的热力图

    # 预处理原始图像以叠加热力图
    rgb_img = np.float32(image.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 保存结果
    cv2.imwrite(output_path, cam_image)
    return output_path

# 使用示例
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = torch.nn.Linear(model._fc.in_features, 5)  # 5个类别
    model.load_state_dict(torch.load('dog_pose_efficientnet_b0.pth', map_location=device))
    model = model.to(device)
    model.eval()

    image_path = '/home/yeling/dog_project/key_frames/frame_750.jpg'
    target_class = 3  # 假设sitting=3，需根据实际类别顺序确认
    output_path = grad_cam_plus_plus(model, image_path, target_class)
    print(f"Grad-CAM++结果已保存至: {output_path}")
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

class CourtLineDetector:
    def __init__(self, model_path):
        # 初始化一个预训练的 ResNet50 模型，但不加载预训练权重
        self.model = models.resnet50(pretrained=False)
        # 修改模型的最后一层全连接层，输出14个点的坐标（x,y），共28个值
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        # 加载训练好的模型权重
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        # 定义图像预处理转换流程
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # 将 numpy 数组转换为 PIL 图像
            transforms.Resize((224, 224)),  # 调整图像大小为 224x224
            transforms.ToTensor(),  # 转换为张量并归一化到 [0,1]
            transforms.Normalize(  # 标准化处理，使用 ImageNet 的均值和标准差
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image):
        # 将 BGR 格式转换为 RGB 格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 对图像进行预处理并添加批次维度
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        # 不计算梯度，进行推理
        with torch.no_grad():
            outputs = self.model(image_tensor)
        # 将输出转换为 numpy 数组
        keypoints = outputs.squeeze().cpu().numpy()
        # 获取原始图像的高度和宽度
        original_h, original_w = image.shape[:2]
        # 将关键点坐标从 224x224 缩放到原始图像大小
        keypoints[::2] *= original_w / 224.0  # x 坐标
        keypoints[1::2] *= original_h / 224.0  # y 坐标

        return keypoints

    def draw_keypoints(self, image, keypoints):
        # 在图像上绘制关键点
        for i in range(0, len(keypoints), 2):
            # 获取 x, y 坐标
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            # 在关键点上方绘制序号
            cv2.putText(image, str(i//2), (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # 在关键点位置绘制红色圆点
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        # 处理视频的每一帧
        output_video_frames = []
        for frame in video_frames:
            # 在每一帧上绘制关键点
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
# 定义目标检测的参数
OBJ_THRESH = 0.25  # 目标检测的置信度阈值
NMS_THRESH = 0.45  # 非极大值抑制（NMS）的阈值
IMG_SIZE = (640, 640)  # 输入图像的尺寸（宽，高）

# 感兴趣范围坐标
CENTER_X = 320
CENTER_Y = 320
WIDTH = 640
HEIGHT = 640

X1 = CENTER_X - WIDTH / 2
X2 = CENTER_X + WIDTH / 2
Y1 = CENTER_Y - HEIGHT / 2
Y2 = CENTER_Y + HEIGHT / 2


# 选择是否输出预测框坐标
OUTPUT_PREDICTION_BOX_COORDINATES = False


# 检测类别顺序列表
CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
)

# COCO 数据集的类别 ID 列表（与类别顺序对应）
coco_id_list = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79
]

import numpy as np
import cv2
import config
from py_utils.coco_utils import COCO_test_helper  # 导入 COCO 测试辅助工具
co_helper = COCO_test_helper(enable_letter_box=True)

def filter_boxes(boxes, box_confidences, box_class_probs):
    """
    根据置信度阈值过滤目标检测框。
    :param boxes: 检测框的坐标
    :param box_confidences: 检测框的置信度
    :param box_class_probs: 检测框的类别概率
    :return: 过滤后的检测框、类别和置信度
    """
    box_confidences = box_confidences.reshape(-1)  # 将置信度展平
    candidate, class_num = box_class_probs.shape  # 获取候选框数量和类别数量

    class_max_score = np.max(box_class_probs, axis=-1)  # 获取每个候选框的最大类别概率
    classes = np.argmax(box_class_probs, axis=-1)  # 获取每个候选框的类别

    # 找出满足置信度阈值的检测框
    _class_pos = np.where(class_max_score * box_confidences >= config.OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]  # 计算最终得分

    boxes = boxes[_class_pos]  # 筛选检测框
    classes = classes[_class_pos]  # 筛选类别

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """
    非极大值抑制（NMS），用于去除冗余的检测框。
    :param boxes: 检测框的坐标
    :param scores: 检测框的得分
    :return: 保留的检测框索引
    """
    x = boxes[:, 0]  # 检测框的左上角 x 坐标
    y = boxes[:, 1]  # 检测框的左上角 y 坐标
    w = boxes[:, 2] - boxes[:, 0]  # 检测框的宽度
    h = boxes[:, 3] - boxes[:, 1]  # 检测框的高度

    areas = w * h  # 计算检测框的面积
    order = scores.argsort()[::-1]  # 按得分降序排列检测框

    keep = []  # 用于存储保留的检测框索引
    while order.size > 0:
        i = order[0]  # 当前得分最高的检测框
        keep.append(i)  # 保留当前检测框

        # 计算当前检测框与其他检测框的交集
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)  # 交集的宽度
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)  # 交集的高度
        inter = w1 * h1  # 交集的面积

        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # 计算重叠率
        inds = np.where(ovr <= config.NMS_THRESH)[0]  # 找出重叠率小于阈值的检测框
        order = order[inds + 1]  # 更新检测框顺序

    keep = np.array(keep)  # 将保留的检测框索引转换为数组
    return keep


def dfl(position):
    """
    Distribution Focal Loss (DFL) 的实现，用于处理位置预测。
    :param position: 位置预测的输入
    :return: 处理后的结果
    """
    import torch
    x = torch.tensor(position)  # 将输入转换为张量
    n, c, h, w = x.shape  # 获取输入的形状
    p_num = 4  # 每个位置的预测数量
    mc = c // p_num  # 每个位置的类别数量
    y = x.reshape(n, p_num, mc, h, w)  # 重塑输入
    y = y.softmax(2)  # 应用 softmax 函数
    acc_matrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)  # 创建累加矩阵
    y = (y * acc_matrix).sum(2)  # 计算加权和
    return y.numpy()  # 返回处理后的结果


def box_process(position):
    """
    处理检测框的位置信息。
    :param position: 位置预测的输入
    :return: 处理后的检测框坐标
    """
    grid_h, grid_w = position.shape[2:4]  # 获取网格的高度和宽度
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))  # 创建网格坐标
    col = col.reshape(1, 1, grid_h, grid_w)  # 重塑列坐标
    row = row.reshape(1, 1, grid_h, grid_w)  # 重塑行坐标
    grid = np.concatenate((col, row), axis=1)  # 将列和行坐标合并为网格
    stride = np.array([config.IMG_SIZE[1] // grid_h, config.IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)  # 计算步长

    position = dfl(position)  # 应用 DFL 处理
    box_xy = grid + 0.5 - position[:, 0:2, :, :]  # 计算检测框的左上角坐标
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]  # 计算检测框的右下角坐标
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)  # 将坐标转换为 (x1, y1, x2, y2) 格式

    return xyxy


def post_process(input_data):
    """
    对模型的输出进行后处理，包括解码检测框、过滤和 NMS。
    :param input_data: 模型的输出数据
    :return: 处理后的检测框、类别和得分
    """
    boxes, scores, classes_conf = [], [], []  # 初始化检测框、得分和类别概率的列表
    default_branch = 3  # 默认的分支数量
    pair_per_branch = len(input_data) // default_branch  # 每个分支的对数

    for i in range(default_branch):  # 遍历每个分支
        boxes.append(box_process(input_data[pair_per_branch * i]))  # 处理检测框
        classes_conf.append(input_data[pair_per_branch * i + 1])  # 获取类别概率
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))  # 创建得分数组

    def sp_flatten(_in):
        """
        将输入数据展平。
        :param _in: 输入数据
        :return: 展平后的数据
        """
        ch = _in.shape[1]  # 获取通道数
        _in = _in.transpose(0, 2, 3, 1)  # 调整维度顺序
        return _in.reshape(-1, ch)  # 展平为二维数组

    boxes = [sp_flatten(_v) for _v in boxes]  # 展平检测框
    classes_conf = [sp_flatten(_v) for _v in classes_conf]  # 展平类别概率
    scores = [sp_flatten(_v) for _v in scores]  # 展平得分

    boxes = np.concatenate(boxes)  # 合并检测框
    classes_conf = np.concatenate(classes_conf)  # 合并类别概率
    scores = np.concatenate(scores)  # 合并得分

    # 根据置信度阈值过滤检测框
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # 应用 NMS
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):  # 遍历每个类别
        inds = np.where(classes == c)  # 找出当前类别的索引
        b = boxes[inds]  # 当前类别的检测框
        c = classes[inds]  # 当前类别的类别
        s = scores[inds]  # 当前类别的得分
        keep = nms_boxes(b, s)  # 应用 NMS

        if len(keep) != 0:  # 如果有保留的检测框
            nboxes.append(b[keep])  # 添加检测框
            nclasses.append(c[keep])  # 添加类别
            nscores.append(s[keep])  # 添加得分

    if not nclasses and not nscores:  # 如果没有检测到任何目标
        return None, None, None

    boxes = np.concatenate(nboxes)  # 合并检测框
    classes = np.concatenate(nclasses)  # 合并类别
    scores = np.concatenate(nscores)  # 合并得分

    return boxes, classes, scores


def draw(image, boxes, scores, classes):
    """
    在图像上绘制检测框和类别信息。
    :param image: 输入图像
    :param boxes: 检测框的坐标
    :param scores: 检测框的得分
    :param classes: 检测框的类别
    """
    for box, score, cl in zip(boxes, scores, classes):  # 遍历每个检测框
        top, left, right, bottom = [int(_b) for _b in box]  # 获取检测框的坐标
        # print("%s @ (%d %d %d %d) %.3f" % (config.CLASSES[cl], top, left, right, bottom, score))  # 打印检测框信息
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)  # 绘制检测框
        cv2.putText(image, '{0} {1:.2f}'.format(config.CLASSES[cl], score),  # 绘制类别和得分
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def setup_model(args):
    """
    根据输入参数初始化模型。
    :param args: 输入参数
    :return: 初始化的模型和平台类型
    """
    model_path = args.model_path  # 获取模型路径
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):  # 如果是 PyTorch 模型
        platform = 'pytorch'
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)  # 初始化 PyTorch 模型
    elif model_path.endswith('.rknn'):  # 如果是 RKNN 模型
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container
        model = RKNN_model_container(args.model_path, args.target, args.device_id)  # 初始化 RKNN 模型
    elif model_path.endswith('.onnx'):  # 如果是 ONNX 模型
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)  # 初始化 ONNX 模型
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)  # 如果不是支持的模型类型，抛出异常

    print('Model-{} is {} model, starting val'.format(model_path, platform))  # 打印模型类型
    return model, platform


def img_check(path):
    """
    检查文件是否是图像文件。
    :param path: 文件路径
    :return: 如果是图像文件返回 True，否则返回 False
    """
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']  # 支持的图像文件类型
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):  # 检查文件扩展名
            return True
    return False

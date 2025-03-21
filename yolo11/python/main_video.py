import os
import sys
import cv2
import time
import argparse
import numpy as np

import func
import config

# 添加项目路径到系统路径，以便导入自定义模块
realpath = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
_sep = os.path.sep  # 获取路径分隔符
realpath = realpath.split(_sep)  # 将路径分割为列表
sys.path.append(os.path.join(realpath[0] + _sep, *realpath[1:realpath.index('rknn_model_zoo') + 1]))  # 添加项目路径

from py_utils.coco_utils import COCO_test_helper  # 导入 COCO 测试辅助工具


cap = cv2.VideoCapture('./720p60hz.mp4')

if __name__ == '__main__':

    args = argparse.Namespace(
        model_path='../model/yolo11.rknn',
        target='rk3588',
        device_id=None,
        img_show=False,
        img_save=True,
        anno_json='../../../datasets/COCO/annotations/instances_val2017.json',
        img_folder='../model',
        coco_map_test=False
    )

    # 初始化模型
    model, platform = func.setup_model(args)  # 根据参数加载模型

    # 初始化 COCO 测试辅助工具
    co_helper = COCO_test_helper(enable_letter_box=True)


    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        img_src = frame  # 读取图像

        # 使用 Letter Box 预处理图像
        pad_color = (0, 0, 0)  # 填充颜色
        img = co_helper.letter_box(im=img_src.copy(), new_shape=(config.IMG_SIZE[1], config.IMG_SIZE[0]), pad_color=pad_color)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像从 BGR 转换为 RGB

        # 针对不同模型进行预处理
        if platform in ['pytorch', 'onnx']:  # 如果是 PyTorch 或 ONNX 模型
            input_data = img.transpose((2, 0, 1))  # 调整维度顺序为 (C, H, W)
            input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)  # 添加批量维度并转换为 float32
            input_data = input_data / 255.  # 归一化到 [0, 1]
        else:  # 如果是 RKNN 模型
            input_data = img  # RKNN 模型不需要额外预处理

        # 运行模型推理
        outputs = model.run([input_data])  # 获取模型输出
        boxes, classes, scores = func.post_process(outputs)  # 对输出进行后处理

        img_p = img_src.copy()  # 复制原始图像用于绘制
        if boxes is not None:  # 如果检测到目标
            func.draw(img_p, co_helper.get_real_box(boxes), scores, classes)  # 在图像上绘制检测框和类别信息

        cv2.imshow("full post process result", img_p)  # 显示图像
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 退出
            break

    # 如果启用 COCO mAP 测试，计算并输出 mAP
    if args.coco_map_test:
        pred_json = args.model_path.split('.')[-2] + '_{}'.format(platform) + '.json'  # 定义预测结果文件名
        pred_json = pred_json.split('/')[-1]  # 获取文件名
        pred_json = os.path.join('./', pred_json)  # 定义保存路径
        co_helper.export_to_json(pred_json)  # 将检测结果导出为 JSON 文件

        from py_utils.coco_utils import coco_eval_with_json  # 导入 COCO 评估工具
        coco_eval_with_json(args.anno_json, pred_json)  # 使用 COCO 标注文件和预测文件计算 mAP

    # 释放模型资源
    model.release()
    cap.release()
    cv2.destroyAllWindows()
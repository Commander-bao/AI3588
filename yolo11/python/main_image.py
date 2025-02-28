import os
import cv2
import sys
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

    # 获取图像文件夹中的所有文件
    file_list = sorted(os.listdir(args.img_folder))  # 获取文件夹中的所有文件并排序
    img_list = []  # 初始化图像文件列表
    for path in file_list:
        if func.img_check(path):  # 检查是否为图像文件
            img_list.append(path)  # 将图像文件添加到列表中

    # 初始化 COCO 测试辅助工具
    co_helper = COCO_test_helper(enable_letter_box=True)

    # 遍历图像列表进行推理
    for i in range(len(img_list)):
        print('infer {}/{}'.format(i + 1, len(img_list)), end='\r')  # 打印当前进度

        img_name = img_list[i]  # 获取当前图像文件名
        img_path = os.path.join(args.img_folder, img_name)  # 获取当前图像的完整路径
        if not os.path.exists(img_path):  # 检查图像文件是否存在
            print("{} is not found".format(img_name))
            continue

        img_src = cv2.imread(img_path)  # 读取图像
        if img_src is None:  # 如果图像无法读取，则跳过
            continue

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

        # 绘制结果并显示/保存
        if args.img_show or args.img_save:
            print('\n\nIMG: {}'.format(img_name))  # 打印当前图像名称
            img_p = img_src.copy()  # 复制原始图像用于绘制
            if boxes is not None:  # 如果检测到目标
                func.draw(img_p, co_helper.get_real_box(boxes), scores, classes)  # 在图像上绘制检测框和类别信息

            if args.img_save:  # 如果需要保存结果
                if not os.path.exists('./result'):  # 如果结果文件夹不存在，则创建
                    os.mkdir('./result')
                result_path = os.path.join('./result', img_name)  # 定义保存路径
                cv2.imwrite(result_path, img_p)  # 保存图像
                print('Detection result save to {}'.format(result_path))

            if args.img_show:  # 如果需要显示结果
                cv2.imshow("full post process result", img_p)  # 显示图像
                cv2.waitKeyEx(0)  # 等待按键

        # 如果启用 COCO mAP 测试，记录检测结果
        if args.coco_map_test:
            if boxes is not None:
                for i in range(boxes.shape[0]):  # 遍历每个检测框
                    co_helper.add_single_record(
                        image_id=int(img_name.split('.')[0]),  # 图像 ID
                        category_id=config.coco_id_list[int(classes[i])],  # 类别 ID
                        bbox=boxes[i],  # 检测框坐标
                        score=round(scores[i], 5).item()  # 得分
                    )

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
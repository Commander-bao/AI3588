# 环境部署

## 主机端克隆官方Yolo仓库&\&Anaconda环境创建

这里需要使用瑞芯微官方修改过的yolo11

    git clone git@github.com:airockchip/ultralytics_yolo11.git
    cd ultralytics_yolo11 && touch requirements.txt # 创建依赖包清单

    conda create -y -n yolo11 python=3.10
    conda activate yolo11

requirements.txt文件内容如下

    # Ultralytics requirements
    # Usage: pip install -r requirements.txt

    # Base ----------------------------------------
    matplotlib>=3.2.2
    numpy>=1.22.2 # pinned by Snyk to avoid a vulnerability
    opencv-python>=4.6.0
    pillow>=7.1.2
    pyyaml>=5.3.1
    requests>=2.23.0
    scipy>=1.4.1
    torch>=1.7.0
    torchvision>=0.8.1
    tqdm>=4.64.0

    # Logging -------------------------------------
    # tensorboard>=2.13.0
    # dvclive>=2.12.0
    # clearml
    # comet

    # Plotting ------------------------------------
    pandas>=1.1.4
    seaborn>=0.11.0

    # Export --------------------------------------
    # coremltools>=7.0.b1  # CoreML export
    onnx>=1.12.0  # ONNX export
    onnxsim>=0.4.1  # ONNX simplifier
    # nvidia-pyindex  # TensorRT export
    # nvidia-tensorrt  # TensorRT export
    # scikit-learn==0.19.2  # CoreML quantization
    # tensorflow>=2.4.1  # TF exports (-cpu, -aarch64, -macos)
    # tflite-support
    # tensorflowjs>=3.9.0  # TF.js export
    # openvino-dev>=2023.0  # OpenVINO export

    # Extras --------------------------------------
    psutil  # system utilization
    py-cpuinfo  # display CPU info
    # thop>=0.1.1  # FLOPs computation
    # ipython  # interactive notebook
    # albumentations>=1.0.3  # training augmentations
    # pycocotools>=2.0.6  # COCO mAP
    # roboflow

创建好文件后运行下面命令安装依赖包

    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

卸载CPU版本pytorch并到pytorch官网[Start Locally | PyTorch](https://pytorch.org/get-started/locally/)安装CUDA版本的pytorch

    pip uninstall torch torchvision

    # 添加镜像源
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    conda config --set show_channel_urls yes
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/

    conda install pytorch torchvision torchaudio pytorch-cuda=版本 -c nvidia

    # 或使用下面命令
    # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

## 板端Anaconda环境创建

安装Anaconda

    cd ~
    wget --user-agent="Mozilla" https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.10-1-Linux-aarch64.sh
    sh Anaconda3-2024.10-1-Linux-aarch64.sh
    # 如果不能使用conda命令，在环境变量最后加上
    nano ~/.bashrc
    export PATH=/home/orangepi/anaconda3/bin:$PATH
    source ~/.bashrc

创建yolo11环境并激活

    conda create -y -n yolo11 python=3.10
    conda activate yolo11

# 模型训练

更新中

# 模型转换

## 板端下载官方预转换的ONNX模型（可选）

    # yolo11n
    wget https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolo11/yolo11n.onnx
    # yolo11s
    wget https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolo11/yolo11s.onnx
    # yolo11m
    wget https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolo11/yolo11m.onnx

## 主机端使用python代码导出

在主机创建文件pt2onnx.py，内容如下，运行即可转换

    from ultralytics import YOLO
    model = YOLO("best.pt")
    model.export(format='rknn') # 实际导出为onnx格式

# 模型部署

## 板端升级Cmake版本

    conda install -c conda-forge cmake=3.25.*

## 板端安装rknn-toolkit（2.3.0版本）

    git clone --branch v2.3.0 git@github.com:airockchip/rknn-toolkit2.git
    cd rknn-toolkit2/packages/arm64

    conda activate yolo11
    pip install -r arm64_requirements_cp310.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install rknn_toolkit2-2.3.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

    cd ~/rknn-toolkit2/rknn-toolkit-lite2/packages
    pip install rknn_toolkit_lite2-2.3.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

    # 拷贝对应文件
    cd ~/rknn-toolkit2/rknpu2/runtime/Linux/rknn_server/aarch64/usr/bin
    sudo cp * /usr/bin/
    cd ~/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64
    sudo cp * /usr/lib/


注意！务必检查安装的torch版本是否等于1.13.1，版本不对会报错

    # 若版本不对，使用下面命令重新安装torch
    pip install torch==1.13.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

## 板端克隆官方Model仓库

    git clone git@github.com:airockchip/rknn_model_zoo.git

## 板端ONNX转RKNN模型

    # 将之前的ONNX模型放在rknn_model_zoo/examples/yolo11/model
    cd rknn_model_zoo/examples/yolo11/python
    python convert.py ../model/best.onnx rk3588 # model文件夹会得到yolo11.rknn

## 板端运行示例代码

    python yolo11.py --model_path ../model/yolo11.rknn --target rk3588 --img_show

# 其他

## 查看NPU驱动版本

    cat /sys/kernel/debug/rknpu/driver_version

## 查看实时NPU占用

    sudo watch -n 1 "cat /sys/kernel/debug/rknpu/load"


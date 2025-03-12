# 环境部署

## 主机端WSL2克隆官方仓库&\&Anaconda环境创建

克隆RKLLM库

    cd ~
    git clone --branch release-v1.1.4 git@github.com:airockchip/rknn-llm.git

安装Anaconda

    wget --user-agent="Mozilla"  https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.10-1-Linux-x86_64.sh

    bash ./Anaconda3-2024.10-1-Linux-x86_64.sh

创建rkllm环境并激活

    conda create -y -n rkllm python=3.10
    conda activate rkllm

安装rkllm-toolkit

    pip install ~/rknn-llm/rkllm-toolkit/packages/rkllm_toolkit-1.1.4-cp310-cp310-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

安装gcc交叉编译工具链

    cd ~
    wget https://developer.arm.com/downloads/-/gnu-a/10-2-2020-11/-/media/Files/downloads/gnu-a/10.2-2020.11/binrel/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu.tar.xz?revision=972019b5-912f-4ae6-864a-f61f570e2e7e&rev=972019b5912f4ae6864af61f570e2e7e&hash=A973F165C6D012E0738F90FB4A0C2BA7
    tar -xvf gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu.tar.xz

安装cmake

    sudo apt-get install -y cmake

编译运行程序

    # 将仓库中的llm_demo.cpp文件替换rknn-llm/examples/rkllm_api_demo/src/llm_demo.cpp
    # 将仓库中的build-linux.sh文件替换rknn-llm/examples/rkllm_api_demo/build-linux.sh
    cd ~/rknn-llm/examples/rkllm_api_demo/
    sudo chmod +x build-linux.sh && ./build-linux.sh

## 板端Anaconda环境创建

安装Anaconda

    cd ~
    wget --user-agent="Mozilla" https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.10-1-Linux-aarch64.sh
    sh Anaconda3-2024.10-1-Linux-aarch64.sh
    # 如果不能使用conda命令，在环境变量最后加上
    nano ~/.bashrc
    export PATH=/home/orangepi/anaconda3/bin:$PATH
    source ~/.bashrc

创建rkllm环境并激活

    conda create -y -n rkllm python=3.10
    conda activate rkllm

# 模型转换

## DeepSeek-R1模型下载

    # hugging face https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d
    git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    # 魔搭社区 https://www.modelscope.cn/collections/DeepSeek-R1-c8e86ac66ed943
    pip install modelscope
    modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

## 主机端WSL2模型转换

转换前需要先安装CUDA

[在 WSL 2 上启用 NVIDIA CUDA | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/ai/directml/gpu-cuda-in-wsl)

    # 将仓库中的convert.py复制到~/rknn-llm/rkllm-toolkit/examples
    cd ~/rknn-llm/rkllm-toolkit/examples
    conda activate rkllm
    python convert.py # 使用前可修改文件自定义配置

# 模型部署

## 板端本地部署

    cd ~ && mkdir rkllm/lib && cd rkllm && conda activate rkllm
    # 将~/rknn-llm/examples/rkllm_api_demo/build/build_linux_aarch64_Release/目录下的llm_demo上传至板端rkllm文件夹内
    # 将~/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64目录下的librkllmrt.so上传至板端rkllm/lib文件夹内

运行tree命令后结果如下

    rkllm/
    ├── deepseek-r1-1.5b-w8a8.rkllm
    ├── lib
    │   └── librkllmrt.so
    └── llm_demo

添加环境变量

    # 每次运行时使用下面命令
    export LD_LIBRARY_PATH=./lib
    # 或配置到文件中
    nano ~/.bashrc
    # 在最后添加
    export LD_LIBRARY_PATH=~/rkllm/lib
    # 保存退出
    source ~/.bashrc

板端本地运行

    ./llm_demo ./deepseek-r1-1.5b-w8a8.rkllm 1024 1024

若查看RKLLM板端推理性能可以运行下面命令

    export RKLLM_LOG_LEVEL=1

## 板端服务部署

    cd ~ && mkdir rkllm_server/lib && cd rkllm_server && conda activate rkllm
    # 将~/rknn-llm/examples/rkllm_server_demo/rkllm_server目录下的flask_server.py和gradio_server.py上传至板端rkllm_server文件夹内，可以将仓库中的gradio_server.py替换官方的文件使用
    # 将~/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64目录下的librkllmrt.so上传至板端rkllm_server/lib文件夹内

运行tree命令后结果如下

    rkllm/
    ├── lib
    │   └── librkllmrt.so
    ├── flask_server.py
    └── gradio_server.py

安装gradio包

    pip install gradio

板端服务运行

    python3 gradio_server.py --target_platform rk3588 --rkllm_model_path ~/rkllm/deepseek-r1-1.5b-w8a8.rkllm
    # 打开同一局域网下的浏览器，输入http://开发板ip地址:8080进行访问

from rkllm.api import RKLLM

modelpath = 'DeepSeek-R1-Distill-Qwen-1.5B'
# 初始化RKLLM对象
llm = RKLLM()

# 模型加载
ret = llm.load_huggingface(model=modelpath, model_lora = None, device='cuda')
if ret != 0:
    print('Load model failed!')
    exit(ret)

# 模型的量化构建
ret = llm.build(do_quantization=True, optimization_level=1, quantized_dtype='w8a8',
                quantized_algorithm='normal', target_platform='rk3588', num_npu_core=3)
#ret = llm.build(do_quantization=True, optimization_level=1, quantized_dtype='w8a8', target_platform='rk3588')
if ret != 0:
    print('Build model failed!')
    exit(ret)

# 导出rkllm模型
ret = llm.export_rkllm("./deepseek-r1-1.5b-w8a8.rkllm")
if ret != 0:
    print('Export model failed!')
    exit(ret)
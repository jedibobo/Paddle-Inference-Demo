# K100检测模型Demo

## 平台信息：

CPU：FT2000

系统：麒麟V10

加速卡：kunlun K100

方法：Paddle Inference不调用Lite Engine下用原生库推理，单独编译lite的方式测试后发现目前推理有些问题。

  至于PaddleLite和Inference在K100上的关系，引用下Paddle同学的话。

> 1、算子的计算加速由k100实现，有些xpu没实现的算子，或者是只是初始化使用（比如：gaussian_random）不频繁使用的算子会在cpu上执行。2、有些模型确实lite没法推理，但是inference可以推理的。不过之后我们在做inferrt，会统一一个推理接口。这里inferen集成了lite能力，其实就是前端写inference的实现，去运行的时候还是会调lite的能力，但是就不用咱们写lite接口那样了。针对lite优化过的效果会更好，没优化过的效果就可能一般
> 

## 准备：

### 1.Paddle环境安装：

建议使用官方的whl安装包，而不是编译安装。我尝试过，会遇到版本和网络不通等问题。

下载地址：

[https://paddle-inference-lib.bj.bcebos.com/2.2.0/python/Linux/XPU/arm64_gcc7.3_py36_openblas/paddlepaddle-2.2.0-cp37-cp37m-linux_aarch64.whl](https://paddle-inference-lib.bj.bcebos.com/2.2.0/python/Linux/XPU/arm64_gcc7.3_py36_openblas/paddlepaddle-2.2.0-cp37-cp37m-linux_aarch64.whl)

中间版本信息应该是可以更改的，但是否有没有试过。

### 2.准备检测模型：

思路是从**PaddleDetection**中导出deploy的模型，注意PaddlePaddle在升级动态图后推理模型格式是:

```bash
model.pdmodel
model.pdiparams
```

导出的方法参考[https://github.com/PaddlePaddle/PaddleDetection/blob/6e65f11d8ec0169655b5ec7614a6360b25090ae3/deploy/EXPORT_MODEL.md](https://github.com/PaddlePaddle/PaddleDetection/blob/6e65f11d8ec0169655b5ec7614a6360b25090ae3/deploy/EXPORT_MODEL.md) 中的例子，具体命令如下：

```bash
# 导出YOLOv3模型，输入是3x640x640
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --output_dir=./inference_model \
 -o weights=https://paddledet.bj.bcebos.com/models/pretrained/DarkNet53_pretrained.pdparams TestReader.inputs_def.image_shape=[3,640,640]
```

上面的脚本需要下载模型，Yolov3-Darknet53也是官方文档中支持的successful-pipeline之一（飞腾平台）。但我测试了下面说的v3_r50的模型，也是成功的。是否支持应该看算子而不是算法。

![Untitled](K100%E6%A3%80%E6%B5%8B%E6%A8%A1%E5%9E%8BDe%205dd15/Untitled.png)

或者下载这个模型，yolov3_r50，官方提供的：

```bash
wget https://paddle-inference-dist.bj.bcebos.com/Paddle-Inference-Demo/yolov3_r50vd_dcn_270e_coco.tgz
tar xzf yolov3_r50vd_dcn_270e_coco.tgz
```

3.测试脚本编写：

这里主要参考了Paddle-Inference-Demo和文档中昆仑推理的部分，需要注意Paddle推理检测模型是三个输入的，具体可以用Netron查看，我放一个Yolov3-darnet53的输入输出：

![Untitled](K100%E6%A3%80%E6%B5%8B%E6%A8%A1%E5%9E%8BDe%205dd15/Untitled%201.png)

脚本见infer-yolov3.py和utils.py

### 4.测试：

需要下载测试图片[kite.jpg]([https://paddle-inference-dist.bj.bcebos.com/inference_demo/images/kite.jpg](https://paddle-inference-dist.bj.bcebos.com/inference_demo/images/kite.jpg))

运行测试脚本后，结果如下：

```bash
(paddle-envs) [lyb@localhost Paddle-test]$ python infer-yolov3.py --rep 100
XPURT /home/lyb/paddle-envs/lib64/python3.7/site-packages/paddle/fluid/../libs/libxpurt.so loaded
--- Running analysis [ir_graph_build_pass]
--- Running analysis [ir_graph_clean_pass]
--- Running analysis [ir_analysis_pass]
I0314 14:07:09.768661 53683 ir_analysis_pass.cc:46] argument has no fuse statis
--- Running analysis [ir_params_sync_among_devices_pass]
--- Running analysis [adjust_cudnn_workspace_size_pass]
--- Running analysis [inference_op_replace_pass]
--- Running analysis [ir_graph_to_program_pass]
I0314 14:07:10.122634 53683 analysis_predictor.cc:717] ======= optimize end =======
I0314 14:07:10.146076 53683 naive_executor.cc:98] ---  skip [feed], feed -> scale_factor
I0314 14:07:10.146131 53683 naive_executor.cc:98] ---  skip [feed], feed -> image
I0314 14:07:10.146144 53683 naive_executor.cc:98] ---  skip [feed], feed -> im_shape
I0314 14:07:10.159504 53683 naive_executor.cc:98] ---  skip [save_infer_model/scale_0.tmp_1], fetch -> fetch
I0314 14:07:10.159548 53683 naive_executor.cc:98] ---  skip [save_infer_model/scale_1.tmp_1], fetch -> fetch
W0314 14:07:10.277076 53683 device_context.cc:221] Please NOTE: xpu device: 0
100
category id is 3.0, bbox is [669.78406  83.1182  672.10815 192.70438]
category id is 3.0, bbox is [544.1343  133.00688 548.03    150.09895]
category id is 3.0, bbox is [1031.9872   641.34045 1062.9467   771.38965]
category id is 3.0, bbox is [940.6428  644.91907 966.17664 757.56934]
category id is 4.0, bbox is [   0.          0.       1351.        102.429794]
category id is 4.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 4.0, bbox is [   0.         0.      1351.       275.98572]
category id is 5.0, bbox is [   0.          0.       1351.        102.429794]
category id is 6.0, bbox is [1024.8052   684.57294 1028.4695   826.9996 ]
category id is 6.0, bbox is [669.78406  83.1182  672.10815 192.70438]
category id is 6.0, bbox is [1097.2235   438.96082 1109.1053   899.     ]
category id is 6.0, bbox is [634.9766 140.161  635.8062 142.7986]
category id is 6.0, bbox is [919.662   661.20746 961.5481  846.74347]
category id is 6.0, bbox is [169.28925 810.51324 199.40015 885.73505]
category id is 8.0, bbox is [669.78406  83.1182  672.10815 192.70438]
category id is 8.0, bbox is [676.07324  84.43204 695.0852   96.96089]
category id is 8.0, bbox is [1024.8052   684.57294 1028.4695   826.9996 ]
category id is 8.0, bbox is [188.14424 674.247   221.4842  841.00507]
category id is 8.0, bbox is [571.4475  183.81213 622.0226  194.67926]
category id is 8.0, bbox is [544.1343  133.00688 548.03    150.09895]
category id is 8.0, bbox is [609.5861   86.63381 641.2109   92.52029]
category id is 8.0, bbox is [919.662   661.20746 961.5481  846.74347]
category id is 8.0, bbox is [1093.8387     28.691805 1351.         36.0655  ]
category id is 8.0, bbox is [634.9766 140.161  635.8062 142.7986]
category id is 9.0, bbox is [1031.9872   641.34045 1062.9467   771.38965]
category id is 9.0, bbox is [1024.8052   684.57294 1028.4695   826.9996 ]
category id is 9.0, bbox is [940.6428  644.91907 966.17664 757.56934]
category id is 9.0, bbox is [919.662   661.20746 961.5481  846.74347]
category id is 13.0, bbox is [   0.          0.       1351.        102.429794]
category id is 14.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 15.0, bbox is [   0.          0.       1351.        102.429794]
category id is 16.0, bbox is [   0.          0.       1351.        102.429794]
category id is 16.0, bbox is [   0.         0.      1351.       275.98572]
category id is 18.0, bbox is [   0.          0.       1351.        102.429794]
category id is 18.0, bbox is [   0.         0.      1351.       275.98572]
category id is 22.0, bbox is [   0.          0.       1351.        102.429794]
category id is 22.0, bbox is [   0.         0.      1351.       275.98572]
category id is 22.0, bbox is [669.78406  83.1182  672.10815 192.70438]
category id is 23.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 24.0, bbox is [   0.          0.       1351.        102.429794]
category id is 27.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 31.0, bbox is [1024.8052   684.57294 1028.4695   826.9996 ]
category id is 31.0, bbox is [1097.2235   438.96082 1109.1053   899.     ]
category id is 31.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 36.0, bbox is [669.78406  83.1182  672.10815 192.70438]
category id is 36.0, bbox is [634.9766 140.161  635.8062 142.7986]
category id is 36.0, bbox is [571.4475  183.81213 622.0226  194.67926]
category id is 36.0, bbox is [676.07324  84.43204 695.0852   96.96089]
category id is 37.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 38.0, bbox is [669.78406  83.1182  672.10815 192.70438]
category id is 38.0, bbox is [609.5861   86.63381 641.2109   92.52029]
category id is 38.0, bbox is [676.07324  84.43204 695.0852   96.96089]
category id is 42.0, bbox is [1031.9872   641.34045 1062.9467   771.38965]
category id is 42.0, bbox is [669.78406  83.1182  672.10815 192.70438]
category id is 42.0, bbox is [544.1343  133.00688 548.03    150.09895]
category id is 42.0, bbox is [500.3708  220.62488 737.12616 249.51828]
category id is 42.0, bbox is [1024.8052   684.57294 1028.4695   826.9996 ]
category id is 42.0, bbox is [571.4475  183.81213 622.0226  194.67926]
category id is 42.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 42.0, bbox is [940.6428  644.91907 966.17664 757.56934]
category id is 42.0, bbox is [676.07324  84.43204 695.0852   96.96089]
category id is 42.0, bbox is [634.9766 140.161  635.8062 142.7986]
category id is 42.0, bbox is [919.662   661.20746 961.5481  846.74347]
category id is 42.0, bbox is [525.38635  114.105896 553.2593   264.62106 ]
category id is 42.0, bbox is [199.18362 802.12085 217.09753 804.7954 ]
category id is 42.0, bbox is [205.58536 293.3661  262.3911  348.87555]
category id is 43.0, bbox is [   0.          0.       1351.        102.429794]
category id is 43.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 44.0, bbox is [   0.          0.       1351.        102.429794]
category id is 44.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 44.0, bbox is [   0.         0.      1351.       275.98572]
category id is 47.0, bbox is [   0.          0.       1351.        102.429794]
category id is 49.0, bbox is [   0.          0.       1351.        102.429794]
category id is 50.0, bbox is [   0.          0.       1351.        102.429794]
category id is 51.0, bbox is [1024.8052   684.57294 1028.4695   826.9996 ]
category id is 51.0, bbox is [1097.2235   438.96082 1109.1053   899.     ]
category id is 51.0, bbox is [634.9766 140.161  635.8062 142.7986]
category id is 51.0, bbox is [669.78406  83.1182  672.10815 192.70438]
category id is 51.0, bbox is [1031.9872   641.34045 1062.9467   771.38965]
category id is 51.0, bbox is [571.4475  183.81213 622.0226  194.67926]
category id is 54.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 55.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 56.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 57.0, bbox is [   0.          0.       1351.        102.429794]
category id is 57.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 58.0, bbox is [   0.          0.       1351.        102.429794]
category id is 58.0, bbox is [1024.8052   684.57294 1028.4695   826.9996 ]
category id is 59.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 60.0, bbox is [1031.9872   641.34045 1062.9467   771.38965]
category id is 62.0, bbox is [   0.          0.       1351.        102.429794]
category id is 63.0, bbox is [   0.          0.       1351.        102.429794]
category id is 63.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 63.0, bbox is [   0.         0.      1351.       275.98572]
category id is 64.0, bbox is [1093.8387     28.691805 1351.         36.0655  ]
category id is 68.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 69.0, bbox is [   0.          0.       1351.        102.429794]
category id is 69.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 72.0, bbox is [1093.8387     28.691805 1351.         36.0655  ]
category id is 73.0, bbox is [1176.3433    0.     1351.      899.    ]
category id is 76.0, bbox is [1176.3433    0.     1351.      899.    ]
time is: 852.9250192642212 ms
```

## 5.测试结果

阈值0.7

![res.jpg](K100%E6%A3%80%E6%B5%8B%E6%A8%A1%E5%9E%8BDe%205dd15/res.jpg)

可以通过xpu_msi命令查看XPU的使用情况，确认模型推理使用到了加速卡。

## Reference

[https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/09_hardware_support/xpu_docs/paddle_2.0_xpu_cn.html](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/09_hardware_support/xpu_docs/paddle_2.0_xpu_cn.html)

[https://github.com/PaddlePaddle/PaddleDetection/](https://github.com/PaddlePaddle/PaddleDetection/tree/6e65f11d8ec0169655b5ec7614a6360b25090ae3)

[https://github.com/PaddlePaddle/Paddle/issues/40403](https://github.com/PaddlePaddle/Paddle/issues/40403)

感谢Paddle同学的支持。
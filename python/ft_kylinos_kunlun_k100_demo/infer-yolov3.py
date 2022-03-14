import argparse
from socketserver import DatagramRequestHandler
import time
import numpy as np
from paddle.inference import Config, PrecisionType
from paddle.inference import create_predictor

import cv2
from PIL import Image
from utils import preprocess, draw_bbox



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,help="model dir")
    parser.add_argument("--model_file", type=str, default = "yolov3_darknet53/model.pdmodel",help="model filename")
    parser.add_argument("--params_file", type=str, default = "yolov3_darknet53/model.pdiparams", help="parameter filename")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--warmup", type=int, default=0, help="warmup")
    parser.add_argument("--repeats", type=int, default=1, help="repeats")
    parser.add_argument("--math_thread_num", type=int,
                        default=1, help="math_thread_num")

    return parser.parse_args()


def run(predictor, img):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_tensor = predictor.get_input_handle(name)
        input_tensor.reshape(img[i].shape)
        #print(input_tensor.shape())
        input_tensor.copy_from_cpu(img[i].copy())

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)
    return results


def set_config(args):
    config = Config(args.model_file, args.params_file)
    #config.enable_lite_engine(PrecisionType.Float32, True)
    # use lite xpu subgraph
    config.enable_xpu(20 * 1024 * 1024)
    # use lite cuda subgraph
    # config.enable_use_gpu(100, 0)
    config.set_cpu_math_library_num_threads(args.math_thread_num)
    return config

def main():
    args = parse_args()
    config = set_config(args)
    predictor = create_predictor(config)

    # load image
    img_name = "kite.jpg"
    save_img_name = "res.jpg"
    im_size = 608

    img = cv2.imread(img_name)
    img_data = preprocess(img, im_size)
    print(len(img_data))
    scale_factor = np.array([im_size * 1. / img.shape[0], im_size *
                            1. / img.shape[1]]).reshape((1, 2)).astype(np.float32)
    img_shape = np.array([im_size, im_size]).reshape((1, 2)).astype(np.float32)

    for i in range(args.warmup):
        results = run(predictor, [img_shape, img_data, scale_factor])

    start_time = time.time()
    for i in range(args.repeats):
        results = run(predictor, [img_shape, img_data, scale_factor])
    print(len(results[0]))
    img = Image.open(img_name).convert('RGB')
    draw_bbox(img, results[0], 0.5)
    end_time = time.time()
    
    print('time is: {}'.format((end_time-start_time)/args.repeats * 1000))

if __name__ == "__main__":
    main()

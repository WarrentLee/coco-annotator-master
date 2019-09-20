from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import numpy as np
import torch, cv2
import argparse, time
from PIL import Image
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
import matplotlib.pyplot as plt


# config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

def init():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="/home/zrj/mdisk/DaihuiYang/PRCV2019/det/R_50_FPN_PRCV_v2_2GPUs.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
             "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--save-txt",
        dest="save_txt",
        help="Save detection box for ic15 submission",
        action="store_true",
    )
    parser.add_argument(
        "--save-json",
        dest="save_json",
        help="Save detection box for ic15 submission",
        action="store_true",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "--weight",
        type=str,
        default='/home/zrj/mdisk/DaihuiYang/PRCV2019/det/model_final.pth',
        help="weight file path ",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--img",
        type=str,
        help="path to the target image",
        default="test.jpg",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="path to the target image",
        default="./output",
    )
    return parser


def main():
    parser = init()
    args = parser.parse_args()
    # update the config options with the config file
    cfg.merge_from_file(args.config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    # cfg.merge_from_list(["MODEL.MASK_ON", False])
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )
    if args.weight is not None:
        checkpoint = torch.load(args.weight)
        load_state_dict(coco_demo.model, checkpoint.pop("model"))
        del checkpoint
    start_time = time.time()
    img = cv2.imread(args.img)
    composite = coco_demo.run_on_opencv_image(img)
    print("Time: {:.2f} s / img".format(time.time() - start_time))
    #     plt.figure()
    plt.imshow(composite)
    plt.savefig("result.jpg")


#     plt.show()


if __name__ == "__main__":
    main()

import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import numpy as np
import os, json, cv2, random

from detectron2.engine import DefaultTrainer

from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import GenericMask

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.data.datasets import register_coco_instances

from detectron2.structures import BoxMode

def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


cv2.namedWindow("input", cv2.WINDOW_NORMAL)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)

if __name__ == "__main__":

    ''' ==== dataset ====='''
    db_name = "train_ceiling_cam"

    register_coco_instances(db_name, {}, "/data/0.DB/FPCB/2021/fpcb_db/2021_08_05/01/" + "fpcb_ann.json",
                            "/data/0.DB/FPCB/2021/fpcb_db/2021_08_05/01/")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # classes (fpcb)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # classes (fpcb)
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.WEIGHTS = os.path.join("./output_picking_20210809_real2/", "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    #cfg.DATASETS.TEST = ("fpcb_val", )
    predictor = DefaultPredictor(cfg)



    output_save_dir = "/data/0.Project/FPCB/experiment/20200716/test/"


    categories = ['pcb', 'cable', 'metal_no_pattern', 'metal_pattern', 'fpcb_no_pattern', 'fpcb_pattern']
    # fpcb_pattern -> connector
    # cable -> yellow film

    colormap = [
        (255, 0, 0),  # pcb
        (128, 64, 0),  # cable
        (0, 255, 0),  # metal_no_pattern
        (128, 0, 0),  # matal_pattern
        (0, 128, 0),  # fpcb_no_pattern
        (128, 255, 0),  # fpcb_pattern
        (255, 255, 0)  # full region
    ]

    out_db_dir = '/data/0.DB/FPCB/experiment/20201013/test_placement/'


    ''' test image'''

    db_name = 'train_ceiling_cam'
    fpcb_metadata = MetadataCatalog.get(db_name)
    dataset_dicts = DatasetCatalog.get(db_name)

    test_db_dir = '/data/0.DB/FPCB/2021/fpcb_db/2021_08_05/test/'
    #test_db_dir = '/data/0.DB/FPCB/2021/fpcb_db/2021_08_05/01/'


    for i in range(1):
        #d = dataset_dicts[i]
        #im = cv2.imread(d["file_name"])

        im = cv2.imread(test_db_dir + '2021_08_05_09_29_32_input.png')

        outputs = predictor(im)
        v = Visualizer(im,
                       metadata=fpcb_metadata,
                       scale=1,
                       instance_mode=ColorMode.SEGMENTATION
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        ''' ==================='''

        predictions = outputs["instances"].to("cpu")

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None

        if boxes:
            boxes = boxes.tensor.numpy()
            scores = scores.numpy()
            classes = classes.numpy()
        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            #
            # (mask_len, mask_height, mask_width) = masks.shape
            #
            # masks = [GenericMask(x, mask_height, mask_width) for x in masks]
        else:
            masks = None

        out_img = out.get_image()

        cv2.imshow("out", out_img)

        cv2.waitKey(0)












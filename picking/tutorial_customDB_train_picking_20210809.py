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


# if your dataset is in COCO format, this cell can be replaced by the following three lines:
from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

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
    '''
    1. [-] Augmented Dataset 34 categories
    2. [+] Add Ceiling Cam Dataset
    
    '''


    #db_dir = "/home/shin/0.Code/fpcb_dtron2/DB/fpcb/train/"
    total_db_num = 34
    db_dir = '/db_aug/aug_db/sample_3/'

    tmp_train_list = []

    ## Augmented Dataset

    '''

    for db_num in range(1, total_db_num + 1, 1):
        sample_db_dir = db_dir + str(db_num).zfill(2) + '/'
        rgb_db_dir = sample_db_dir + 'rgb/'
        db_name = 'fpcb_train_' + str(db_num).zfill(2)

        tmp_train_list.append(db_name)

        register_coco_instances(db_name, {}, sample_db_dir + "fpcb_ann.json", rgb_db_dir)

        fpcb_metadata = MetadataCatalog.get(db_name)
        dataset_dicts = DatasetCatalog.get(db_name)


    ###
    '''

    ### Ceiling Cam
    db_name = "train_ceiling_cam_01"
    register_coco_instances(db_name, {}, "/data/0.DB/FPCB/2021/fpcb_db/2021_08_05/01/" + "fpcb_ann.json",
                            "/data/0.DB/FPCB/2021/fpcb_db/2021_08_05/01/")
    tmp_train_list.append(db_name)
    ###


    db_name = "train_ceiling_cam_02"
    register_coco_instances(db_name, {}, "/data/0.DB/FPCB/2021/fpcb_db/2021_08_05/02/" + "fpcb_ann.json",
                            "/data/0.DB/FPCB/2021/fpcb_db/2021_08_05/02/")
    tmp_train_list.append(db_name)



    db_name = "train_arm_cam_01"
    register_coco_instances(db_name, {}, "/data/0.DB/FPCB/2021/fpcb_db/2021_12_20/01/" + "fpcb_ann.json",
                            "/data/0.DB/FPCB/2021/fpcb_db/2021_12_20/01/")
    tmp_train_list.append(db_name)

    db_name = "train_ceiling_03"
    register_coco_instances(db_name, {}, "/data/0.DB/FPCB/2021/fpcb_db/2021_12_20/02/" + "fpcb_ann.json",
                            "/data/0.DB/FPCB/2021/fpcb_db/2021_12_20/02/")
    tmp_train_list.append(db_name)

    db_name = "train_all_01"
    register_coco_instances(db_name, {}, "/data/0.DB/FPCB/2021/fpcb_db/picking_objects/01/" + "fpcb_ann.json",
                            "/data/0.DB/FPCB/2021/fpcb_db/picking_objects/01/")
    tmp_train_list.append(db_name)

    db_name = "train_all_02"
    register_coco_instances(db_name, {}, "/data/0.DB/FPCB/2021/fpcb_db/picking_objects/02/" + "fpcb_ann.json",
                            "/data/0.DB/FPCB/2021/fpcb_db/picking_objects/02/")
    tmp_train_list.append(db_name)

    db_name = "train_all_03"
    register_coco_instances(db_name, {}, "/data/0.DB/FPCB/2021/fpcb_db/picking_objects/03/" + "fpcb_ann.json",
                            "/data/0.DB/FPCB/2021/fpcb_db/picking_objects/03/")
    tmp_train_list.append(db_name)


    ##
    fpcb_metadata = MetadataCatalog.get(db_name)
    dataset_dicts = DatasetCatalog.get(db_name)


    #fpcb_metadata = MetadataCatalog.get(db_name)
    #dataset_dicts = DatasetCatalog.get(db_name)


    train_list = tuple(tmp_train_list)

    tmp = 0

    ### verification dataset
    #val_db_dir = '/data/0.DB/FPCB/paper/0.val/output/val/'

    ### train val set ###
    #db_name = "eval_val_dataset"
    #rgb_db_dir = val_db_dir + 'rgb/'
    #register_coco_instances(db_name, {}, val_db_dir + "fpcb_ann.json", rgb_db_dir)

    '''
    dataset_dicts = get_balloon_dicts(db_dir + "train")

    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=fpcb_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        sample_img = out.get_image()
        cv2.imshow("input", sample_img)
        #cv2.imshow(out.get_image()[:, :, ::-1])
    '''

    ''' ========= train ========================='''

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    #cfg.DATASETS.TRAIN = ("train_ceiling_cam", )
    cfg.DATASETS.TRAIN = train_list

    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    #cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.BASE_LR = 0.00015  # pick a good LR
    cfg.SOLVER.MAX_ITER = 3000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256*2  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 * 2  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # only has one class (ballon)
    cfg.OUTPUT_DIR = './output_picking_20211228_all_v2'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    '''

    im = cv2.imread("./input.jpg")
    cv2.imshow("input", im)

    cfg = get_cfg()

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    outimg = out.get_image()

    cv2.imshow("output", outimg)
    '''

    cv2.waitKey(0)





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



cv2.namedWindow("input", cv2.WINDOW_NORMAL)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)

if __name__ == "__main__":

    ''' ==== dataset ====='''


    #db_dir = "/home/shin/0.Code/fpcb_dtron2/DB/fpcb/train/"
    #db_dir = '/data/0.DB/FPCB/dtron2_db/via_DB/20200806/placement_training_downsample/'
    #db_dir2 = '/data/0.DB/FPCB/dtron2_db/via_DB/20200804/'

    db_dir = '/data/0.DB/FPCB/dtron2_db/placement/augdb2/coco_out/'
    rgb_db_dir = '/data/0.DB/FPCB/dtron2_db/placement/augdb2/rgb/'

    annotation_file = 'fpcb_ann_no_full_region.json'

    #register_coco_instances("fpcb_train_placement", {}, db_dir + "via_project_6Aug2020_placement_coco.json", db_dir)
    #register_coco_instances("fpcb_train2", {}, db_dir2 + "via_project_fpcb3_coco.json", db_dir2)

    # [20201013]
    '''
    register_coco_instances("fpcb_train_placement", {}, db_dir + annotation_file, rgb_db_dir)
    fpcb_metadata = MetadataCatalog.get("fpcb_train_placement")
    dataset_dicts = DatasetCatalog.get("fpcb_train_placement")
    '''

    '''

    for d in random.sample(dataset_dicts, 1):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=fpcb_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        sample_img = out.get_image()
        cv2.imshow("input", sample_img)

    cv2.waitKey(100)
    '''

    # aug data
    db_dir = '/data/0.DB/FPCB/2021/fpcb_db/placement/aug_dataset/augdb/out/01/'
    rgb_db_dir = '/data/0.DB/FPCB/2021/fpcb_db/placement/aug_dataset/augdb/out/01/rgb/'
    register_coco_instances("fpcb_placement_aug_01", {}, db_dir + 'fpcb_ann.json', rgb_db_dir)
    fpcb_metadata = MetadataCatalog.get("fpcb_placement_aug_01")
    dataset_dicts = DatasetCatalog.get("fpcb_placement_aug_01")

    # real data
    db_dir = '/data/0.DB/FPCB/2021/fpcb_db/placement/sample/01/'
    rgb_db_dir = '/data/0.DB/FPCB/2021/fpcb_db/placement/sample/01/rgb/'
    register_coco_instances("fpcb_placement_01", {}, db_dir + 'fpcb_ann.json', rgb_db_dir)
    fpcb_metadata = MetadataCatalog.get("fpcb_placement_01")
    dataset_dicts = DatasetCatalog.get("fpcb_placement_01")

    db_dir = '/data/0.DB/FPCB/2021/fpcb_db/placement/sample/02/'
    rgb_db_dir = '/data/0.DB/FPCB/2021/fpcb_db/placement/sample/02/rgb/'
    register_coco_instances("fpcb_placement_02", {}, db_dir + 'fpcb_ann.json', rgb_db_dir)
    fpcb_metadata = MetadataCatalog.get("fpcb_placement_02")
    dataset_dicts = DatasetCatalog.get("fpcb_placement_02")

    db_dir = '/data/0.DB/FPCB/2021/fpcb_db/placement/sample/03/'
    rgb_db_dir = '/data/0.DB/FPCB/2021/fpcb_db/placement/sample/03/'
    register_coco_instances("fpcb_placement_03", {}, db_dir + 'fpcb_ann.json', rgb_db_dir)
    fpcb_metadata = MetadataCatalog.get("fpcb_placement_03")
    dataset_dicts = DatasetCatalog.get("fpcb_placement_03")



    ### verification dataset

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
    cfg.DATASETS.TRAIN = ("fpcb_placement_aug_01", "fpcb_placement_01", "fpcb_placement_02", "fpcb_placement_03" )
    cfg.OUTPUT_DIR = "./output_placement_1221"
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    #cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.BASE_LR = 0.00012  # pick a good LR
    cfg.SOLVER.MAX_ITER = 3000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 #128*2  # faster, and good enough for this toy dataset (default: 512)
    ####
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # only has one class (ballon)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()








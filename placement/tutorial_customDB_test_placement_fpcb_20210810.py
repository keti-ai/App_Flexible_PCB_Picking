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

    db_dir = '/data/0.DB/FPCB/2021/fpcb_db/placement/sample/01/'
    rgb_db_dir = '/data/0.DB/FPCB/2021/fpcb_db/placement/sample/01/rgb/'
    register_coco_instances("fpcb_placement_01", {}, db_dir + 'fpcb_ann.json', rgb_db_dir)
    fpcb_metadata = MetadataCatalog.get("fpcb_placement_01")
    dataset_dicts = DatasetCatalog.get("fpcb_placement_01")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # classes (fpcb)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # classes (fpcb)
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.WEIGHTS = os.path.join("./output_placement_0810/", "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.DATASETS.TEST = ("fpcb_val", )
    predictor = DefaultPredictor(cfg)

    size = len(dataset_dicts)

    output_save_dir = "/data/0.Project/FPCB/experiment/20200716/test/"

    #categories = ['pcb', 'cable', 'metal_no_pattern', 'matal_pattern', 'fpcb_no_pattern', 'fpcb_pattern', 'full_region']
    categories = ['pcb', 'cable', 'metal_no_pattern', 'metal_pattern', 'fpcb_no_pattern', 'fpcb_pattern']
    # fpcb_pattern -> connector

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

    for i in range(size):
        d = dataset_dicts[i]
        im = cv2.imread(d["file_name"])
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
            #(mask_len, mask_height, mask_width) = masks.shape
            #
            #masks = [GenericMask(x, mask_height, mask_width) for x in masks]
        else:
            masks = None

        out_img_path = out_db_dir + "img_" + str(i).zfill(4) + ".png"
        out_img_path2 = out_db_dir + "results/" + "img_result_" + str(i).zfill(4) + ".png"

        ####
        out_boxes_path = out_db_dir + "boxes_" + str(i).zfill(4) + ".npy"
        out_scores_path = out_db_dir + "scores_" + str(i).zfill(4) + ".npy"
        out_masks_path = out_db_dir + "masks_" + str(i).zfill(4) + ".npy"
        out_classes_path = out_db_dir + "classes_" + str(i).zfill(4) + ".npy"
        ###

        out_img = out.get_image()

        #cv2.imwrite(out_img_path, im)
        cv2.imwrite(out_img_path2, out_img)

        cv2.imshow("output", out_img)
        #ㄷoutput_path = output_save_dir + str(i) + ".png"

        #cv2.imwrite(output_path, out_img)
        # cv2.imshow(out.get_image()[:, :, ::-1])

        cv2.waitKey(300)

        #np.save(out_boxes_path, boxes)
        #np.save(out_scores_path, scores)
        #np.save(out_masks_path, masks)
        #np.save(out_classes_path, classes)

        ### read example
        #xx = np.load(out_masks_path)
        #(mask_len, mask_height, mask_width) = xx.shape
        #masks = [GenericMask(x, mask_height, mask_width) for x in xx]
        #tmp = 0





        '''
        # mask size
        mask_size = [len(np.where(x.mask == True)[0]) for x in masks]
        mask_size = np.asarray(mask_size)
        # sort index depending on the mask size
        sorted_idxs = np.argsort(-mask_size).tolist()

        # find target
        target_idx = 0
        target_score = 0

        index = 0
        tmp_cnt = 0
        while True:

            if index >= len(mask_size):
                break

            target_idx = sorted_idxs[index]
            cat = classes[target_idx]

            if part_categories[cat] is not 'cable':

                instance_score = scores[target_idx]

                if instance_score > 0.8:
                    break
                else:
                    tmp_cnt = tmp_cnt + 1

                    if tmp_cnt >= 10:
                        break


            index = index +1


        box_x0, box_y0, box_x1, box_y1 = boxes[target_idx]

        print(boxes[target_idx])

        final_x = (box_x0 + box_x1) / 2
        final_y = (box_y0 + box_y1) / 2



        '''

        tmp = 0


        '''
        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5
        '''

        '''==================================='''









    '''
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    '''









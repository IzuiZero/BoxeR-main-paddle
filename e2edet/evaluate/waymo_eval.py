import argparse
import os
import numpy as np
import torch
from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2

def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.
    """
    val = val - torch.floor(val / period + offset) * period
    if not ((val >= -offset * period) & (val <= offset * period)).all().item():
        val = torch.clamp(val, -offset * period, offset * period)
    return val

class WaymoEvaluator:

    WAYMO_CLASSES = ("UNKNOWN", "VEHICLE", "PEDESTRIAN", "SIGN", "CYCLIST")

    def generate_waymo_type_results(self, infos, token_list, class_names, is_gt=False):
        score, overlap_nlz, difficulty = [], [], []
        frame_id, boxes3d, obj_type = [], [], []
        for token in token_list:
            info = infos[token]
            if not is_gt:
                info = {"scores": info["pred_scores"], "labels": info["pred_labels"], "boxes3d": info["pred_boxes3d"]}
                
            info["boxes"] = info["boxes3d"][:, [0, 1, 2, 3, 4, 5, -1]].numpy()
            info["names"] = np.array(
                [self.WAYMO_CLASSES[i] for i in info["labels"].tolist()]
            )

            if is_gt:
                info["difficulty"] = info["difficulty"].numpy()
                info["num_points_in_gt"] = info["num_points_in_gt"].numpy()

                box_mask = np.array(
                    [
                        self.WAYMO_CLASSES[i] in class_names
                        for i in info["labels"].tolist()
                    ],
                    dtype=bool
                )
                if "num_points_in_gt" in info:
                    zero_difficulty_mask = info["difficulty"] == 0
                    info["difficulty"][
                        (info["num_points_in_gt"] > 5) & zero_difficulty_mask
                    ] = 1
                    info["difficulty"][
                        (info["num_points_in_gt"] <= 5) & zero_difficulty_mask
                    ] = 2
                    nonzero_mask = info["num_points_in_gt"] > 0
                    box_mask = box_mask & nonzero_mask
                else:
                    raise NotImplementedError("Please provide num_points_in_gt for evaluating on Waymo Dataset.")

                num_boxes = box_mask.sum()
                box_name = info["names"][box_mask]

                difficulty.append(info["difficulty"][box_mask])
                score.append(torch.ones(num_boxes))
                boxes3d.append(info["boxes"][box_mask])
            else:
                info["scores"] = info["scores"].numpy()

                num_boxes = info["boxes"].shape[0]
                difficulty.append(np.zeros(num_boxes))
                score.append(info["scores"])
                boxes3d.append(np.array(info["boxes"]))
                box_name = info["names"]

            obj_type += [
                self.WAYMO_CLASSES.index(name) for i, name in enumerate(box_name)
            ]

            seq_id = int(token.split("_")[1])
            f_id = int(token.split("_")[3][:-4])

            idx = seq_id * 1000 + f_id
            frame_id.append(np.array([idx] * num_boxes))
            overlap_nlz.append(np.zeros(num_boxes))

        frame_id = np.concatenate(frame_id).reshape(-1).astype(np.int64)
        boxes3d = np.concatenate(boxes3d, axis=0)
        obj_type = np.array(obj_type).reshape(-1)
        score = np.concatenate(score).reshape(-1)
        overlap_nlz = np.concatenate(overlap_nlz).reshape(-1)
        difficulty = np.concatenate(difficulty).reshape(-1).astype(np.int8)

        boxes3d[:, -1] = limit_period(torch.tensor(boxes3d[:, -1], dtype=torch.float32), offset=0.5, period=np.pi * 2).numpy()

        return frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty

    def build_config(self):
        config = metrics_pb2.Config()
        config_text = """
        breakdown_generator_ids: OBJECT_TYPE
        difficulties {
        levels:1
        levels:2
        }
        matcher_type: TYPE_HUNGARIAN
        iou_thresholds: 0.0
        iou_thresholds: 0.7
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        box_type: TYPE_3D
        """
        for x in range(0, 100):
            config.score_cutoffs.append(x * 0.01)
        config.score_cutoffs.append(1.0)

        text_format.Merge(config_text, config)
        return config

    def waymo_evaluation(self, infos, class_name, distance_thresh=100):
        print("Start the waymo evaluation...")

        token_list = tuple(infos.keys())

        pd_frameid, pd_boxes3d, pd_type, pd_score, pd_overlap_nlz, _ = self.generate_waymo_type_results(infos, token_list, class_name, is_gt=False)
        gt_frameid, gt_boxes3d, gt_type, gt_score, gt_overlap_nlz, gt_difficulty = self.generate_waymo_type_results(infos, token_list, class_name, is_gt=True)

        print("Number: (pd, %d) VS. (gt, %d)" % (len(pd_boxes3d), len(gt_boxes3d)))
        print("Level 1: %d, Level 2: %d)" % ((gt_difficulty == 1).sum(), (gt_difficulty == 2).sum()))

        return {"aps": "Results"}

def main():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--root-path", type=str, default=None, help="pickle file")
    args = parser.parse_args()

    infos = torch.load(os.path.join(args.root_path, "results.pth"), map_location="cpu")

    classes = ["VEHICLE", "PEDESTRIAN"]
    if "classes" in infos:
        classes = infos["classes"]

    print("Start to evaluate the waymo format results...")
    evaluator = WaymoEvaluator()

    waymo_AP = evaluator.waymo_evaluation(infos, classes)

    print(waymo_AP)


if __name__ == "__main__":
    main()
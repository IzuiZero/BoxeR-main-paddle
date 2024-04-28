import os
import pickle

import numpy as np
import paddle
import uuid

from paddle.io import Dataset
from e2edet.dataset import BaseDataset, register_task
from e2edet.dataset.helper import PointDetection, DataBaseSampler, collate3d

class UUIDGeneration:
    def __init__(self):
        self.mapping = {}

    def get_uuid(self, seed):
        if seed not in self.mapping:
            self.mapping[seed] = uuid.uuid4().hex
        return self.mapping[seed]


uuid_gen = UUIDGeneration()


@register_task("detection3d")
class WaymoDetection(BaseDataset):
    LABEL_TO_IDX = {
        "UNKNOWN": 0,
        "VEHICLE": 1,
        "PEDESTRIAN": 2,
        "SIGN": 3,
        "CYCLIST": 4,
    }
    IDX_TO_LABEL = ("UNKNOWN", "VEHICLE", "PEDESTRIAN", "SIGN", "CYCLIST")

    def __init__(self, config, dataset_type, imdb_file, **kwargs):
        super(WaymoDetection, self).__init__(
            config,
            kwargs.get("name", kwargs.get("dataset_name", "waymo")),
            dataset_type,
            current_device=kwargs.get("current_device"),
            global_config=kwargs.get("global_config"),
        )

        self.root_path = os.path.abspath(imdb_file["root_path"])
        self.waymo_dataset = PointDetection(
            self.root_path,
            os.path.abspath(imdb_file["info_path"]),
            self.get_num_point_features(config["nsweeps"]),
            test_mode=(dataset_type != "train"),
            nsweeps=config["nsweeps"],
            load_interval=imdb_file["load_interval"],
        )

        self.db_sampler = None
        if imdb_file.get("db_sampler", None) is not None:
            db_sampler_config = imdb_file["db_sampler"]
            db_info_path = os.path.abspath(db_sampler_config["db_info_path"])
            with open(db_info_path, "rb") as file:
                db_info = pickle.load(file)
            groups = db_sampler_config["groups"]
            min_points = db_sampler_config["min_points"]
            difficulty = db_sampler_config["difficulty"]
            rate = db_sampler_config["rate"]

            self.db_sampler = DataBaseSampler(
                db_info, groups, min_points=min_points, difficulty=difficulty, rate=rate
            )

        classes = [self.LABEL_TO_IDX[name] for name in config["classes"]]
        self.classes = config["classes"]
        assert len(classes) > 0, "No classes found!"

        self.prepare = WaymoPreparation(classes, config["min_points"])
        self.pc_range = paddle.to_tensor(config["pc_range"])

    def get_num_point_features(self, nsweeps):
        return 5 if nsweeps == 1 else 6

    def get_answer_size(self):
        return len(self.LABEL_TO_IDX)

    def __len__(self):
        return len(self.waymo_dataset)

    def _load(self, idx):
        res, points, annos = self.waymo_dataset[idx]
        annos["labels"] = paddle.to_tensor(
            [self.LABEL_TO_IDX[name] for name in annos["names"]], dtype='int64'
        ).reshape([-1])

        sample = {
            "nsweeps": self.nsweeps,
            "calib": None,
        }
        target = {
            "metadata": res["metadata"],
            "boxes": annos["boxes"],
            "raw_boxes": annos["boxes"].copy(),
            "labels": annos["labels"],
            "raw_labels": annos["labels"].copy(),
            "num_points_in_gt": annos["num_points_in_gt"],
            "difficulty": annos["difficulty"],
        }

        points, target = self.prepare(points, target)

        if self.db_sampler is not None:
            sampled_dict = self.db_sampler.sample_all(
                self.root_path,
                annos["boxes"],
                annos["names"],
                res["metadata"]["num_point_features"],
            )

            if sampled_dict is not None:
                sampled_labels = paddle.to_tensor(
                    [self.LABEL_TO_IDX[name] for name in sampled_dict["gt_names"]], dtype='int64'
                )
                sampled_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                target["boxes"] = paddle.concat([target["boxes"], sampled_boxes], axis=0)
                target["labels"] = paddle.concat([target["labels"], sampled_labels], axis=0)

                points = paddle.concat([sampled_points, points], axis=0)
        sample["points"] = points

        if self.dataset_type == "train":
            sample, target = self.train_processor(sample, target)
        else:
            sample, target = self.test_processor(sample, target)

        return sample, target

    def get_collate_fn(self):
        return collate3d

    @paddle.no_grad()
    def prepare_for_evaluation(self, predictions, result_path, tracking=False):
        # Implement the evaluation preparation using PaddlePaddle methods
        pass

    @paddle.no_grad()
    def format_for_evalai(self, output, targets, local_eval=True, threshold=None):
        # Implement the evaluation formatting using PaddlePaddle methods
        pass


class WaymoPreparation(object):
    def __init__(self, classes, min_points):
        super(WaymoPreparation, self).__init__()
        self.classes = paddle.to_tensor(classes, dtype='int64')
        self.min_points = min_points

    def __call__(self, points, target):
        # Implement the preparation logic using PaddlePaddle methods
        pass

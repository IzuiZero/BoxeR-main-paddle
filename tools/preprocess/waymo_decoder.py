import zlib
import numpy as np
import paddle
from paddle.quaternion import Quaternion
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils

paddle.disable_static()

def decode_frame(frame, frame_id):
    lidars = extract_points(frame.lasers, frame.context.laser_calibrations, frame.pose)

    frame_name = "{scene_name}_{location}_{time_of_day}_{timestamp}".format(
        scene_name=frame.context.name,
        location=frame.context.stats.location,
        time_of_day=frame.context.stats.time_of_day,
        timestamp=frame.timestamp_micros,
    )

    example_data = {
        "scene_name": frame.context.name,
        "frame_name": frame_name,
        "frame_id": frame_id,
        "lidars": lidars,
    }

    return example_data


def decode_annos(frame, frame_id):
    veh_to_global = np.array(frame.pose.transform)

    ref_pose = np.reshape(np.array(frame.pose.transform), [4, 4])
    global_from_ref_rotation = ref_pose[:3, :3]
    objects = extract_objects(frame.laser_labels, global_from_ref_rotation)

    frame_name = "{scene_name}_{location}_{time_of_day}_{timestamp}".format(
        scene_name=frame.context.name,
        location=frame.context.stats.location,
        time_of_day=frame.context.stats.time_of_day,
        timestamp=frame.timestamp_micros,
    )

    annos = {
        "scene_name": frame.context.name,
        "frame_name": frame_name,
        "frame_id": frame_id,
        "veh_to_global": veh_to_global,
        "objects": objects,
    }

    return annos


def extract_points_from_range_image(laser, calibration, frame_pose):
    if laser.name != calibration.name:
        raise ValueError("Laser and calibration do not match")
    if laser.name == dataset_pb2.LaserName.TOP:
        frame_pose = paddle.to_tensor(
            np.reshape(np.array(frame_pose.transform), [4, 4])
        )
        range_image_top_pose = dataset_pb2.MatrixFloat.FromString(
            zlib.decompress(laser.ri_return1.range_image_pose_compressed)
        )
        range_image_top_pose_tensor = paddle.reshape(
            paddle.to_tensor(range_image_top_pose.data),
            range_image_top_pose.shape.dims,
        )
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0],
            range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2],
        )
        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation,
        )
        frame_pose = paddle.unsqueeze(frame_pose, axis=0)
        pixel_pose = paddle.unsqueeze(range_image_top_pose_tensor, axis=0)
    else:
        pixel_pose = None
        frame_pose = None
    first_return = zlib.decompress(laser.ri_return1.range_image_compressed)
    second_return = zlib.decompress(laser.ri_return2.range_image_compressed)
    points_list = []
    for range_image_str in [first_return, second_return]:
        range_image = dataset_pb2.MatrixFloat.FromString(range_image_str)
        if not calibration.beam_inclinations:
            beam_inclinations = range_image_utils.compute_inclination(
                paddle.to_tensor(
                    [calibration.beam_inclination_min, calibration.beam_inclination_max]
                ),
                height=range_image.shape.dims[0],
            )
        else:
            beam_inclinations = paddle.to_tensor(calibration.beam_inclinations)
        beam_inclinations = paddle.flip(beam_inclinations, [0])
        extrinsic = np.reshape(np.array(calibration.extrinsic.transform), [4, 4])
        range_image_tensor = paddle.reshape(
            paddle.to_tensor(range_image.data), range_image.shape.dims
        )
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            paddle.unsqueeze(range_image_tensor[..., 0], axis=0),
            paddle.unsqueeze(paddle.to_tensor(extrinsic), axis=0),
            paddle.unsqueeze(paddle.to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose,
            frame_pose=frame_pose,
        )
        range_image_cartesian = paddle.squeeze(range_image_cartesian, axis=0)
        points_tensor = paddle.gather_nd(
            paddle.concat([range_image_cartesian, range_image_tensor[..., 1:4]], axis=-1),
            paddle.where(range_image_mask),
        )
        points_list.append(points_tensor.numpy())
    return points_list


def extract_points(lasers, laser_calibrations, frame_pose):
    sort_lambda = lambda x: x.name
    lasers_with_calibration = zip(
        sorted(lasers, key=sort_lambda), sorted(laser_calibrations, key=sort_lambda)
    )
    points_xyz = []
    points_feature = []
    points_nlz = []
    for laser, calibration in lasers_with_calibration:
        points_list = extract_points_from_range_image(laser, calibration, frame_pose)
        points = np.concatenate(points_list, axis=0)
        points_xyz.extend(points[..., :3].astype(np.float32))
        points_feature.extend(points[..., 3:5].astype(np.float32))
        points_nlz.extend(points[..., 5].astype(np.float32))
    return {
        "points_xyz": np.asarray(points_xyz),
        "points_feature": np.asarray(points_feature),
    }


def global_vel_to_ref(vel, global_from_ref_rotation):
    vel = [vel[0], vel[1], 0]
    ref = np.dot(
        Quaternion(matrix=global_from_ref_rotation).inverse.rotation_matrix, vel
    )
    ref = [ref[0], ref[1], 0.0]

    return ref


def extract_objects(laser_labels, global_from_ref_rotation):
    objects = []
    for object_id, label in enumerate(laser_labels):
        category_label = label.type
        box = label.box

        speed = [label.metadata.speed_x, label.metadata.speed_y]
        accel = [label.metadata.accel_x, label.metadata.accel_y]
        num_lidar_points_in_box = label.num_lidar_points_in_box
        if num_lidar_points_in_box <= 0:
            combined_difficulty_level = 999
        if label.detection_difficulty_level == 0:
            if num_lidar_points_in_box >= 5:
                combined_difficulty_level = 1
            else:
                combined_difficulty_level = 2
        else:
            combined_difficulty_level = label.detection_difficulty_level

        ref_velocity = global_vel_to_ref(speed, global_from_ref_rotation)

        objects.append(
            {
                "id": object_id,
                "name": label.id,
                "label": category_label,
                "box": np.array(
                    [
                        box.center_x,
                        box.center_y,
                        box.center_z,
                        box.length,
                        box.width,
                        box.height,
                        ref_velocity[0],
                        ref_velocity[1],
                        box.heading,
                    ],
                    dtype=np.float32,
                ),
                "num_points": num_lidar_points_in_box,
                "detection_difficulty_level": label.detection_difficulty_level,
                "combined_difficulty_level": combined_difficulty_level,
                "tracking_difficulty_level": label.tracking_difficulty_level,
                "global_speed": np.array(speed, dtype=np.float32),
                "global_accel": np.array(accel, dtype=np.float32),
            }
        )
    return objects

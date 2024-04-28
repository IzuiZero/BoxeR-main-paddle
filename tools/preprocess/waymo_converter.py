import os
import pickle
import sys
from multiprocessing import Pool
import glob
import tqdm
import tensorflow.compat.v2 as tf
from waymo_open_dataset import dataset_pb2
from tools.preprocess.waymo_decoder import decode_frame, decode_annos

fnames = None
LIDAR_PATH = None
ANNO_PATH = None


def convert(idx):
    global fnames
    global LIDAR_PATH
    global ANNO_PATH
    fname = fnames[idx]
    dataset = tf.data.TFRecordDataset(fname, compression_type="")
    for frame_id, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        decoded_frame = decode_frame(frame, frame_id)
        decoded_annos = decode_annos(frame, frame_id)

        with open(
            os.path.join(LIDAR_PATH, "seq_{}_frame_{}.pkl".format(idx, frame_id)), "wb"
        ) as f:
            pickle.dump(decoded_frame, f)

        with open(
            os.path.join(ANNO_PATH, "seq_{}_frame_{}.pkl".format(idx, frame_id)), "wb"
        ) as f:
            pickle.dump(decoded_annos, f)


def main(args):
    global fnames
    global LIDAR_PATH
    global ANNO_PATH
    fnames = list(glob.glob(args.record_path))

    print("Number of fields {}".format(len(fnames)))

    with Pool(32) as p:
        r = list(tqdm.tqdm(p.imap(convert, range(len(fnames))), total=len(fnames)))


if __name__ == "__main__":
    args = {
        "root_path": "./",
        "record_path": "path/to/records/*.tfrecord",
    }

    if not os.path.isdir(args["root_path"]):
        os.mkdir(args["root_path"])

    LIDAR_PATH = os.path.join(args["root_path"], "lidar")
    ANNO_PATH = os.path.join(args["root_path"], "annos")

    if not os.path.isdir(LIDAR_PATH):
        os.mkdir(LIDAR_PATH)

    if not os.path.isdir(ANNO_PATH):
        os.mkdir(ANNO_PATH)

    main(args)

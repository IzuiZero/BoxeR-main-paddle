import random
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import paddle

import e2edet
from e2edet.trainer import build_trainer
from e2edet.utils.configuration import Configuration
from e2edet.utils.distributed import distributed_init, infer_init_method
from e2edet.utils.env import set_seed, get_parser


def main(configuration, init_distributed=False):
    config = configuration.get_config()

    if paddle.device.is_compiled_with_cuda():
        device_id = config.device_id
        paddle.device.set_device("gpu:{}".format(device_id))

    if init_distributed:
        distributed_init(config)

    config.training.seed = set_seed(config.training.seed)
    print("Using seed {}".format(config.training.seed))

    trainer = build_trainer(configuration)
    trainer.load()
    trainer.train()


def distributed_main(device_id, configuration):
    config = configuration.get_config()
    config.device_id = device_id

    if config.distributed.rank is None:
        config.distributed.rank = config.start_rank + device_id

    main(configuration, init_distributed=True)


def run():
    parser = get_parser()
    args = parser.parse_args()

    configuration = Configuration(args)
    configuration.args = args
    config = configuration.get_config()
    config.start_rank = 0
    if config.distributed.init_method is None:
        print("Inferring distributed setting...")
        infer_init_method(config)

    if config.distributed.init_method is not None:
        if not config.distributed.no_spawn:
            config.start_rank = config.distributed.rank
            config.distributed.rank = None
            paddle.multiprocessing.spawn(
                fn=distributed_main,
                args=(configuration,),
                nprocs=paddle.device.get_device_count(),
            )
        else:
            main(configuration, init_distributed=True)
    elif config.distributed.world_size > 1:
        assert config.distributed.world_size <= paddle.device.get_device_count()
        port = random.randint(10000, 20000)
        config.distributed.init_method = "tcp://localhost:{port}".format(port=port)
        config.distributed.rank = None
        paddle.multiprocessing.spawn(
            fn=distributed_main,
            args=(configuration,),
            nprocs=config.distributed.world_size,
        )
    else:
        config.device_id = 0
        main(configuration)


if __name__ == "__main__":
    run()

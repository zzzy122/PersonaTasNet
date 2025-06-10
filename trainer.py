import os
import time
import logging
import argparse

from utils.train import train
from utils.hparams import HParam
from utils.writer import MyWriter
from datasets.dataloader import create_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Root directory of run.")

    # parser.add_argument('-c', '--config', type=str, required=True,
    # parser.add_argument('-c', '--config', type=str ,default=os.getcwd() + '/config/default.yaml',
    parser.add_argument('-c', '--config', type=str ,default=os.getcwd() + '/config/default_train.yaml',
                help="yaml file for configuration")

    # parser.add_argument('-e', '--embedder_path', type=str, required=True,
    # parser.add_argument('-e', '--embedder_path', type=str, default=os.getcwd() + '/data_ziyun/train-clean-100-1percent/embedder_path',
    parser.add_argument('-e', '--embedder_path', type=str, default=os.getcwd() + '/data_ziyun/train-clean-100-1percent/embedder_path/embedder.pt',
                        help="path of embedder model pt file")

    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file")

    # parser.add_argument('-m', '--model', type=str, required=True,
    parser.add_argument('-m', '--model', type=str, default=os.getcwd() + '/data_ziyun/train-clean-100-1percent/model_path',
                        help="Name of the model. Used for both logging and saving checkpoints.")


    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, 'r', encoding='UTF-8') as f:
        # store hparams as string
        hp_str = ''.join(f.readlines())

    pt_dir = os.path.join(args.base_dir, hp.log.chkpt_dir, args.model)
    os.makedirs(pt_dir, exist_ok=True)
    print(pt_dir)
    
    log_dir = os.path.join(args.base_dir, hp.log.log_dir, args.model)
    os.makedirs(log_dir, exist_ok=True)

    chkpt_path = args.checkpoint_path if args.checkpoint_path is not None else None

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir,
                '%s-%d.log' % (args.model, time.time()))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    if hp.data.train_dir == '' or hp.data.test_dir == '':
        logger.error("train_dir, test_dir cannot be empty.")
        raise Exception("Please specify directories of data in %s" % args.config)

    writer = MyWriter(hp, log_dir)

    trainloader = create_dataloader(hp, args, train=True)
    # trainloader = create_dataloader(hp, args, train=False)           # 需要音频设为 False

    testloader = create_dataloader(hp, args, train=False)

    # train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str)
    train(args, pt_dir, chkpt_path, trainloader, testloader, writer, logger, hp, hp_str)


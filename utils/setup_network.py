from models.Resnet import Resnet
from utils.checkpoint import restore
from utils.logger import Logger

nets = {
    'Resnet':  Resnet.ResNet
}


def setup_network(hps):
    net = nets[hps['network']]()

    # Prepare logger
    logger = Logger()
    if hps['restore_epoch']:
        restore(net, logger, hps)

    return logger, net

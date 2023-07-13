import sys
import warnings

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

from utils.write import write_to_excel
from data.datast import get_dataloaders
from utils.checkpoint import save
from utils.hparams import setup_hparams
from utils.loops import train, evaluate
from utils.setup_network import setup_network

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(net, logger, hps):
    # Create dataloaders
    trainloader, valloader, testloader = get_dataloaders(bs=hps['bs'])

    net = net.to(device)

    learning_rate = float(hps['lr'])
    scaler = GradScaler()

    #优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.0001)
    #学习率调整策略
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=hps['n_epochs'])  # * iters

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    print("Training", hps['name'], "on", device)
    training_losses=[]
    testing_losses = []
    training_accuracies = []
    testing_accuracies = []
    for epoch in range(hps['start_epoch'], hps['n_epochs']):

        acc_tr, loss_tr = train(net, trainloader, criterion, optimizer, scaler)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)
        training_accuracies.append(acc_tr)
        training_losses.append(loss_tr)


        acc_v, loss_v = evaluate(net, valloader, criterion)
        logger.loss_val.append(loss_v)
        logger.acc_val.append(acc_v)
        testing_losses.append(loss_v)
        testing_accuracies.append(acc_v)


        # Update learning rate 使用余弦优化学习率
        scheduler.step()


        if acc_v > best_acc :
            best_acc = acc_v
            # 下面的代码是保存训练时候的模型的代码 现在先不保存
            save(net, logger, hps, epoch + 1)
            logger.save_plt(hps)

        if (epoch + 1) % hps['save_freq'] == 0:
            # 下面的代码是保存训练时候的模型的代码 现在先不保存
            # save(net, logger, hps, epoch + 1)
            logger.save_plt(hps)
        #只输出精度不输出损失
        print('Epoch %2d' % (epoch + 1),
              'Train Accuracy: %2.4f %%' % acc_tr,
              'Val Accuracy: %2.4f %%' % acc_v,
              sep='\t\t')


    #最后将训练时候的数据输入到excel
    write_to_excel(training_losses, testing_losses, training_accuracies, testing_accuracies,hps['model_save_dir'])
    # Calculate performance on test set 现在不在test集合上测试了
    # acc_test, loss_test = evaluate(net, testloader, criterion)
    # print('Test Accuracy: %2.4f %%' % acc_test,
    #       'Test Loss: %2.6f' % loss_test,
    #       sep='\t\t')


if __name__ == "__main__":
    # Important parameters
    hps = setup_hparams(sys.argv[1:])
    logger, net = setup_network(hps)

    run(net, logger, hps)

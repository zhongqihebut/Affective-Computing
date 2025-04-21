import pdb
import numpy as np
import jittor as jt

# import torch.utils.data as data
from jittor.dataset import DataLoader
# jt.flags.use_cuda = jt.has_cuda

import utils
from options import *
from config import *
from train import train_all, Total_loss
from test_all import test
from model_tsl import *
import tensorboardX
from tensorboard_logger import Logger
from senti_features import *
from tqdm import tqdm
# jt.flags.use_statistic = True  # 启用静态图模式
import os

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()
    # set hyperparameter
    config = Config(args)
    worker_init_fn = None
   
    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)
    # save config
    utils.save_config(config, os.path.join(config.output_path, "config.txt"))
    # initial network
    net = Model(config.len_feature, config.num_classes, config.r_act)
    # net = net.cuda()
    # create dataloader
    train_loader = DataLoader(
        SentiFeature(data_path=config.data_path, mode='train',
                        modal=config.modal, feature_fps=config.feature_fps,
                        num_segments=-1, sampling='random',
                        supervision='point', seed=config.seed),
            batch_size=1,
            shuffle=True, num_workers=config.num_workers)
    # for i in tqdm(train_loader):
    #     print(i.shape)

    test_loader = DataLoader(
        SentiFeature(data_path=config.data_path, mode='test',
                        modal=config.modal, feature_fps=config.feature_fps,
                        num_segments=-1, sampling='random',
                        supervision='point', seed=config.seed),
            batch_size=1,
            shuffle=False, num_workers=config.num_workers)

    # log test results
    test_info = {"step": [],
                "average_mAP[0.1:0.3]": [], "average_nAP[0.1:0.3]": [],"average_pAP[0.1:0.3]": [],
                "mAP@0.10": [], "mAP@0.15": [], "mAP@0.20": [], "mAP@0.25": [], "mAP@0.30": [],
                "Rc@0.10": [], "Rc@0.20": [], "Rc@0.30": [], "Rc@0.15": [], "Rc@0.25": [],
                "F2@0.10": [], "F2@0.20": [], "F2@0.30": [], "F2@0.15": [], "F2@0.25": []}

    best_mAP = -1
    # create loss
    criterion = Total_loss(config.lambdas)

    # build optimizer
    a_params = list(map(id, net.cls_module.a_extractor.parameters()))
    base_params = filter(lambda p: id(p) not in a_params, net.parameters())
    optimizer = jt.optim.Adam([{'params': base_params},
                                {'params': net.cls_module.a_extractor.parameters(), 'lr':10*config.lr[0]}],
                                lr=config.lr[0], betas=(0.9, 0.999), weight_decay=0.0005)
    # intial logger
    logger = Logger(config.log_path)
    memory_usages = []  # 用于存储每次训练时的显存占用
    forward_time = []  #
    time_all = 0
    count_iters=0

    for step in tqdm(
            range(1, config.num_iters + 1),
            total = config.num_iters,
            dynamic_ncols = True
        ):
        # lr update
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]
        
        if (step - 1) % (len(train_loader) // config.batch_size) == 0:
            loader_iter = iter(train_loader)

        # train a iteration
        begin_time=time.time()
        train_all(net, config, loader_iter, optimizer, criterion, logger, step,memory_usages,forward_time)
        end_time=time.time()
        if end_time-begin_time<2:
            time_all += end_time-begin_time
            count_iters += 1


        # test
        if step % 100 == 0:      
            test(net, config, logger, test_loader, test_info, step)
            jt.gc()
            for name, layer in net.named_parameters():
                if layer.requires_grad == True and layer.grad is not None:
                    logger.log_histogram(name + '_grad', layer.grad.numpy(), step)
                    logger.log_histogram(name + '_data', layer.numpy(), step)

            print(test_info["average_mAP[0.1:0.3]"][-1])
            if test_info["average_mAP[0.1:0.3]"][-1] > best_mAP:
                best_mAP = test_info["average_mAP[0.1:0.3]"][-1]
                # save test results
                utils.save_best_record(test_info, 
                    os.path.join(config.output_path, "best_record_seed_{}.txt".format(config.seed)))

                net.save(os.path.join(args.model_path,"model_seed_{}.pth".format(config.seed)))

    with open('./logs/train/memory_usages.txt', 'a') as file:
        for item in memory_usages:
            file.write(f"{item}\n")

    with open('./logs/train/forward_time_need.txt', 'a') as file:
        for item in forward_time:
            file.write(f"{item}\n")

    tmp_time="训练一个iters所用时间:" + str(time_all/count_iters)+'\n'

    # 打开文件并以追加写入的方式打开
    with open('./logs/train/time_need.txt', 'a') as f:
        # 你想写入的字符串
        f.write(tmp_time)
    # 打开文件以写入模式 ('w' 表示写入，'a' 表示追加)
    print(f"平均显存占用：{sum(memory_usages)/len(memory_usages)}\n")
    print(f"平均forward时间：{sum(forward_time)/len(forward_time)}\n")

    with open('./logs/train/time_need.txt', 'a') as f:
        # 你想写入的字符串
        f.write(f"平均显存占用：{sum(memory_usages)/len(memory_usages)}\n")
        f.write(f"平均forward时间：{sum(forward_time)/len(forward_time)}\n")

 
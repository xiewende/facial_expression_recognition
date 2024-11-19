import os
import sys

from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from model.DDAM import DDAMNet
import torch.nn.functional as F

from utils.utils import ImbalancedDatasetSampler, AttentionLoss
from utils.utils import get_logger, getModelSize

eps = sys.float_info.epsilon

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, default='/data/affectnet/', help='AffectNet dataset path.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=7, help='Number of class.')
    parser.add_argument('--pretrained_path', type=str, default='pretrained/affecnet7_epoch19_acc0.671.pth',
                        help='directory to pretrain model')
    parser.add_argument('--experiment_dir_name', type=str, default='./experiments',help='directory to project')
    parser.add_argument('--ckpt_path', type=str, default='./ckpts')
    return parser.parse_args() 

      
def run_training():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    
    # 获取当前时间
    now = datetime.now()
    # 将当前时间转换为字符串
    time_string = now.strftime('%Y%m%d%H%M%S')
    # log
    writer = SummaryWriter(log_dir=os.path.join(args.experiment_dir_name, time_string))
    if not os.path.exists(os.path.join(args.experiment_dir_name, time_string)):
        os.makedirs(os.path.join(args.experiment_dir_name, time_string))
    logger = get_logger(os.path.join(args.experiment_dir_name, os.path.join(time_string, 'exps.log'))) 
    logger.info('start training!')
    logger.info(args)


    model = DDAMNet(num_class=args.num_class, num_head=args.num_head)
    model.to(device)

    params_size = getModelSize(model)
    logger.info('模型总大小为：{:.3f}MB'.format(params_size))
        
    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(),
        ])
    
    train_dataset = datasets.ImageFolder(f'{args.aff_path}/train', transform = data_transforms)   
    if args.num_class == 7:   # ignore the 8-th class
        idx = [i for i in range(len(train_dataset)) if train_dataset.imgs[i][1] != 7]
        train_dataset = data.Subset(train_dataset, idx)

    print('Whole train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               sampler=ImbalancedDatasetSampler(train_dataset),
                                               shuffle = False, 
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])      

    val_dataset = datasets.ImageFolder(f'{args.aff_path}/val', transform = data_transforms_val)  
    if args.num_class == 7:   # ignore the 8-th class 
        idx = [i for i in range(len(val_dataset)) if val_dataset.imgs[i][1] != 7]
        val_dataset = data.Subset(val_dataset, idx)

    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)


    criterion_cls = torch.nn.CrossEntropyLoss().to(device)
    criterion_at = AttentionLoss()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params,args.lr,weight_decay = 0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.6)
    
    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        running_loss_cls = 0.0
        running_loss_att = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.to(device)
            targets = targets.to(device)
                        
            out,feat,heads = model(imgs)

            # loss
            cls_loss = criterion_cls(out,targets)
            att_loss = criterion_at(heads)

            loss = cls_loss  + 0.1*att_loss

            loss.backward()
            optimizer.step()
            
            running_loss += loss
            running_loss_cls += cls_loss
            running_loss_att += att_loss

            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        running_loss_cls = running_loss_cls/iter_cnt
        running_loss_att = running_loss_att/iter_cnt

        # tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))
        logger.info('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. Running_loss_cls: %.3f. Running_loss_att: %.3f. LR %.6f' % (epoch, acc, running_loss, running_loss_cls, running_loss_att, optimizer.param_groups[0]['lr']))

        writer.add_scalar('running_loss', running_loss, epoch)
        writer.add_scalar('running_loss_cls', running_loss_cls, epoch)
        writer.add_scalar('running_loss_att', running_loss_att, epoch)
        writer.add_scalar('acc', acc, epoch)
        
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            model.eval()
            for imgs, targets in val_loader:
        
                imgs = imgs.to(device)
                targets = targets.to(device)
                out,feat,heads = model(imgs)

                loss = criterion_cls(out,targets)  + 0.1*criterion_at(heads)

                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(out, 1)
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                
            running_loss = running_loss/iter_cnt   
            scheduler.step()

            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            best_acc = max(acc,best_acc)

            writer.add_scalar('val_loss', running_loss, epoch)
            writer.add_scalar('val acc', acc, epoch)


            # tqdm.write("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (epoch, acc, running_loss))
            # tqdm.write("best_acc:" + str(best_acc))
            logger.info("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (epoch, acc, running_loss))
            logger.info("best_acc:" + str(best_acc))

            logger.info('Saving model...')
            save_path = os.path.join(args.ckpt_path, time_string)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if args.num_class == 7 and  acc > 0.5:
                # torch.save({'iter': epoch,
                #             'model_state_dict': model.state_dict(),
                #              'optimizer_state_dict': optimizer.state_dict(),},
                #             os.path.join('checkpoints', "affecnet7_epoch"+str(epoch)+"_acc"+str(acc)+".pth"))
                # tqdm.write('Model saved.')
                model_name = "affecnet7_epoch"+str(epoch)+"_acc"+str(acc)+".pth"
                torch.save(model.state_dict(), os.path.join(save_path, model_name))
                logger.info('Model saved of epoch {}'.format(epoch))
                # 保存最好的
                if acc == best_acc :
                    model_name = "affecnet7_epoch"+str(epoch)+"_acc"+str(acc)+"_best.pth"
                    torch.save(model.state_dict(), os.path.join(save_path, model_name))

            elif args.num_class == 8 and  acc > 0.5:
                # torch.save({'iter': epoch,
                #             'model_state_dict': model.state_dict(),
                #              'optimizer_state_dict': optimizer.state_dict(),},
                #             os.path.join('checkpoints', "affecnet8_epoch"+str(epoch)+"_acc"+str(acc)+".pth"))
                # tqdm.write('Model saved.')
                model_name = "affecnet8_epoch"+str(epoch)+"_acc"+str(acc)+".pth"
                torch.save(model.state_dict(), os.path.join(save_path, model_name))
                logger.info('Model saved of epoch {}'.format(epoch))
        
if __name__ == "__main__":                    
    run_training()
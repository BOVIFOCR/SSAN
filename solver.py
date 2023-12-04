import torch
import torch.nn as nn
import os
from networks import get_model
from datasets import data_merge
from optimizers import get_optimizer
from torch.utils.data import Dataset, DataLoader
from transformers import *
from utils import *
from configs import parse_args
import time
import numpy as np
import random
from loss import *


torch.manual_seed(16)
np.random.seed(16)
random.seed(16)


def main(args):
    data_bank = data_merge(args.data_dir)
    # define train loader
    if args.trans in ["o"]:
        train_set = data_bank.get_datasets(train=True, protocol=args.protocol, img_size=args.img_size, map_size=args.map_size, transform=transformer_train(), debug_subset_size=args.debug_subset_size)
    elif args.trans in ["p"]:
        train_set = data_bank.get_datasets(train=True, protocol=args.protocol, img_size=args.img_size, map_size=args.map_size, transform=transformer_train_pure(), debug_subset_size=args.debug_subset_size)
    elif args.trans in ["I"]:
        train_set = data_bank.get_datasets(train=True, protocol=args.protocol, img_size=args.img_size, map_size=args.map_size, transform=transformer_train_ImageNet(), debug_subset_size=args.debug_subset_size)
    else:
        raise Exception
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    max_iter = args.num_epochs*len(train_loader)
    # define model
    model = get_model(args.model_type, max_iter).cuda()
    # def optimizer
    optimizer = get_optimizer(
        args.optimizer, model, 
        lr=args.base_lr,
        momentum=args.momentum, 
        weight_decay=args.weight_decay)
    # def scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # TODO change this and state_dict refs below to return to DataParallel
    # model = nn.DataParallel(model).cuda()

    # make dirs
    model_root_path = os.path.join(args.result_path, args.result_name, "model")
    check_folder(model_root_path)
    score_root_path = os.path.join(args.result_path, args.result_name, "score")
    check_folder(score_root_path)
    csv_root_path = os.path.join(args.result_path, args.result_name, "csv")
    check_folder(csv_root_path)


    # define loss
    binary_fuc = nn.CrossEntropyLoss()
    map_fuc = nn.MSELoss()
    contra_fun = ContrastLoss()

    # metrics
    eva = {
        "test": {
            "best_epoch": -1,
            "best_HTER": 100,
            "best_auc": -100
            },
        "dev": {
            "best_epoch": -1,
            "best_HTER": 100,
            "best_auc": -100
            }
    }

    for epoch in range(args.start_epoch, args.num_epochs):
        binary_loss_record = AvgrageMeter()
        constra_loss_record = AvgrageMeter()
        adv_loss_record = AvgrageMeter()
        loss_record = AvgrageMeter()
        # train
        model.train()
        for i, sample_batched in enumerate(train_loader):
            image_x, label, UUID = sample_batched["image_x"].cuda(), sample_batched["label"].cuda(), sample_batched["UUID"].cuda()
            if args.model_type in ["SSAN_R"]:
                rand_idx = torch.randperm(image_x.shape[0])
                cls_x1_x1, fea_x1_x1, fea_x1_x2, domain_invariant = model(image_x, image_x[rand_idx, :, :, :])
                binary_loss = binary_fuc(cls_x1_x1, label[:, 0].long())
                contrast_label = label[:, 0].long() == label[rand_idx, 0].long()
                contrast_label = torch.where(contrast_label==True, 1, -1)
                constra_loss = contra_fun(fea_x1_x1, fea_x1_x2, contrast_label)
                adv_loss = binary_fuc(domain_invariant, UUID.long())
                loss_all = binary_loss + constra_loss + adv_loss
            elif args.model_type in ["SSAN_M"]:
                map_x = sample_batched["map_x"].cuda()
                rand_idx = torch.randperm(image_x.shape[0])
                cls_x1_x1, fea_x1_x1, fea_x1_x2, domain_invariant = model(image_x, image_x[rand_idx, :, :, :])
                binary_loss = map_fuc(cls_x1_x1, map_x)
                contrast_label = label[:, 0].long() == label[rand_idx, 0].long()
                contrast_label = torch.where(contrast_label==True, 1, -1)
                constra_loss = contra_fun(fea_x1_x1, fea_x1_x2, contrast_label)
                adv_loss = binary_fuc(domain_invariant, UUID.long())
                loss_all = binary_loss + constra_loss + adv_loss
            n = image_x.shape[0]
            binary_loss_record.update(binary_loss.data, n)
            constra_loss_record.update(constra_loss.data, n)
            adv_loss_record.update(adv_loss.data, n)
            loss_record.update(loss_all.data, n)

            model.zero_grad()
            loss_all.backward()
            optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            if i % args.print_freq == args.print_freq - 1:
                print("epoch:{:d}, mini-batch:{:d}, lr={:.4f}, binary_loss={:.4f}, constra_loss={:.4f}, adv_loss={:.4f}, Loss={:.4f}".format(epoch + 1, i + 1, lr, binary_loss_record.avg, constra_loss_record.avg, adv_loss_record.avg, loss_record.avg))
        
        # whole epoch average
        #print("epoch:{:d}, Train: lr={:f}, Loss={:.4f}".format(epoch + 1, lr, loss_record.avg))
        print(f"WHOLE_EPOCH_REPORT epoch:{epoch+1}, Train: {lr=}, Loss={loss_record.avg:.4f}, binary_loss={binary_loss_record.avg:.4f}, constra_loss={constra_loss_record.avg:.4f}, adv_loss={adv_loss_record.avg:.4f}")
        scheduler.step()

        # dev + test
        USE_THR_FROM_DEV = False
        prev_val_thr = None
        for TEST_NAME in ["dev", "test"]:
            data_bank.set_test_name(TEST_NAME)
            epoch_test = 1
            if epoch % epoch_test == epoch_test-1:
                if args.trans in ["o", "p"]:
                    test_data_dic = data_bank.get_datasets(train=False, protocol=args.protocol, img_size=args.img_size, transform=transformer_test_video(), debug_subset_size=args.debug_subset_size)
                elif args.trans in ["I"]:
                    test_data_dic = data_bank.get_datasets(train=False, protocol=args.protocol, img_size=args.img_size, transform=transformer_test_video_ImageNet(), debug_subset_size=args.debug_subset_size)
                else:
                    raise Exception
                map_score_name_list = []
                score_path = os.path.join(score_root_path, "epoch_{}".format(epoch+1))
                check_folder(score_path)
                for i, test_name in enumerate(test_data_dic.keys()):
                    print("[{}/{}]Testing {} {}...".format(i+1, len(test_data_dic), TEST_NAME, test_name))
                    test_set = test_data_dic[test_name]
                    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
                    HTER, auc_test, found_thr = test_video(model, args, test_loader, score_path, epoch, name=test_name, TEST_NAME=TEST_NAME, prev_val_thr=prev_val_thr if (TEST_NAME == "test" and USE_THR_FROM_DEV) else None)
                    if TEST_NAME == "dev":
                        prev_val_thr = found_thr
                    if auc_test-HTER>=eva[TEST_NAME]["best_auc"]-eva[TEST_NAME]["best_HTER"]:
                        eva[TEST_NAME]["best_auc"] = auc_test
                        eva[TEST_NAME]["best_HTER"] = HTER
                        eva[TEST_NAME]["best_epoch"] = epoch+1
                        model_path = os.path.join(model_root_path, "{}_p{}_best_{}.pth".format(args.model_type, args.protocol, TEST_NAME))
                        torch.save({
                            'epoch': epoch+1,
                            'state_dict': model.state_dict(),
                            # 'state_dict':model.module.state_dict(),
                            'optimizer':optimizer.state_dict(),
                            'scheduler':scheduler,
                            'args':args,
                        }, model_path)
                        print("Model saved to {}".format(model_path))
                    print("({}) [Best result] epoch:{}, HTER={:.4f}, AUC={:.4f}".format(TEST_NAME, eva[TEST_NAME]["best_epoch"],  eva[TEST_NAME]["best_HTER"], eva[TEST_NAME]["best_auc"]))
                model_path = os.path.join(model_root_path, "{}_p{}_recent.pth".format(args.model_type, args.protocol))
                torch.save({
                    'epoch': epoch+1,
                    # 'state_dict':model.module.state_dict(),
                    'state_dict': model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'scheduler':scheduler,
                    'args':args,
                }, model_path)
                print("Model saved to {}".format(model_path))


def test_video(model, args, test_loader, score_root_path, epoch, name="", TEST_NAME="test", prev_val_thr=None):
    model.eval()
    is_ssan_r = args.model_type == "SSAN_R"
    with torch.no_grad():
        loss_fn = nn.CrossEntropyLoss() if is_ssan_r else nn.MSELoss()
        start_time = time.time()
        scores_list = []
        total_loss = 0
        for i, sample_batched in enumerate(test_loader):
            image_x, label, map_x = sample_batched["image_x"].cuda(), sample_batched["label"].cuda(), sample_batched["map_x"].cuda()
            map_score = 0
            loss_inc = 0
            for frame_i in range(image_x.shape[1]):
                if args.model_type in ["SSAN_R"]:
                    cls_x1_x1, fea_x1_x1, fea_x1_x2, _ = model(image_x[:,frame_i,:,:,:], image_x[:,frame_i,:,:,:])
                    loss_inc += loss_fn(cls_x1_x1, label[:, 0].long()).item()
                    score_norm = torch.softmax(cls_x1_x1, dim=1)[:, 1]
                elif args.model_type in ["SSAN_M"]:
                    pred_map, fea_x1_x1, fea_x1_x2, _ = model(image_x[:,frame_i,:,:,:], image_x[:,frame_i,:,:,:])
                    try:
                        loss_inc += loss_fn(pred_map, map_x[:, frame_i, :, :])
                    except:
                        print(pred_map.shape, map_x.shape)
                        exit(1)
                    score_norm = torch.sum(pred_map, dim=(1, 2))/(args.map_size*args.map_size)
                map_score += score_norm
            map_score = map_score/image_x.shape[1]
            total_loss += loss_inc
            for ii in range(image_x.shape[0]):
                scores_list.append("{} {}\n".format(map_score[ii], label[ii][0]))

        map_score_val_filename = os.path.join(score_root_path, "{}_score.txt".format(name))
        print("score: write test scores to {}".format(map_score_val_filename))
        with open(map_score_val_filename, 'w') as file:
            file.writelines(scores_list)

        test_ACC, fpr, FRR, HTER, auc_test, test_err, thr = performances_val(map_score_val_filename, prev_val_thr)
        print("## {} score:".format(name))
        print("epoch:{:d}, {}:  val_ACC={:.4f}, HTER={:.4f}, AUC={:.4f}, val_err={:.4f}, ACC={:.4f}, thr={}, total_loss={}".format(epoch+1, TEST_NAME, test_ACC, HTER, auc_test, test_err, test_ACC, thr, total_loss))
        print("{} phase cost {:.4f}s".format(TEST_NAME, time.time()-start_time))
    return HTER, auc_test, thr


if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args=args)

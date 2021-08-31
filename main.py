import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
from PMCNet import *
from utils import *
from modeling.deeplab import *
from dataset import *
from config import *
from copy import deepcopy

def set_seed(global_seed):
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(999)
    np.random.seed(global_seed)
    random.seed(global_seed)


DATA_DICT = {'n':'SYM_NYU', 'l': 'SYM_LDRS', 's': 'SYNTHETIC_COCO', 'd': 'SYM_SDRW'}

def train(net, args, train_loader, val_loaders, test_loaders, device, mode='synth'):
    param_groups = net.parameters() # freaeze backbone?
    max_f1 = 0.0
    optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.num_epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        net.train()
        print('%s image train' % mode)
        print(epoch)
        running_sample = 0
        running_loss = 0
        train_log_images = []
        
        for idx, data in enumerate(tqdm(train_loader)):
            img, mask, axis, axis_gs, is_syn = data
            a_lbl = None

            _mask = (mask > 0).float().to(device)
            axis_out, mask_out, total_loss, losses = net(img=img.to(device), lbl=axis_gs.float().to(device), mask=_mask, is_syn = is_syn.to(device), a_lbl=a_lbl)
            loss = total_loss.mean()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            running_sample += 1
            running_loss += loss.item()

            if (idx == len(train_loader) - 1):
                log_dict = {"Train Loss": running_loss / running_sample,}
                running_sample = 0
                running_loss = 0
        
        if epoch % 5 == 0:
            rec, prec, f1, f1_max = test(net, args, val_loaders, device, mode='val')  # subset of testset

            _max_f1 = f1_max[0]
            max_f1 = max(max_f1, _max_f1)

            if max_f1 == _max_f1:
                print('best model renewed.')
                print(max_f1)
                best_ckpt = deepcopy(net.state_dict())
                checkpoint = {
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': args,
                    # 'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'max_f1': max_f1,
                }
                if not os.path.exists('./weights'):
                    os.makedirs('./weights')
                torch.save(checkpoint, './weights/v_' + args.ver + '_best_checkpoint.pt')

    net.load_state_dict(best_ckpt)
    rec, prec, f1, f1_max  = test(net, args, test_loaders, device, mode='test')
    checkpoint['stats'] = rec, prec, f1, f1_max
    print(f1_max)
    torch.save(checkpoint, './weights/v_' + args.ver + '_best_checkpoint.pt')

def test(net, args, test_loaders, device, mode='test'):
    net.eval()
    val_sample, val_loss = 0, 0

    n_thresh = 100 if mode == 'test' else 10

    recs, precs, f1s, f1_maxs= [], [], [], []
    with torch.no_grad():
        for i, test_loader in enumerate(test_loaders):
            axis_eval = AxisEvaluation(n_thresh)
            val_log_images = []

            for idx, data in enumerate(tqdm(test_loader)):
                img, mask, axis, axis_gs, is_syn = data
                _mask = (mask > 0).float().to(device)

                axis_out, mask_out, total_loss, losses = net(img=img.to(device), lbl=axis_gs.float().to(device), mask=_mask, is_syn = is_syn)
                axis_out = F.interpolate(axis_out, size=axis.size()[2:], mode='bilinear', align_corners=True)
                img = F.interpolate(img, size=axis.size()[2:], mode='bilinear', align_corners=True)
                
                axis_eval(axis_out, axis)
                val_loss += total_loss.mean().item()
                val_sample += 1
            
            rec, prec, f1 = axis_eval.f1_score()
            recs.append(rec), precs.append(prec), f1s.append(f1), f1_maxs.append(f1.max())
            print('test loader %d, max F1 : ' % i, f1.max())

    return recs, precs, f1s, f1_maxs

if __name__ == '__main__':
    set_seed(1)
    args = get_parser()
    comment = str(args.ver)
    
    print(comment)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.device_count() > 0, 'GPUs are not specified for training or testing.'
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    args.sync_bn = True
    args = check_ablation(args)
        
    train_data = list(args.train_data)
    train_data = [DATA_DICT[data] for data in train_data]
    
    net = SymmetryDetectionNetwork(args)
    net = nn.DataParallel(net)
    net.to(device)

    testset1 = SymmetryDatasets(dataset=['SYM_SDRW'], split='test', input_size=[args.input_size, args.input_size])
    test_loader1 = data.DataLoader(testset1, batch_size=1, shuffle=False, num_workers=4)
    testset2 = SymmetryDatasets(dataset=['SYM_LDRS'], split='test', input_size=[args.input_size, args.input_size])
    test_loader2 = data.DataLoader(testset2, batch_size=1, shuffle=False, num_workers=4)

    if args.test_only:
        print('load pretrained model')
        ckpt_path = './weights/v_' + args.ver + '_best_checkpoint.pt'
        checkpoint = torch.load(ckpt_path)
        args = checkpoint['args']
        net.load_state_dict(checkpoint['state_dict'], strict=True)
        print(args)
        rec, prec, f1, f1_max = test(net, args, (test_loader1, test_loader2), device, mode='test')
        print(f1_max)
    else:
        # Create Dataset
        trainset = SymmetryDatasets(dataset=train_data, split='train', input_size=[args.input_size, args.input_size])
        valset = SymmetryDatasets(dataset=['SYM_LDRS'], split='val', input_size=[args.input_size, args.input_size])
        train_loader = data.DataLoader(trainset, batch_size=args.bs_train, shuffle=True, num_workers=4, drop_last=True)
        val_loader = data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=4)
        train(net, args, train_loader, (val_loader, ), (test_loader1, test_loader2), device)

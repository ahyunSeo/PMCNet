import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Symmetry detection with pytorch')
    parser.add_argument('--dsc_size', default=5, type=int)
    parser.add_argument('--sym_size', default=5, type=int)
    parser.add_argument('--dsc_ray_length', default=4, type=int)
    parser.add_argument('--sym_ray_length', default=8, type=int)
    parser.add_argument('--input_size', default=417, type=int)
    parser.add_argument('--angle_interval', default=45, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('-td', '--train_data', default='nlsd', type=str) # N: 'SYM_NYU', L: 'SYM_LDRS', S: 'SYNTHETIC_COCO', D: 'SYM_SDRW'
    parser.add_argument('--bs_train', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--bs_val', '-bs', default=16, type=int,
                        help='Batch size for validation')
    parser.add_argument('--ver', default='init', type=str)
    parser.add_argument('--lr',         type=float, default=0.001,      metavar='LR',
                        help="base learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0,        metavar='DECAY',
                        help="weight decay, if 0 nothing happens")
    parser.add_argument('-t', '--test_only', action='store_true')
    parser.add_argument('-wf', '--wandb_off', action='store_true', default=False)
    parser.add_argument('--sync_bn', action='store_true', default=False)

    parser.add_argument('--unfreeze', action='store_true', default=False) # unfreeze backbone
    parser.add_argument('--use_sym_region', action='store_true', default=False,
                        help='to establish symmetry region')
    parser.add_argument('--use_selfsim', action='store_true', default=False,
                    help='to use self-sim feature descriptor')
    parser.add_argument('--residual_score', '-res_score', action='store_true', default=False)
    parser.add_argument('-ab', '--ablation', default=0, type=int)
    
    args = parser.parse_args()

    return args

def check_ablation(args):

    if args.ablation == 0:
        args.unfreeze = True
        args.use_sym_region = True
        args.use_selfsim = True
        args.residual_score = True
    
    return args

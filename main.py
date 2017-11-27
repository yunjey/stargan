import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    # Data loader
    celebA_loader = None
    rafd_loader = None

    if config.dataset in ['CelebA', 'Both']:
        celebA_loader = get_loader(config.celebA_image_path, config.metadata_path, config.celebA_crop_size,
                                   config.image_size, config.batch_size, 'CelebA', config.mode)
    if config.dataset in ['RaFD', 'Both']:
        rafd_loader = get_loader(config.rafd_image_path, None, config.rafd_crop_size,
                                 config.image_size, config.batch_size, 'RaFD', config.mode)

    # Solver
    solver = Solver(celebA_loader, rafd_loader, config)

    if config.mode == 'train':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.train()
        elif config.dataset in ['Both']:
            solver.train_multi()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--c_dim', type=int, default=5)
    parser.add_argument('--c2_dim', type=int, default=8)
    parser.add_argument('--celebA_crop_size', type=int, default=178)
    parser.add_argument('--rafd_crop_size', type=int, default=256)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--lambda_cls', type=float, default=1)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--d_train_repeat', type=int, default=5)

    # Training settings
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_epochs_decay', type=int, default=10)
    parser.add_argument('--num_iters', type=int, default=200000)
    parser.add_argument('--num_iters_decay', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Test settings
    parser.add_argument('--test_model', type=str, default='20_1000')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--celebA_image_path', type=str, default='./data/CelebA_nocrop/images')
    parser.add_argument('--rafd_image_path', type=str, default='./data/RaFD/train')
    parser.add_argument('--metadata_path', type=str, default='./data/list_attr_celeba.txt')
    parser.add_argument('--log_path', type=str, default='./test/logs')
    parser.add_argument('--model_save_path', type=str, default='./test/models')
    parser.add_argument('--sample_path', type=str, default='./test/samples')
    parser.add_argument('--result_path', type=str, default='./test/results')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
import argparse

import data
import distillers
from .base_options import BaseOptions


class DistillOptions(BaseOptions):
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, isTrain=True):
        """Reset the class; indicates the class hasn't been initailized"""
        super(DistillOptions, self).__init__()
        self.isTrain = isTrain

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # log parameters
        parser.add_argument('--log_dir', type=str, default='logs/distill',
                            help='specify an experiment directory')
        parser.add_argument('--tensorboard_dir', type=str, default=None,
                            help='tensorboard is saved here')
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=20000,
                            help='frequency of evaluating and save the latest model')
        parser.add_argument('--save_epoch_freq', type=int, default=5,
                            help='frequency of saving checkpoints at the end of epoch')
        parser.add_argument('--epoch_base', type=int, default=1,
                            help='the epoch base of the training (used for resuming)')
        parser.add_argument('--iter_base', type=int, default=0,
                            help='the iteration base of the training (used for resuming)')

        # model parameters
        parser.add_argument('--distiller', type=str, default='resnet',
                            help='specify which distiller you want to use [resnet | spade]')
        parser.add_argument('--netD', type=str, default='n_layers',
                            help='specify discriminator architecture [n_layers | pixel]. '
                                 'The basic model is a 70x70 PatchGAN. '
                                 'n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--ndf', type=int, default=128, help='the base number of discriminator filters')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--gan_mode', type=str, default='hinge', choices=['lsgan', 'vanilla', 'hinge'],
                            help='the type of GAN objective. [vanilla| lsgan | hinge]. '
                                 'vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')

        # training parameters
        parser.add_argument('--nepochs', type=int, default=5,
                            help='number of epochs with the initial learning rate')
        parser.add_argument('--nepochs_decay', type=int, default=15,
                            help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--pre_pix2pix_D', type=str, default='pretrained/pix2pix/cityscapes/latest_net_D.pth', help='the path of pretrained discriminator')
        parser.add_argument('--eval_batch_size', type=int, default=1, help='the evaluation batch size')
        parser.add_argument('--real_stat_path', type=str,
                            help='the path to load the ground-truth images information to compute FID.')
        parser.add_argument('--update_freq', type=int, default=2, help='the frequency(e.g. 2 epoch) update the guide inds')
        return parser

    def gather_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()

        distiller_name = opt.distiller
        distiller_option_setter = distillers.get_option_setter(distiller_name)
        parser = distiller_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

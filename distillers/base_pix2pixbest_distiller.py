import itertools
import os

import numpy as np
import torch
from torch import nn
from torch.nn import DataParallel

from torchprofile import profile_macs
from collections import OrderedDict
import models.modules.loss
from data import create_eval_dataloader
from metric import create_metric_models
from models import networks
from models.base_model import BaseModel
from utils import util
import math

class BasePix2PixBestDistiller(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert is_train
        parser = super(BasePix2PixBestDistiller, BasePix2PixBestDistiller).modify_commandline_options(parser, is_train)
        parser.add_argument('--teacher_netG', type=str, default='unet_256',
                            help='specify teacher generator architecture',)
        parser.add_argument('--student_netG', type=str, default='unet_256',
                            help='specify student generator architecture',)

        parser.add_argument('--teacher_ngf', type=int, default=64,
                            help='the base number of filters of the teacher generator')
        parser.add_argument('--student_ngf', type=int, default=16,
                            help='the base number of filters of the student generator')

        parser.add_argument('--restore_teacher_G_path', type=str, default=None,
                            help='the path to restore the wider teacher generator')
        parser.add_argument('--restore_student_G_path', type=str, default=None,
                            help='the path to restore the student generator')
        parser.add_argument('--restore_A_path', type=str, default=None,
                            help='the path to restore the adaptors for distillation')
        parser.add_argument('--restore_D_path', type=str, default=None,
                            help='the path to restore the discriminator')
        parser.add_argument('--restore_O_path', type=str, default=None,
                            help='the path to restore the optimizer')

        parser.add_argument('--recon_loss_type', type=str, default='l1',
                            choices=['l1', 'l2', 'smooth_l1', 'vgg'],
                            help='the type of the reconstruction loss')
        parser.add_argument('--lambda_CD', type=float, default=0,
                            help='weights for the intermediate activation distillation loss')
        parser.add_argument('--lambda_recon', type=float, default=100,
                            help='weights for the reconstruction loss.')
        parser.add_argument('--lambda_gan', type=float, default=1,
                            help='weight for gan loss')


        parser.add_argument('--teacher_dropout_rate', type=float, default=0)
        parser.add_argument('--student_dropout_rate', type=float, default=0)

        parser.add_argument('--n_share', type=int, default=0, help='shared blocks in D')
        parser.add_argument('--project', type=str, default=None, help='the project name of this trail')
        parser.add_argument('--name', type=str, default=None, help='the name of this trail')
        return parser

    def __init__(self, opt):
        assert opt.isTrain
        super(BasePix2PixBestDistiller, self).__init__(opt)
        self.loss_names = ['G_gan',  'G_recon', 'D_fake', 'D_real',
                           'G_SSIM', 'G_feature', 'G_style', 'G_tv', 'G_CD',
                           'G_row', 'G_column', 'G_patch', 'D_student']
        self.optimizers = []
        self.image_paths = []
        self.visual_names = ['real_A', 'Sfake_B', 'Tfake_B', 'real_B']
        self.model_names = ['netG_student', 'netG_teacher', 'netD_teacher']
        self.netG_teacher = networks.define_G(opt.input_nc, opt.output_nc, opt.teacher_ngf,
                                              opt.teacher_netG, opt.norm, opt.teacher_dropout_rate,
                                              opt.init_type, opt.init_gain, self.gpu_ids, opt=opt)
        self.netG_student = networks.define_G(opt.input_nc, opt.output_nc, opt.student_ngf,
                                              opt.student_netG, opt.norm, opt.student_dropout_rate,
                                              opt.init_type, opt.init_gain, self.gpu_ids, opt=opt)

        if opt.dataset_mode == 'aligned':
            self.netD_teacher = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, n_share=self.opt.n_share)
        elif opt.dataset_mode == 'unaligned':
            self.netD_teacher = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, n_share=self.opt.n_share)
        else:
            raise NotImplementedError('Unknown dataset mode [%s]!!!' % opt.dataset_mode)

        self.netG_teacher.train()
        self.netG_student.train()
        self.netD_teacher.train()
        print("netD name:")
        for n, m in self.netD_teacher.named_modules():
            print(n)
        self.criterionGAN = models.modules.loss.GANLoss(opt.gan_mode).to(self.device)
        if opt.recon_loss_type == 'l1':
            self.criterionRecon = torch.nn.L1Loss()
        elif opt.recon_loss_type == 'l2':
            self.criterionRecon = torch.nn.MSELoss()
        elif opt.recon_loss_type == 'smooth_l1':
            self.criterionRecon = torch.nn.SmoothL1Loss()
        elif opt.recon_loss_type == 'vgg':
            self.criterionRecon = models.modules.loss.VGGLoss().to(self.device)
        else:
            raise NotImplementedError('Unknown reconstruction loss type [%s]!' % opt.loss_type)

        self.mapping_layers = {'unet_256':['model.model.1.model.3.model.0',     # 2 * ngf
                                            'model.model.1.model.3.model.3.model.3.model.0',      # 8 * ngf
                                            'model.model.1.model.3.model.3.model.4',      # 16 * ngf
                                            'model.model.1.model.4'],     # 4 * ngf
                                'mobile_resnet_9blocks':['model.9',  # 4 * ngf
                                                         'model.12',
                                                         'model.15',
                                                         'model.18']}
        self.extrac_D = ['block4s.1.block.2']
        self.netTs = nn.ModuleList()
        self.netSs = nn.ModuleList()
        self.Tacts, self.Sacts = {}, {}
        self.Dacts = {}
        G_params = [self.netG_student.parameters()]
        if self.opt.lambda_CD:
            for i, n in enumerate(self.mapping_layers[self.opt.teacher_netG]):
                ft, fs = self.opt.teacher_ngf, self.opt.student_ngf

                if 'resnet' in self.opt.teacher_netG:
                    netT = self.build_feature_connector(opt.input_nc + opt.output_nc, 4 * ft)
                    netS = self.build_feature_connector(opt.input_nc + opt.output_nc, 4 * fs)

                networks.init_net(netS)
                networks.init_net(netT)
                G_params.append(netT.parameters())
                G_params.append(netS.parameters())
                self.netTs.append(netT)
                self.netSs.append(netS)
                self.netTs.cuda()
                self.netSs.cuda()

        self.optimizer_G_student = torch.optim.Adam(itertools.chain(*G_params), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G_teacher = torch.optim.Adam(self.netG_teacher.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D_teacher = torch.optim.Adam(self.netD_teacher.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.optimizers.append(self.optimizer_G_student)
        self.optimizers.append(self.optimizer_G_teacher)
        self.optimizers.append(self.optimizer_D_teacher)

        self.eval_dataloader = create_eval_dataloader(self.opt, direction=opt.direction)
        self.inception_model, self.drn_model = create_metric_models(opt, device=self.device)
        self.npz = np.load(opt.real_stat_path)
        self.is_best = False
        self.loss_D_fake, self.loss_D_real = 0, 0

    def build_feature_connector(self, t_channel, s_channel):
        C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(t_channel),
             nn.ReLU(inplace=True)]

        for m in C:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return nn.Sequential(*C)

    def setup(self, opt, verbose=True):
        self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        self.load_networks(verbose)
        if verbose:
            self.print_networks()
        if self.opt.lambda_CD > 0:
            def get_activation(mem, name):
                def get_output_hook(module, input, output):
                    mem[name + str(output.device)] = output

                return get_output_hook

            def add_hook(net, mem, mapping_layers):
                for n, m in net.named_modules():
                    if n in mapping_layers:
                        m.register_forward_hook(get_activation(mem, n))
                        
            add_hook(self.netD_teacher, self.Dacts, self.extrac_D)
            add_hook(self.netG_teacher, self.Tacts, self.mapping_layers[self.opt.teacher_netG])
            add_hook(self.netG_student, self.Sacts, self.mapping_layers[self.opt.teacher_netG])

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_single_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        raise NotImplementedError

    def backward_D_teacher(self):
        fake_AB = torch.cat((self.real_A, self.Tfake_B), 1).detach()
        real_AB = torch.cat((self.real_A, self.real_B), 1).detach()
        pred_fake = self.netD_teacher(fake_AB)
        self.loss_D_fake = self.criterionGAN(pred_fake, False, for_discriminator=True)
        pred_real = self.netD_teacher(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True, for_discriminator=True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def optimize_parameters(self, steps):
        raise NotImplementedError

    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if hasattr(self, name):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
                with open(os.path.join(self.opt.log_dir, name + '.txt'), 'w') as f:
                    f.write(str(net) + '\n')
                    f.write('[Network %s] Total number of parameters : %.3f M\n' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def load_networks(self, verbose=True):
        if self.opt.restore_student_G_path is not None:
            util.load_network(self.netG_student, self.opt.restore_student_G_path, verbose)
        if self.opt.restore_teacher_G_path is not None:
            util.load_network(self.netG_teacher, self.opt.restore_teacher_G_path, verbose)
        if self.opt.restore_D_path is not None:
            util.load_network(self.netD_teacher, self.opt.restore_D_path, verbose)
        if self.opt.restore_A_path is not None:
            for i, netA in enumerate(self.netAs):
                path = '%s-%d.pth' % (self.opt.restore_A_path, i)
                util.load_network(netA, path, verbose)
        if self.opt.restore_O_path is not None:
            for i, optimizer in enumerate(self.optimizers):
                path = '%s-%d.pth' % (self.opt.restore_O_path, i)
                util.load_optimizer(optimizer, path, verbose)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.opt.lr

    def save_net(self, net, save_path):
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            if isinstance(net, DataParallel):
                torch.save(net.module.cpu().state_dict(), save_path)
            else:
                torch.save(net.cpu().state_dict(), save_path)
            net.cuda(self.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)

    def save_networks(self, epoch):

        save_filename = '%s_net_%s_student.pth' % (epoch, 'G')
        save_path = os.path.join(self.save_dir, save_filename)
        net = getattr(self, 'net%s_student' % 'G')
        self.save_net(net, save_path)

        save_filename = '%s_net%s_teacher.pth' % (epoch, 'G')
        save_path = os.path.join(self.save_dir, save_filename)
        net = getattr(self, 'net%s_teacher' % 'G')
        self.save_net(net, save_path)

        save_filename = '%s_net_%s_teacher.pth' % (epoch, 'D')
        save_path = os.path.join(self.save_dir, save_filename)
        net = getattr(self, 'net%s_teacher' % 'D')
        self.save_net(net, save_path)

        for i, optimizer in enumerate(self.optimizers):
            save_filename = '%s_optim-%d.pth' % (epoch, i)
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(optimizer.state_dict(), save_path)

        if self.opt.lambda_CD:
            for i, net in enumerate(self.netAs):
                save_filename = '%s_net_%s-%d.pth' % (epoch, 'A', i)
                save_path = os.path.join(self.save_dir, save_filename)
                self.save_net(net, save_path)

    def evaluate_model(self, step):
        raise NotImplementedError

    def test(self):
        with torch.no_grad():
            self.forward()

    def get_current_visuals(self):
        """Return visualization images. """
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def profile(self, config=None, verbose=True):
        for name in self.model_names:
            if hasattr(self,name) and 'D' not in name:
                netG = getattr(self,name)
                if isinstance(netG, nn.DataParallel):
                    netG = netG.module
                if config is not None:
                    netG.configs = config
                with torch.no_grad():
                    macs = profile_macs(netG, (self.real_A[:1],))
                    # flops, params = profile(netG, inputs=(self.real_A[:1],))
                params = 0
                for p in netG.parameters():
                    params += p.numel()
                if verbose:
                    print('MACs: %.3fG\tParams: %.3fM' % (macs / 1e9, params / 1e6), flush=True)

        return None
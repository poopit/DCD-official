import torch
from torch import nn
from torchprofile import profile_macs

from models import networks
from .base_model import BaseModel


class TestModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        assert not is_train
        return parser

    def __init__(self, opt):
        super(TestModel, self).__init__(opt)
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, opt.dropout_rate, opt.init_type,
                                      opt.init_gain, self.gpu_ids, opt=opt)
        self.netG.eval()

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        if self.opt.dataset_mode != 'single':
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def optimize_parameters(self):
        assert False, 'This model is only for testing, you cannot optimize the parameters!!!'

    def save_networks(self, epoch):
        assert False, 'This model is only for testing!!!'

    def profile(self, verbose=True):
        netG = self.netG
        if isinstance(netG, nn.DataParallel):
            netG = netG.module
        with torch.no_grad():
            macs = profile_macs(netG, (self.real_A[:1],))
        params = 0
        for p in netG.parameters():
            params += p.numel()
        if verbose:
            print('MACs: %.3fG\tParams: %.3fM' % (macs / 1e9, params / 1e6), flush=True)
        return macs, params

    def test(self):
        with torch.no_grad():
            self.forward()

    def get_current_losses(self):
        assert False, 'This model is only for testing!!!'

    def update_learning_rate(self, f=None):
        assert False, 'This model is only for testing!!!'

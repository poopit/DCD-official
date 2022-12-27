#!/usr/bin/env bash
python distill.py --dataroot database/edges2shoes-r \
  --gpu_ids 3 --print_freq 100 \
  --distiller pix2pixbest \
  --lambda_CD 1 \
  --log_dir logs/resnet_pix2pix/edges2shoes-r/DCD_S16 \
  --batch_size 4 \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --teacher_ngf 64 --student_ngf 16 \
  --teacher_netG mobile_resnet_9blocks \
  --student_netG mobile_resnet_9blocks --netD n_layers \
  --nepochs 100 --nepochs_decay 100 --n_dis 1 \
  --AGD_weights 1e1,1e4,1e1,1e-5
#Please remove all the files and folders under Result/train before training
#Sample use:
#python train.py --dataroot datasets/horse2zebra/  --saveroot checkpoints --batch_size 16
import os
import torch
import model
import argparse
import itertools
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
from utils import recorder
from utils import allocate
from utils import init_weights
from utils import tensor2image
from utils import CustomImgDataset
import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# argparse options here take the reference from the original paper code, as it's designed well
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--cuda', action='store_true', default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--device', type=str, default='cuda:0', help='choose device to use during training')
parser.add_argument('--total_epochs', type=int, default=200, help='total number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')

parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='crop image to (size, size)')
parser.add_argument('--load_size', type=int, default=286, help='scale image to (load_size, load_size)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')

parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--saveroot', type=str, required=True, help='save directory of the output')  # MUST IN FORM: X/Y/


# Followed the paper design
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term beta1 of adam')
parser.add_argument('--beta2', type=float, default=0.999, help='momentum term beta2 of adam')
opt = parser.parse_args()
print(opt)


# Set device, a gpu with high ram is required
device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

# To save models, used torch.nn.DataParallel even with single GPU
# Generators
G_AB = nn.DataParallel(model.g_net(opt.input_nc, opt.output_nc).to(device), device_ids=[opt.device]).apply(init_weights)
G_BA = nn.DataParallel(model.g_net(opt.output_nc, opt.input_nc).to(device), device_ids=[opt.device]).apply(init_weights)

# Discriminators
D_A = nn.DataParallel(model.d_net(opt.input_nc).to(device)).apply(init_weights)
D_B = nn.DataParallel(model.d_net(opt.output_nc).to(device)).apply(init_weights)

# Criterion for loss computing
criterion_GAN = nn.MSELoss()

# CycleGAN paper claimed using nn.MSELoss() for cycle and identity criterion cannot improve the performance
criterion_cycle = nn.L1Loss()   # nn.MSELoss()
criterion_idt = nn.L1Loss()     # nn.MSELoss()


# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(),
                                               G_BA.parameters()),
                               lr=opt.lr, betas=(opt.beta1, opt.beta2))
optimizer_D = torch.optim.Adam(itertools.chain(D_A.parameters(),
                                               D_B.parameters()),
                               lr=opt.lr, betas=(opt.beta1, opt.beta2))


assert(opt.decay_epoch < opt.total_epochs)
lambda1 = lambda epoch: 1.0 - max((epoch + opt.epoch - opt.decay_epoch)/(opt.total_epochs - opt.decay_epoch), 0)
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda1)
lr_scheduler_D= torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda1)



#Memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batch_size, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batch_size, opt.output_nc, opt.size, opt.size)

fake_A_buffer = allocate()
fake_B_buffer = allocate()

#Dataset loader
transforms_ = transforms.Compose([transforms.Resize([opt.load_size, opt.load_size], transforms.InterpolationMode.BICUBIC),
                                  transforms.RandomCrop(opt.size),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataloader = DataLoader(CustomImgDataset(opt.dataroot, transform=transforms_, data_mode='Unaligned', train=True),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)


# We want dir_to_save in form: X/Y/
dir_to_save = opt.saveroot
assert(dir_to_save[0] != '/')
assert(dir_to_save[-1] == '/')
dir_to_save = "Result/train/" + dir_to_save


# Loss recorder initialize
Recorder = recorder(opt.total_epochs, len(dataloader))
os.makedirs('Result/train/saveimgs', exist_ok=True)
os.makedirs('Result/train/saveloss', exist_ok=True)


print("Training start:")
force_saveimag= 0
for epoch in range(opt.epoch, opt.total_epochs):
    imagedict = dict()
    lossdict = dict()
    for i, batch in enumerate(dataloader):
        if len(batch['A']) != opt.batch_size:
            force_saveimag = 1
            Recorder.record(recordloss=lossdict, epochs=epoch+1, iter=i, force_print=1)
            continue
        if len(batch['B']) != opt.batch_size:
            force_saveimag = 1
            Recorder.record(recordloss=lossdict, epochs=epoch+1, iter=i, force_print=1)
            continue

        real_A = input_A.copy_(batch['A'])
        real_B = input_B.copy_(batch['B'])

        optimizer_G.zero_grad()

        # Identity loss
        idt_B = G_AB(real_B)
        idt_A = G_BA(real_A)
        idt_loss_B = criterion_idt(idt_B, real_B) * 5.0  # 10 * 0.5
        idt_loss_A = criterion_idt(idt_A, real_A) * 5.0  # 10 * 0.5

        # GAN loss
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        pred_fake_B = D_B(fake_B)
        pred_fake_A = D_A(fake_A)
        real1 = torch.full(pred_fake_B.shape, 1.0).to(device)
        real2 = torch.full(pred_fake_A.shape, 1.0).to(device)
        gan_loss_AB = criterion_GAN(pred_fake_B, real1)
        gan_loss_BA = criterion_GAN(pred_fake_A, real2)

        # Cycle loss ABA and BAB
        rec_A = G_BA(fake_B)
        rec_B = G_AB(fake_A)
        cycle_loss_ABA = criterion_cycle(rec_A, real_A) * 10.0  # 10 followed the value used by authors
        cycle_loss_BAB = criterion_cycle(rec_B, real_B) * 10.0  # 10 followed the value used by authors

        # Total:
        loss_G = idt_loss_A + idt_loss_B + gan_loss_AB + gan_loss_BA + cycle_loss_ABA + cycle_loss_BAB
        loss_G.backward()

        optimizer_G.step()

        # Discriminators
        # net D_A
        optimizer_D.zero_grad()

        pred_real = D_A(real_A)
        fake_A = fake_A_buffer.stack(fake_A)
        pred_fake = D_A(fake_A.detach())
        real = torch.full(pred_real.shape, 1.0).to(device)
        fake = torch.full(pred_fake.shape, 0.0).to(device)
        # loss for real
        loss_D_real = criterion_GAN(pred_real, real)
        # loss for fake
        loss_D_fake = criterion_GAN(pred_fake, fake)
        # Total loss of D_A
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5  # 0.5 followed the value used by authors
        loss_D_A.backward()
        optimizer_D.step()

        # net D_B
        optimizer_D.zero_grad()

        pred_real = D_B(real_B)
        fake_B = fake_B_buffer.stack(fake_B)
        pred_fake = D_B(fake_B.detach())
        real = torch.full(pred_real.shape, 1.0).to(device)
        fake = torch.full(pred_fake.shape, 0.0).to(device)
        # Real loss
        loss_D_real = criterion_GAN(pred_real, real)
        # Fake loss
        loss_D_fake = criterion_GAN(pred_fake, fake)

        # Total loss of D_B
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5  # 0.5 followed the value used by authors
        loss_D_B.backward()
        optimizer_D.step()

        imagedict['real_A'], imagedict['fake_B'], imagedict['recovered_A'] = real_A, fake_B, rec_A
        imagedict['real_B'], imagedict['fake_A'], imagedict['recovered_B'] = real_B, fake_A, rec_B

        lossdict['loss_G_identity'], lossdict['loss_G_GAN'] = (idt_loss_A + idt_loss_B), (gan_loss_AB + gan_loss_BA)
        lossdict['loss_G_cycle'], lossdict['loss_G'] = (cycle_loss_ABA + cycle_loss_BAB), loss_G
        lossdict['loss_D_A'], lossdict['loss_D_B'], lossdict['loss_D'] = loss_D_A, loss_D_B, (loss_D_A + loss_D_B)

        # Since loss_D is just for observation and does not affect the training, we mannually set loss_D=loss_D_A + loss_D_B to clearly show the curves.
        Recorder.record(recordloss=lossdict, epochs=epoch+1, iter=i)

        if ((epoch+1) % 5) == 0:
            if i + 2 == len(dataloader) and force_saveimag == 1:  # save image every 5 epochs
                for img_name, img_tensor in imagedict.items():
                    Image.fromarray(tensor2image(img_tensor.data)).save(
                        "Result/train/saveimgs/epoch" + str(epoch+1) + "_" + img_name + ".jpg")
            elif i+1 == len(dataloader):
                for img_name, img_tensor in imagedict.items():
                    Image.fromarray(tensor2image(img_tensor.data)).save(
                        "Result/train/saveimgs/epoch" + str(epoch+1) + "_" + img_name + ".jpg")

    # Save models checkpoints
    Path(dir_to_save).mkdir(parents=True, exist_ok=True)
    torch.save(G_AB.module.state_dict(), dir_to_save + 'G_AB.pth')
    torch.save(G_BA.module.state_dict(), dir_to_save + 'G_BA.pth')
    torch.save(D_A.module.state_dict(), dir_to_save + 'D_A.pth')
    torch.save(D_B.module.state_dict(), dir_to_save + 'D_B.pth')

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D.step()


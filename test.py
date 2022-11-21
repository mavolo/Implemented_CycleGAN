# python test.py --dataroot datasets/horse2zebra/  --path_gA2B Result/train/checkpoints/G_AB.pth --path_gB2A Result/checkpoints/G_BA.pth --saveroot horse2zebra/

import os
import sys
import argparse
from model import g_net
from utils import CustomImgDataset
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# argparse options here take the reference from the original paper code, as it's designed well
parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=256, help='size of the resized data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--device', type=str, default='cuda:0', help='choose device to use during training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')

parser.add_argument('--mode', type=str, default='two_sides', help='choose test mode (select two_sides or A2B or B2A)')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='directory of the dataset')
parser.add_argument('--saveroot', type=str, required=True, help='save directory of the output')
parser.add_argument('--path_gA2B', type=str, default='output/G_AB.pth', help='A2B generator checkpoint file')
parser.add_argument('--path_gB2A', type=str, default='output/G_BA.pth', help='B2A generator checkpoint file')
parser.add_argument('--channel', type=int, default=3, help='number of channels')

opt = parser.parse_args()

# Set device, a gpu with high ram is required
device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor
# Data loader
transforms_ = transforms.Compose([transforms.Resize([opt.img_size, opt.img_size], transforms.InterpolationMode.BICUBIC),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataloader = DataLoader(CustomImgDataset(opt.dataroot, transform=transforms_, data_mode='Unaligned', train=False),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
# Prepare save root
dir_to_save = opt.saveroot
dir_to_save = "Result/test/" + dir_to_save

# prepare A2B
if opt.mode == 'two_sides' or opt.mode == 'A2B':
    # Generators
    G_AB = g_net(opt.channel, opt.channel).to(device)
    G_AB.load_state_dict(torch.load(opt.path_gA2B))
    # Set test mode
    G_AB.eval()
    # memory allocation
    input_A = Tensor(opt.batch_size, opt.channel, opt.img_size, opt.img_size).to(device)

# prepare B2A
if opt.mode == 'two_sides' or opt.mode == 'B2A':
    # Generators
    G_BA = g_net(opt.channel, opt.channel).to(device)
    G_BA.load_state_dict(torch.load(opt.path_gB2A))
    G_BA.to(device)
    # Set test mode
    G_BA.eval()
    # memory allocation
    input_B = Tensor(opt.batch_size, opt.channel, opt.img_size, opt.img_size).to(device)



# test A2B
if opt.mode == 'two_sides' or opt.mode == 'A2B':
# directory
    os.makedirs(dir_to_save + 'A2B', exist_ok=True)
# remove image format of A
    filename_A = []
    for file in os.listdir(opt.dataroot + "testA"):
        filename_A.append(file[:-4]) if (file.endswith(".jpg") or file.endswith(".png")) else exit("image format error")

    for i, batch in enumerate(dataloader):
        if i <= len(filename_A) - 1:
            # Input
            real_A = Variable(input_A.copy_(batch['A']))
            # Translate
            fake_B = 0.5 * (G_AB(real_A).data + 1.0)
            real_A = 0.5 * (real_A.data + 1.0)
            # Save
            save_image(real_A, dir_to_save + 'A2B/' + filename_A[i] + '.png')
            save_image(fake_B, dir_to_save + 'A2B/'+filename_A[i]+'_fake.png')
            sys.stdout.write('\rTest A2B images %04d of %04d done!' % (i + 1, len(filename_A)))
    print('Test A2B finished!')

# test B2A
if opt.mode == 'two_sides' or opt.mode == 'B2A':
    # directory
    os.makedirs(dir_to_save + 'B2A', exist_ok=True)
# remove image format of A
    filename_B = []
    for file in os.listdir(opt.dataroot + "testB"):
        filename_B.append(file[:-4]) if (file.endswith(".jpg") or file.endswith(".png")) else exit("image format error")

    for i, batch in enumerate(dataloader):
        if i <= len(filename_B) - 1:
            # Input
            real_B = Variable(input_B.copy_(batch['B']))
            # Translate
            fake_A = 0.5 * (G_BA(real_B).data + 1.0)
            real_B = 0.5 * (real_B.data + 1.0)
            # Save
            save_image(real_B, dir_to_save + 'B2A/' + filename_B[i] + '.png')
            save_image(fake_A, dir_to_save + 'B2A/'+filename_B[i]+'_fake.png')
            sys.stdout.write('\rTest B2A images %04d of %04d done!' % (i + 1, len(filename_B)))
    print('Test B2A finished!')
print('Test finished!')



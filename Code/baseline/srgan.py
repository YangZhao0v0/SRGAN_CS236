import argparse
import numpy as np
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import Generator, Discriminator, FeatureExtractor
import tqdm
import matplotlib.pyplot as pyplot
import utils as ut
from torch.autograd import Variable
import copy

#########################################
#                                       #
# Argument preprocessing:               #
#                                       #
#########################################


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--iter_max',  type=int, default=2000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=1000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
parser.add_argument('--pretrain',  type=int, default=1,     help="Flag for pre-training")
parser.add_argument('--batch_size',type=int, default=100,     help="batch size")
parser.add_argument('--gen_w',     type=str, default='', help="path to generator weights")
parser.add_argument('--dis_w',     type=str, default='', help="path to discriminator weights")
args = parser.parse_args()


#########################################
#                                       #
# Data Loading & Preprocessing module:  #
#                                       #
#########################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Current test is carried on Tiny imageNet dataset, need to reshape to 96 * 96 size.
# Loading data for training, val and test set:
data_transforms = {
	'train': transforms.Compose([transforms.Resize((96,96)),
	    transforms.ToTensor(),
	    ]),
	    'val': transforms.Compose([transforms.Resize((96,96)),
	        transforms.ToTensor(),
	    ]),
	    'test': transforms.Compose([transforms.Resize((96,96)),
	        transforms.ToTensor(),
	    ])
	}


data_dir = '/home/yangz16/Class/CS236/Project/tiny-imagenet-200/'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
	                                          data_transforms[x])
	                  for x in ['train', 'val','test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
	                                             shuffle=True, num_workers=10)
	              for x in ['train', 'val', 'test']}

#########################################
#                                       #
# GAN training module:                  #
#                                       #
#########################################

# G & D initialization:
generator = Generator(num_ResBlock = 4, upFactor = 4) 
if args.gen_w != '':
    generator.load_state_dict(torch.load(args.gen_w))
print (generator)

discriminator = Discriminator()
if args.dis_w != '':
	discriminator.load_state_dict(torch.load(args.dis_w))
print (discriminator)

generator = generator.to(device)
discriminator = discriminator.to(device)

# Loss criterion:
encoding = FeatureExtractor(torchvision.models.vgg19(pretrained=True)).to(device)

content_criterion = nn.MSELoss().to(device)
adversarial_criterion = nn.BCELoss().to(device)
ones = torch.nn.Parameter(torch.ones(args.batch_size,1), requires_grad=False).to(device)
zeros = torch.nn.Parameter(torch.zeros(args.batch_size,1), requires_grad=False).to(device)

optim_generator = optim.Adam(generator.parameters(), lr=1e-4)

# The transformation that downscale the image to get the low resolution counterpart.
scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((24,24)),
                            transforms.ToTensor(),
                            ])

# Stage 1: Pre-train generator using MSE loss:


print ('Stage 1: Pre-train generator ...')
if args.pretrain:
	with tqdm.tqdm(total=1000) as pbar:
		for epoch in range(1):
			for batch_id, (image, _) in enumerate(dataloaders['train']):
				image = image.to(device)
				size = image.size()
				# Get low resolution counterparts:
				low_res = torch.FloatTensor(size[0], 3, 24, 24).to(device)
				for i in range(size[0]):
					low_res[i] = scale(image[i])
				high_res_fake = generator(low_res)
				optim_generator.zero_grad()
				loss = content_criterion(image, high_res_fake)
				loss.backward()
				optim_generator.step()
				pbar.set_postfix(loss='{:.2e}'.format(loss))
				pbar.update(1)
			

	# Save pretrained weights:
	save_dir = os.path.join('checkpoints','srgan')
	if not os.path.exists(save_dir):
	    os.makedirs(save_dir)
	file_path = os.path.join(save_dir, 'model-pre.pt')
	state = generator.state_dict()
	torch.save(state, file_path)


	# Reconstruct several images:
	save_dir = os.path.join('reconstructions','pretrained')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	for i in range(5):
		file_path = os.path.join(save_dir, 'image' + str(i) + '.png')
		idx = np.random.randint(0,args.batch_size,)
		ut.Reconstruct(low_res[idx], image[idx], high_res_fake[idx], file_path)

# tensor placeholders
high_res = torch.FloatTensor(args.batch_size, 3, 96, 96)
high_res_fake = torch.FloatTensor(args.batch_size, 3, 96, 96)
output_fake = torch.FloatTensor(args.batch_size)
output_real = torch.FloatTensor(args.batch_size)
low_res = torch.FloatTensor(args.batch_size, 3, 24, 24).to(device)
high_res = Variable(high_res)
high_res_fake = Variable(high_res_fake)
output_fake = Variable(output_fake)
output_real = Variable(output_real)
low_res = Variable(low_res)


# Stage 2: Train the GAN with content + adversary loss:

optim_generator = optim.Adam(generator.parameters(), lr=1e-4)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=3e-5)

print ('Stage 2: Train the GAN ...')
if args.train:
	log_lossG = []
	log_lossG_content = []
	log_lossG_adv = []
	log_lossD = []
	with tqdm.tqdm(total=args.iter_max) as pbar:
		iter_count = 0
		while True:
			if iter_count > args.iter_max:
				break
			for batch_id, (image, _) in enumerate(dataloaders['train']):
				image = image.to(device)
				size = image.size()
				high_res.data.resize_as_(image).copy_(image)
				# Get low resolution counterparts:
				for i in range(size[0]):
					low_res[i] = scale(high_res[i])
				high_res_fake = generator(low_res)

				# --- Train the discriminator: --- #
				
				optim_discriminator.zero_grad()
				output_real = discriminator(high_res)
				output_fake = discriminator(high_res_fake.detach())
				lossD = adversarial_criterion(output_real, ones) + adversarial_criterion(output_fake, zeros)
				log_lossD.append(lossD.detach())
				lossD.backward()
				optim_discriminator.step()
		
				
				# --- Train the generator: --- #
				optim_generator.zero_grad()
				real_feature = encoding(high_res)
				#high_res_fake = generator(low_res)
				fake_feature = encoding(high_res_fake)
				output_fake = discriminator(high_res_fake)
				lossG_content = content_criterion(real_feature, fake_feature) + 1e-3 * content_criterion(high_res, high_res_fake)
				lossG_adv = 0.2 * adversarial_criterion(output_fake, ones)
				lossG = lossG_content + lossG_adv
				log_lossG.append(lossG.detach())
				log_lossG_content.append(lossG_content.detach())
				log_lossG_adv.append(lossG_adv.detach())
				lossG.backward()
				optim_generator.step()
				pbar.set_postfix(lossD='{:.2e}'.format(lossD),
								 lossG='{:.2e}'.format(lossG))
				pbar.update(1)	
				iter_count += 1

				if iter_count % args.iter_save == 0:
					# Save trained weights:
					save_dir = os.path.join('checkpoints','srgan')
					if not os.path.exists(save_dir):
					    os.makedirs(save_dir)
					# Save weights for generator:
					file_path = os.path.join(save_dir, 'gen-train' + str(iter_count) + '.pt')
					state = generator.state_dict()
					torch.save(state, file_path)
					# Save weights for discriminator:
					file_path = os.path.join(save_dir, 'dis-train' + str(iter_count) + '.pt')
					state = discriminator.state_dict()
					torch.save(state, file_path)

					# Reconstruct several images:
					save_dir = os.path.join('reconstructions','trained')
					if not os.path.exists(save_dir):
						os.makedirs(save_dir)
					for i in range(5):
						file_path = os.path.join(save_dir, 'image_' + str(iter_count) + '_' + str(i) + '.png')
						idx = np.random.randint(0,args.batch_size)
						ut.Reconstruct(low_res[idx], image[idx], high_res_fake[idx], file_path)

					# Get the learning curves:
					pyplot.close('all') 
					save_dir = os.path.join('learningCurves','trained')
					if not os.path.exists(save_dir):
						os.makedirs(save_dir)
					# Discriminator learning curve:
					file_path = os.path.join(save_dir, 'lossD' + str(iter_count) + '.png')
					ut.LearningCurve([log_lossD], file_path, False)
					# Generator learning curve			
					file_path = os.path.join(save_dir, 'lossG' + str(iter_count) + '.png')
					ut.LearningCurve([log_lossG,log_lossG_content, log_lossG_adv], file_path, True)

				if iter_count > args.iter_max:
					break

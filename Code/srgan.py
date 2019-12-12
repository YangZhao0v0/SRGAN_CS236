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
import pytorch_ssim

#########################################
#                                       #
# Argument preprocessing:               #
#                                       #
#########################################


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num_epochs',  type=int, default=100, help="Number of training iterations")
parser.add_argument('--num_save',    type=int, default=10,  help="Save model every n iterations")
parser.add_argument('--run',         type=int, default=0,   help="Run ID. In case you want to run replicates")
parser.add_argument('--batch_size',  type=int, default=100, help="batch size")
parser.add_argument('--gen_w',       type=str, default='',  help="path to generator weights")
parser.add_argument('--dis_w',       type=str, default='',  help="path to discriminator weights")
parser.add_argument('--upFactor',    type=int, default=4,   help="upscaling factor")
parser.add_argument('--num_res',     type=int, default=8,   help="number of residual blocks in the generator")
parser.add_argument('--crop_size',   type=int, default=96,  help="crop size for training images")
parser.add_argument('--validation',  type=int, default=0,   help="whether to validate")
args = parser.parse_args()


#########################################
#                                       #
# Data Loading & Preprocessing module:  #
#                                       #
#########################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Loading data for training and test set:

train_data_dir = 'JPEGImages/'
test_data_dir = 'valset/'
train_set = ut.TrainDatasetFromFolder(train_data_dir, crop_size=args.crop_size, upscale_factor=args.upFactor)
train_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=4, batch_size=args.batch_size, shuffle=True)
val_set = ut.ValDatasetFromFolder(test_data_dir, crop_size=args.crop_size, upscale_factor=args.upFactor)
val_loader = torch.utils.data.DataLoader(dataset=val_set, num_workers=4, batch_size=args.batch_size, shuffle=True)
#########################################
#                                       #
# GAN training module:                  #
#                                       #
#########################################

# Step 1 --- G & D initialization:
generator = Generator(num_ResBlock = args.num_res, upFactor = args.upFactor) 
if args.gen_w != '':
	print ('loading weights of the generator ...')
	generator.load_state_dict(torch.load(args.gen_w))
print (generator)

discriminator = Discriminator()
if args.dis_w != '':
	print ('loading weights of the discriminator ...')
	discriminator.load_state_dict(torch.load(args.dis_w))
print (discriminator)

# Put both G & D to device:
generator = generator.to(device)
discriminator = discriminator.to(device)

# Loss criterion:
encoding = FeatureExtractor(torchvision.models.vgg19(pretrained=True)).to(device) # VGG19 encoding;
content_criterion = nn.MSELoss().to(device)
adversarial_criterion = nn.BCELoss().to(device)
ones = torch.nn.Parameter(torch.ones(args.batch_size), requires_grad=False).to(device)
zeros = torch.nn.Parameter(torch.zeros(args.batch_size), requires_grad=False).to(device)

# tensor placeholders
'''
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
'''
if (args.validation):
# perform validation on the val_set:
	with torch.no_grad():
		val_bar = tqdm.tqdm(val_loader)
		val_results = {'batch_sizes':0,'mse': 0, 'ssim': 0, 'psnr': 0, 'mse_b': 0, 'ssim_b': 0, 'psnr_b': 0}
		for val_lr, val_hr, val_hr_bicubic in val_bar:
			batch_size = val_lr.size(0)
			val_results['batch_sizes'] += 1.
			lr = val_lr
			hr = val_hr
			lr = lr.to(device)
			hr = hr.to(device)
			val_hr_bicubic = val_hr_bicubic.to(device)
			sr = generator(lr)
			batch_mse = content_criterion(sr, hr)
			batch_mse_b = content_criterion(val_hr_bicubic, hr)
			val_results['mse'] += batch_mse
			val_results['mse_b'] += batch_mse_b
			batch_ssim = 1
			batch_ssim_b = 1
			val_results['ssim'] += batch_ssim 
			val_results['ssim_b'] += batch_ssim_b
			val_results['psnr'] = 10 * torch.log10(1 / (val_results['mse'] / val_results['batch_sizes']))
			val_results['psnr_b'] = 10 * torch.log10(1 / (val_results['mse_b'] / val_results['batch_sizes']))
			val_bar.set_description(
	                    desc='[converting LR images to SR images] PSNR: %.4f SSIM: %.4f MSE: %.4f PSNR_b: %.4f SSIM_b: %.4f MSE_b: %.4f' % (
	                        val_results['psnr'], val_results['ssim'] / val_results['batch_sizes'], val_results['mse'] / val_results['batch_sizes'], val_results['psnr_b'], val_results['ssim_b'] / val_results['batch_sizes'], val_results['mse_b'] / val_results['batch_sizes']))	
# Train the GAN with content + adversary loss:

optim_generator = optim.Adam(generator.parameters(), lr=1e-4)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=1e-4)

train_log = {'lossD':[],'lossG':[], 'lossG_adv':[],'lossG_content':[]}
print ('Stage 2: Train the GAN ...')

stats = {'num_batches':0,'lossG':0,'lossD':0,'lossG_adv':0,'lossG_content':0}
for epoch in range(args.num_epochs):
	train_bar = tqdm.tqdm(train_loader)
	#stats = {'num_batches':0,'lossG':0,'lossD':0,'lossG_adv':0,'lossG_content':0}
	for lr_image, hr_image in train_bar:
		### Update discriminator ###
		optim_discriminator.zero_grad()
		high_res = Variable(hr_image).to(device)
		low_res = Variable(lr_image).to(device)
		high_res_fake = generator(low_res)
		output_real = discriminator(high_res)
		output_fake = discriminator(high_res_fake)
		lossD = adversarial_criterion(output_real, ones) + adversarial_criterion(output_fake, zeros)
		if args.run != 2:
			lossD.backward(retain_graph=True)
			optim_discriminator.step()

		### Update generator ###
		optim_generator.zero_grad()
		real_feature = encoding(high_res)
		#high_res_fake = generator(low_res)
		fake_feature = encoding(high_res_fake)
		output_fake = discriminator(high_res_fake)
		if args.run == 1:
			lossG_content = content_criterion(real_feature, fake_feature)
		elif args.run == 2:
			lossG_content = 5e-3 * content_criterion(real_feature, fake_feature) + content_criterion(high_res, high_res_fake)
		else:
			lossG_content = 5e-3 * content_criterion(real_feature, fake_feature) + content_criterion(high_res, high_res_fake)
		lossG_adv = 1e-3 * adversarial_criterion(output_fake, ones)
		if args.run == 2:
			lossG = lossG_content
		else:
			lossG = lossG_content + lossG_adv
		lossG.backward()
		optim_generator.step()	

		stats['num_batches'] += 1.
		stats['lossG'] += lossG.item()
		stats['lossG_adv'] += lossG_adv.item()
		stats['lossG_content'] += lossG_content.item()
		stats['lossD'] += lossD.item()
		train_log['lossD'].append(stats['lossD'] / stats['num_batches'])
		train_log['lossG'].append(stats['lossG'] / stats['num_batches'])
		train_log['lossG_adv'].append(stats['lossG_adv'] / stats['num_batches'])
		train_log['lossG_content'].append(stats['lossG_content'] / stats['num_batches'])

		train_bar.set_description(desc='[%d/%d] LossD: %.4f LossG: %.4f LossG_adv: %.4f LossG_content: %.4f' % (
        epoch, args.num_epochs, stats['lossD'] / stats['num_batches'], stats['lossG'] / stats['num_batches'], stats['lossG_adv'] / stats['num_batches'], stats['lossG_content'] / stats['num_batches']))	
	# Reconstruct several images:
	save_dir = os.path.join('reconstructions','trained')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	for i in range(5):
		file_path = os.path.join(save_dir, 'image_' + '_' + str(i) + '_run' + str(args.run) + str(args.upFactor) + 'xup.png')
		idx = np.random.randint(0,args.batch_size)
		ut.Reconstruct(lr_image[idx].cpu(), hr_image[idx].cpu(), high_res_fake[idx].cpu(), file_path)

	# Get the learning curves:
	pyplot.close('all') 
	save_dir = os.path.join('learningCurves','trained')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	# Discriminator learning curve:
	file_path = os.path.join(save_dir, 'lossD'  + '_run' + str(args.run) + str(args.upFactor) + 'xup.png')
	ut.LearningCurve([train_log['lossD']], file_path, False)
	# Generator learning curve			
	file_path = os.path.join(save_dir, 'lossG' + '_run' + str(args.run) + str(args.upFactor) + 'xup.png')
	ut.LearningCurve([train_log['lossG'],train_log['lossG_content'], train_log['lossG_adv']], file_path, True)

	#train_log['lossD'].append(stats['lossD'] / stats['num_batches'])
	#train_log['lossG'].append(stats['lossG'] / stats['num_batches'])
	#train_log['lossG_adv'].append(stats['lossG_adv'] / stats['num_batches'])
	#train_log['lossG_content'].append(stats['lossG_content'] / stats['num_batches'])


	if (epoch + 1) % args.num_save == 0:
		# Save trained weights:
		save_dir = os.path.join('checkpoints','srgan')
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		# Save weights for generator:
		file_path = os.path.join(save_dir, 'gen-train' + '_run' + str(args.run) + '_upFactor' + str(args.upFactor) + '.pt')
		state = generator.state_dict()
		torch.save(state, file_path)
		# Save weights for discriminator:
		file_path = os.path.join(save_dir, 'dis-train' + '_run' + str(args.run) + '_upFactor' + str(args.upFactor)+ '.pt')
		state = discriminator.state_dict()
		torch.save(state, file_path)

		# Reconstruct several images:
		save_dir = os.path.join('reconstructions','trained')
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		for i in range(5):
			file_path = os.path.join(save_dir, 'image_' + str(epoch) + '_' + str(i)  + '_run' + str(args.run) + str(args.upFactor) + 'xup.png')
			idx = np.random.randint(0,args.batch_size)
			ut.Reconstruct(lr_image[idx].cpu(), hr_image[idx].cpu(), high_res_fake[idx].cpu(), file_path)

		# Get the learning curves:
		pyplot.close('all') 
		save_dir = os.path.join('learningCurves','trained')
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		# Discriminator learning curve:
		file_path = os.path.join(save_dir, 'lossD' + str(epoch) + '_run' + str(args.run) + str(args.upFactor) + 'xup.png')
		ut.LearningCurve([train_log['lossD']], file_path, False)
		# Generator learning curve			
		file_path = os.path.join(save_dir, 'lossG' + str(epoch)  + '_run' + str(args.run) + str(args.upFactor) + 'xup.png')
		ut.LearningCurve([train_log['lossG'],train_log['lossG_content'], train_log['lossG_adv']], file_path, True)


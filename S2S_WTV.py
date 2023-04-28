import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import self2self

class soft(nn.Module):
	def __init__(self):
		super(soft, self).__init__()
	def forward(self, x, lam):
		x_abs = x.abs()-lam
		zeros = x_abs - x_abs
		n_sub = torch.max(x_abs, zeros)
		x_out = torch.mul(torch.sign(x), n_sub)
		return x_out
soft_thres = soft()

def image_loader(image, device):
	image = image.transpose(2, 0, 1)
	image = image.astype(np.float32)
	image = torch.tensor(image)
	image = image.float()
	image = image.unsqueeze(0)
	return image.to(device)



if __name__ == "__main__":
	##Enable GPU
	USE_GPU = True
	dtype = torch.float32
	if USE_GPU and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	print('using device:', device)
	model = self2self(1,0.4)
	img = np.load('sigmoid_noisy.npy')
	imgmax = img.max()
	imgmin = img.min()
	img = (img-imgmin)/(imgmax-imgmin)
	img = img[:,:,np.newaxis]
	learning_rate = 1e-4
	#set GPU device
	torch.cuda.set_device(0)
	model = model.cuda()
	optimizer = optim.Adam(model.parameters(), lr = learning_rate)
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 5000, gamma=0.8)
	w,h,c = img.shape
	p=0.5
	NPred=100
	thres = 0.1
	mu = 0.1
	weight = torch.ones(1,1,w,h-1).cuda()
	t = torch.zeros((1,c,w,h-1),device=0)
	mu_t = torch.zeros((1,c,w,h-1),device=0)

	#iteration number
	for itr in range(5000):
		#trace
		p_mtx = np.random.uniform(size=[1, img.shape[1], img.shape[2]])
		p_mtx = np.repeat(p_mtx, img.shape[0], axis=0)
		mask = (p_mtx>p).astype(np.double)
		img_input = img
		y = img
		img_input_tensor = image_loader(img_input, device)
		y = image_loader(y, device)
		mask = np.expand_dims(np.transpose(mask,[2,0,1]),0)
		mask = torch.tensor(mask).to(device, dtype=torch.float32)


		# train
		model.train()
		img_input_tensor = img_input_tensor*mask
		img_input_tensor = model(img_input_tensor, mask)
		D_h_ = img_input_tensor[:,:,:,1:] - img_input_tensor[:,:,:,:-1]
		D_h = D_h_.clone().detach()

		#ADMM
		if itr == 0: 
			global D_1,V_1
			D_1 = torch.zeros(1,1,w,h-1).cuda()
			thres_tv = thres * torch.ones(1,1,w,h-1).cuda()
			V_1 = D_h.type(dtype)
		V_1 = soft_thres(D_h + D_1 / mu, weight * thres_tv)
		loss = torch.norm((img_input_tensor-y)*(1-mask),2)
		loss = loss + mu/2 * torch.norm(D_h_-(V_1-D_1/mu),2)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		lr_scheduler.step()
		
		D_1 = (D_1 + mu * (D_h  - V_1)).clone().detach()
		print("iteration %d, loss = %.4f" % (itr+1, loss.item()*100))


		if (itr+1)%100 == 0:
			weight = (torch.div(torch.pow(torch.norm(img_input_tensor-y,2),2)/(2*h*w),torch.sqrt(torch.pow(D_h,2))+1e-2)).detach().clone()
			
		#Inference Strategy
		if (itr+1)%1000 == 0:
			model.train()
			sum_preds = np.zeros((img.shape[0],img.shape[1],img.shape[2]))
			#NPred is the sampling number at the inference stage
			for j in range(NPred):
				p_mtx = np.random.uniform(size=[1, img.shape[1], img.shape[2]])
				p_mtx = np.repeat(p_mtx, img.shape[0], axis=0)
				mask = (p_mtx>p).astype(np.double)
				img_input = img*mask
				img_input_tensor = image_loader(img_input, device)
				mask = np.expand_dims(np.transpose(mask,[2,0,1]),0)
				mask = torch.tensor(mask).to(device, dtype=torch.float32)
				img_input_tensor = model(img_input_tensor,mask)
				sum_preds[:,:,:] += np.transpose(img_input_tensor.detach().cpu().numpy(),[2,3,1,0])[:,:,:,0]
			avg_preds = np.squeeze(sum_preds/NPred)
			write_img = avg_preds
			write_img = np.array(write_img)
			write_img = write_img*(imgmax-imgmin)+imgmin
			np.save('sigmoid_denoise'+str(itr+1)+'.npy', write_img)

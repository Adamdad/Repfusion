import torch
import torch.nn as nn
import torch.nn.functional as F

'''
AT with sum of absolute values with power p
'''
class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p):
		super(AT, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		if fm_s.shape[2] != fm_t.shape[2] or fm_s.shape[3] != fm_t.shape[3]:
			fm_s = F.interpolate(fm_s, (fm_t.size(2), fm_t.size(3)), mode='bilinear')
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am
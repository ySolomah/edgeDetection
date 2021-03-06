import numpy as np
import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import math
from torch.autograd import Variable

im = Image.open("test.png");


print('Hello');

transform = transforms.Compose([transforms.ToTensor()]);
edge_img = transform(im);
print("size of img: " + str(edge_img.shape));
h = edge_img.shape[1];
w = edge_img.shape[2];
edge_img.resize_(1, 3, h, w);

x = torch.FloatTensor(1, 3, 1, 1).zero_();

# RGB -> Luma
for i in range(0, 1):
	x[i][0][0][0] = 0.2126;
	x[i][1][0][0] = 0.7152;
	x[i][2][0][0] = 0.0722;


print(x);

result = torch.nn.functional.conv2d(Variable(edge_img), Variable(x))

print("size of output: " + str(result.shape));

torchvision.utils.save_image(result.data, "test_bw.png")

sobel_filter_x = torch.FloatTensor(1, 1, 3, 3).zero_();
sobel_filter_y = torch.FloatTensor(1, 1, 3, 3).zero_();


sobel_filter_x[0][0][0][0] = 1;
sobel_filter_x[0][0][0][1] = 1;
sobel_filter_x[0][0][0][2] = 1;

sobel_filter_x[0][0][2][0] = -1;
sobel_filter_x[0][0][2][1] = -1;
sobel_filter_x[0][0][2][2] = -1;

sobel_filter_x[0][0][0][0] = 1;
sobel_filter_x[0][0][1][0] = 1;
sobel_filter_x[0][0][2][0] = 1;

sobel_filter_x[0][0][0][2] = -1;
sobel_filter_x[0][0][1][2] = -1;
sobel_filter_x[0][0][2][2] = -1;

g_y = torch.nn.functional.conv2d(result, Variable(sobel_filter_x));
g_x = torch.nn.functional.conv2d(result, Variable(sobel_filter_y));

edgeResult =  torch.sqrt(torch.pow(g_y, 2) + torch.pow(g_x, 2));

print("max of tensor: " + str(torch.max(edgeResult)) + " min of tensor: " + str(torch.min(edgeResult)));

torchvision.utils.save_image(edgeResult.data, "test_edge.png");

tensorMax = (torch.max(edgeResult))*0.5;

whitenedResult = tensorMax - edgeResult;

torchvision.utils.save_image(whitenedResult.data, "test_whitened.png");


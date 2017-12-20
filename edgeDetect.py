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

x = torch.FloatTensor(3, 3, 1, 1).zero_();

# RGB -> Luma
for i in range(0, 3):
	x[i][0][0][0] = 0.2126;
	x[i][1][0][0] = 0.7152;
	x[i][2][0][0] = 0.0722;


print(x);

result = torch.nn.functional.conv2d(Variable(edge_img), Variable(x))

print("size of output: " + str(result.shape));

torchvision.utils.save_image(transform(im) - result.data, "test_bw.jpg")
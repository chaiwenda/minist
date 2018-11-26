# date:2018/11/24 15:08
# -*- coding: utf-8 -*-
# author;cwd
"""
function:
    tensor张量
"""
import torch
import numpy as np
from torch.autograd import Variable  # torch 中 Variable 模块

x = Variable(torch.Tensor([1]), requires_grad = True)
w = Variable(torch.Tensor([2]), requires_grad = True)
d = Variable(torch.Tensor([3]), requires_grad = True)
y = x + w + d
y.backward()
print(y)


# a = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [9, 8, 7, 6, 5]] )
# print("a is {}".format(a))
# print("size of a is:" + str(a.size()))
# print(a.size())

# b = torch.zeros((3, 2))
# print(format(b))
#
# d = torch.randn((3, 2))
# # print(format(d))
# d[2][1] = 100
# # print(format(d))
#
# numpy_d = d.numpy()
# print(format(numpy_d))

# e = np.array([[1, 23], [2, 4]])
# torch_e = torch.from_numpy(e)
# # print(format(torch_e))
# f_torch = torch_e.float()
# print(format(f_torch))

# print(torch.cuda.is_available())



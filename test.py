import torch

from utilities.utils import mse

test_input = torch.randn([3, 50, 100])
test_target = torch.randn([3, 50, 100])

#print(test_input)
#print(test_target)
#a = mse(test_input, test_target)
#print("a: ", a.item())

loss = (test_target - test_input) ** 2
print("loss: ", loss)
n = len(test_target)
sum = 0
for i in range(n):
    sum = sum + loss[i]

print("sum: ",  torch.sum(sum)/len(test_target))
print("sum2: ", (torch.sum((test_target - test_input) ** 2)) / len(test_target))
print("sum correct: ", torch.nn.functional.mse_loss(test_input, test_target))


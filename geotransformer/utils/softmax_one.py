import torch


# Define the softmax_one function with added one in the denominator , which helps to reduce
# the negative impact impact of tiny values in the softmax function and improves numerical stability
def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    # subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    # compute exponentials
    exp_x = torch.exp(x)
    # compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))

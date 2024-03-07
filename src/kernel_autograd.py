# This module is a template.
# Only supports 1 input in the forward & backward functions for now.

import torch
import NEW

class NEWKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = NEW.forward(x)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        result, = ctx.saved_tensors
        return NEW.backward(result, grad)

def NEW_wrapper(input):
    return NEWKernel.apply(input)

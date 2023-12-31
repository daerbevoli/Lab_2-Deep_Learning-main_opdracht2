import torch

from options.options import Options
from utilities.utils import plot_tensor, mse, init_pytorch, not_implemented

# TENSORS

def create_image(options: Options) -> torch.Tensor:
    """
    TODO: implement this method
    use options to put the tensor to the correct device.
    """
    rgb = [(0.5021, 0.1138, 0.9047), (0.2843, 0.0684, 0.6829), (0.1935, 0.5483, 0.3117),
           (0.8017, 0.8733, 0.6258), (0.5914, 0.6004, 0.2893), (0.7038, 0.5983, 0.9914)]

    # Used to test knowledge of tensors

    tt = torch.FloatTensor(rgb).to(options.device)
    return tt



def lin_layer_forward(weights: torch.Tensor, random_image: torch.Tensor) -> torch.Tensor:
    """TODO: implement this method"""
    # input function inj = matrix multiplication of the weights and inputs , Deep Learning slide 9
    return torch.sum(weights * random_image)


def tensor_network():
    target = torch.FloatTensor([0.5]).to(options.device)
    print(f"The target is: {target.item():.2f}")
    plot_tensor(target, "Target")

    input_tensor = torch.FloatTensor([0.4, 0.8, 0.5, 0.3]).to(options.device)
    weights = torch.FloatTensor([0.1, -0.5, 0.9, -1]).to(options.device)
    """START TODO:  ensure that the tensor 'weights' saves the computational graph and the gradients after backprop"""
    # when setting the requires_grad bool to True
    # the computational graph containing the order of the functions will be saved as well , Google Collab Gradients
    weights.requires_grad = True
    """END TODO"""

    # remember the activation a of a unit is calculated as follows:
    #
    # a = W * x, with W the weights and x the inputs of that unit
    output = lin_layer_forward(weights, input_tensor).to(options.device)
    print(f"Output value: {output.item(): .2f}")
    plot_tensor(output.detach(), "Initial Output")

    # We want a measure of how close we are according to our target
    loss = mse(output, target)
    print(f"The initial loss is: {loss.item():.2f}\n")

    # Lets update the weights now using our loss..
    print(f"The current weights are: {weights}")

    """START TODO: the loss needs to be backpropagated"""
    # computes the gradient of current tensor w.r.t. graph leaves
    # https://pytorch.org/docs/master/generated/torch.Tensor.backward.html#torch.Tensor.backward
    loss.backward()
    """END TODO"""

    print(f"The gradients are: {weights.grad}")
    """START TODO: implement the update step with a learning rate of 0.5"""
    # use tensor operations, recall the following formula we've seen during class: x <- x - alpha * x'
    # Tensor.grad computes gradients for self , Learning from examples slide 62
    lr = 0.0001 # learning rate
    weights = weights - lr*weights.grad

    """END TODO"""
    print(f"The new weights are: {weights}\n")

    # What happens if we forward through our layer again?
    output = lin_layer_forward(weights, input_tensor)
    print(f"Output value: {output.item(): .2f}")
    plot_tensor(output.detach(), "Improved Output")

    # What happens here is that we get a target tensor to begin with
    # We make the initial approximation using the initial inputs and weights
    # We then calculate the loss to our target and then we start backpropagating
    # We adjust our weights using a 0.5 learning rate and the gradients of the weights

if __name__ == "__main__":
    options = Options()
    init_pytorch(options)
    tensor_network()


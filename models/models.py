import torch
import torch.nn as nn

from options.classification_options import ClassificationOptions


class Print(nn.Module):
    """"
    This model is for debugging purposes (place it in nn.Sequential to see tensor dimensions).
    """

    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        return x


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        """START TODO: replace None with a Linear layer"""
        # A linear regression model is trying to find a relation between the input and the output
        # Input: size of the given house -> 1 = in_features
        # Output: price of the house based on the size -> 1 = out_features
        # Single input feature to singe output feature
        self.linear_layer = nn.Linear(1, 1)
        """END TODO"""

    def forward(self, x: torch.Tensor):
        """START TODO: forward the tensor x through the linear layer and return the outcome (replace None)"""
        out = self.linear_layer(x)
        """END TODO"""
        return out

class Classifier(nn.Module):
    def __init__(self, options: ClassificationOptions):
        super().__init__()
        """ START TODO: fill in all three layers. 
            Remember that each layer should contain 2 parts, a linear layer and a nonlinear activation function.
            Use options.hidden_sizes to store all hidden sizes, (for simplicity, you might want to 
            include the input and output as well).
            The input size should be 28*28=784, because we are dealing with 28x28 MNIST images
            The output size should be 10, because they correspond with the digits 0 to 9 that are the "classes"
        """
        input_size = options.hidden_sizes[0]
        hidden_size1 = options.hidden_sizes[1]
        hidden_size2 = options.hidden_sizes[2]
        # hidden_size3 = options.hidden_sizes[3]
        # hidden_size4 = options.hidden_sizes[4]
        output_size = options.hidden_sizes[3]

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU()
        )

        """self.layer3 = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(hidden_size3, hidden_size4),
            nn.ReLU()
        )"""

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size2, output_size),
            nn.Softmax(dim=1)      # dim=1 for batch processing (the function is applied to each item in the batch
                                   # separately, rather than across all batches)
        )
        """END TODO"""

    def forward(self, x: torch.Tensor):
        """START TODO: forward tensor x through all layers."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # the output of the previous layer, becomes the input of the next
        # x = self.layer4(x)
        # x = self.layer5(x)
        """END TODO"""
        return x


class ClassifierVariableLayers(nn.Module):
    def __init__(self, options: ClassificationOptions):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(len(options.hidden_sizes) - 1):
            self.layers.add_module(
                f"lin_layer_{i + 1}",
                nn.Linear(options.hidden_sizes[i], options.hidden_sizes[i + 1])
            )
            if i < len(options.hidden_sizes) - 2:
                self.layers.add_module(
                    f"relu_layer_{i + 1}",
                    nn.ReLU()
                )
            else:
                self.layers.add_module(
                    f"softmax_layer",
                    nn.Softmax(dim=1)
                )
        print(self)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        return x

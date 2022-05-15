import torch
import torch.nn as nn
import itertools as it

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: list,
        pool_every: int,
        hidden_dims: list,
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},

    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions.
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        N = len(self.channels) # N is the total number of convolutional layers,  A list of of length N 
        channel_idx = 0
        for pool_idx in range(int(N / self.pool_every)): # int(), floor rounded
            for convolution_idx in range(self.pool_every):
                out_channels = self.channels[channel_idx]
                layers.append(nn.Conv2d(in_channels, out_channels, **self.conv_params, bias=True))
                layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))
                in_channels = out_channels # for next layer
                channel_idx += 1
            layers.append(POOLINGS[self.pooling_type](**self.pooling_params))
        
        for convolution_idx in range(N % self.pool_every):
            out_channels = self.channels[channel_idx]
            layers.append(nn.Conv2d(in_channels, out_channels, **self.conv_params, bias=True))
            layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))
            in_channels = out_channels  # for next layer
            channel_idx += 1
            
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        layers = []
        # TODO: Create the classifier part of the model:
        #  (FC -> ACT)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        #find input parameter using the feature extractor layers
        _, in_h, in_w, = tuple(self.in_size)
        for layer in self.feature_extractor:
            if isinstance(layer, torch.nn.Conv2d): # if it convolution layer
                K = layer.kernel_size[0]
                S = layer.stride[0]
                P = layer.padding[0]
            elif isinstance(layer, torch.nn.MaxPool2d) or isinstance(layer, torch.nn.AvgPool2d): #max pooling or average pooling
                K = layer.kernel_size
                S = layer.stride
                P = layer.padding
            else:
                continue
            in_h = ((in_h - K +2*P)//S) + 1
            in_w = ((in_w - K +2*P)//S) + 1
        c = self.channels[-1]
        input_dimension = c * in_h * in_w
        for hidden_dimension in self.hidden_dims:
            layers.append(nn.Linear(input_dimension, hidden_dimension))
            layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))
            input_dimension = hidden_dimension
        layers.append(nn.Linear(input_dimension, self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: list,
        kernel_sizes: list,
        batchnorm=False,
        dropout=0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        # --- main sequential: ---
        conv_num = len(channels)          # The length determines the number of convolutions.
        main_list = []                    # add layers here and insrert it as a sequential to main_path at the end
        input_channels = in_channels      # input for dirst conv2d - need to be updated between layers
        last_conv_idx = len(channels) - 1 # finish without dropout batchnorm, don't add them after this index
        
        for i, conv_i_out_size in enumerate(channels):
            padding = kernel_sizes[i]//2 #to keep dimension - adding padding of half of the kernel size (centers in edges in the end and beginning
            main_list.append(nn.Conv2d(input_channels, conv_i_out_size, kernel_sizes[i], padding = padding, bias=True))
            if i < last_conv_idx:
                if dropout>0:
                    main_list.append(nn.Dropout2d(dropout))
                if batchnorm:
                    main_list.append(nn.BatchNorm2d(conv_i_out_size))
                main_list.append(ACTIVATIONS[activation_type](**activation_params))

                input_channels = conv_i_out_size # update input to next layer
        self.main_path = nn.Sequential(*main_list)
            
        
        # --- shortcut sequential: ---
        shortcut_list = []
        if channels[last_conv_idx] == in_channels: # same size, do nothing
            # add something? identity()? do nothing?
            self.shortcut_path = nn.Sequential() 
        else: # need to add convolution layer, without bias!
            self.shortcut_path = nn.Sequential(nn.Conv2d(in_channels, channels[last_conv_idx], kernel_size=1, bias=False))
        
        
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        # ====== YOUR CODE: ======

        N = len(self.channels)
        rest = N % self.pool_every
        N_rounded = N - rest            # size without rest (-N%p)
        kernels = [3] * self.pool_every # for ResidualBloack
        input_channels = in_channels    # for first ResidualBloack, will be update to self.channels[index(0 to floor(N/P))]
        for i in range(0, N_rounded, self.pool_every):
            output_channels = self.channels[i:i + self.pool_every]
            layers.append(ResidualBlock(input_channels, output_channels, kernels, batchnorm = self.batchnorm, dropout = self.dropout,
                                       activation_type = self.activation_type, activation_params = self.activation_params))
            layers.append(POOLINGS[self.pooling_type](**self.pooling_params))
            input_channels = self.channels[i]

        if rest: # N%p correction
            input_channels = self.channels[N_rounded - 1]
            output_channels = self.channels[N_rounded:N_rounded + rest]
            layers.append(ResidualBlock(input_channels, output_channels, [3] * rest, batchnorm = self.batchnorm, dropout = self.dropout,
                                       activation_type = self.activation_type, activation_params = self.activation_params))
            
         # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every, hidden_dims,**kwargs):
        super().__init__(in_size, out_classes, channels, pool_every, hidden_dims,**kwargs)

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======
    
    
#     #################### our model ####################   ~75-80%
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)
        layers = []
        conv_params = dict(kernel_size=3, stride=1, padding=1)
        N = len(self.channels)
        rest = N % self.pool_every
        N_rounded = N - rest            # size without rest (-N%p)
        kernels = [3] * self.pool_every # for ResidualBloack
        input_channels = in_channels    # for first ResidualBloack, will be update to self.channels[index(0 to floor(N/P))]
        maxpool = nn.MaxPool2d(kernel_size = 2, stride=2)
        for i in range(0, N_rounded, self.pool_every):
            output_channels = self.channels[i:i + self.pool_every]
            layers.append(nn.Conv2d(input_channels,input_channels,kernel_size=2))
            layers.append(ResidualBlock(input_channels, output_channels, kernels, batchnorm = True, dropout = 0,
                                       activation_type = 'relu', activation_params = self.activation_params))
            layers.append(maxpool)
            input_channels = self.channels[i]
        layers.append(nn.Dropout(p=0.2))
        if rest: # N%p correction
            input_channels = self.channels[N_rounded - 1]
            output_channels = self.channels[N_rounded:N_rounded + rest]
            layers.append(ResidualBlock(input_channels, output_channels, [3] * rest, batchnorm = True, dropout = 0.1,
                                       activation_type = 'relu', activation_params = self.activation_params))
            
        # ========================
        seq = nn.Sequential(*layers)
        return seq
#     # ========================
    
    
    
    
    
    
    
    
    
    
    
    
    
# #     #################### old model ####################   ~65%
#     def _make_feature_extractor(self):
#         in_channels, in_h, in_w, = tuple(self.in_size)
#         layers = []
#         conv_params = dict(kernel_size=3, stride=1, padding=1)
#         N = len(self.channels)
#         rest = N % self.pool_every
#         N_rounded = N - rest            # size without rest (-N%p)
#         kernels = [3] * self.pool_every # for ResidualBloack
#         input_channels = in_channels    # for first ResidualBloack, will be update to self.channels[index(0 to floor(N/P))]
#         maxpool = nn.MaxPool2d(kernel_size = 2, stride=2)
#         for i in range(0, N_rounded, self.pool_every):
#             output_channels = self.channels[i:i + self.pool_every]
#             layers.append(ResidualBlock(input_channels, output_channels, kernels, batchnorm = True, dropout = 0.1,
#                                        activation_type = 'relu', activation_params = self.activation_params))
#             layers.append(maxpool)
#             input_channels = self.channels[i]

#         if rest: # N%p correction
#             input_channels = self.channels[N_rounded - 1]
#             output_channels = self.channels[N_rounded:N_rounded + rest]
#             layers.append(ResidualBlock(input_channels, output_channels, [3] * rest, batchnorm = True, dropout = 0.1,
#                                        activation_type = 'relu', activation_params = self.activation_params))
            
#         # ========================
#         seq = nn.Sequential(*layers)
#         return seq
#     # ========================


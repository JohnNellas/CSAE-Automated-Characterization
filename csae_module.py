import torch
import numpy as np 
import PIL 
import PIL.Image 
import os


def get_activation_function(activation_function_name: str):

    assert isinstance(activation_function_name, str)
    activation = None
    if activation_function_name == "relu":
        activation = torch.nn.ReLU()
    elif activation_function_name == "gelu":
        activation = torch.nn.GELU()
    elif activation_function_name == "hs":
        activation = torch.nn.Hardswish()

    return activation

class LoadDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: The path to the directory that contains the dataset
            csv_file: The path to the csv file that contained the dataset's metadata
            transform: transforms that will be applied to the images
        """
        super(LoadDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = np.array(os.listdir(root_dir), dtype="object")
        
        

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
            
        image_path = os.path.join(
            self.root_dir, self.filenames[idx])

        img = PIL.Image.open(image_path)
        filename =  self.filenames[idx]
        
        if self.transform is not None:
            # following the way that torchvision.datasets.MNIST handles the alternative case
            # in order to convert to tensor, let the user choose the corresponding transform.
            img = self.transform(img) 

        return (img, filename)


class Encoder(torch.nn.Module):
    def __init__(self, input_shape, latent_space_dims, activation_name="gelu", num_conv_blocks=3):
        super(Encoder, self).__init__()

        self.num_conv_blocks = num_conv_blocks

        self.activation_name = activation_name

        # save the input shape
        self.input_shape = input_shape

        # Convolutional Module
        self.enc_conv = torch.nn.Sequential(
            # conv
            self.create_convolution_block(input_filter=self.input_shape[-1],
                                          containing_filters=64,
                                          kernel_size=(5, 5),
                                          stride=(1, 1),
                                          padding=2,
                                          num_conv_blocks=self.num_conv_blocks
                                          ),
            # downscale 64x64
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # conv
            self.create_convolution_block(input_filter=64,
                                          containing_filters=64,
                                          kernel_size=(5, 5),
                                          stride=(1, 1),
                                          padding=2,
                                          num_conv_blocks=self.num_conv_blocks
                                          ),

            # downscale 32x32
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # conv
            self.create_convolution_block(input_filter=64,
                                          containing_filters=64,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          num_conv_blocks=self.num_conv_blocks
                                          ),

            # downscale 16x16
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # conv
            self.create_convolution_block(input_filter=64,
                                          containing_filters=128,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          num_conv_blocks=self.num_conv_blocks
                                          ),

            # downscale 8x8
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # conv
            self.create_convolution_block(input_filter=128,
                                          containing_filters=128,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          num_conv_blocks=self.num_conv_blocks
                                          )
        )

        flatten_out_shape = (
            self.input_shape[0]//(2**4)) * (self.input_shape[1]//(2**4)) * 128

        # NN Module
        self.enc_nn = torch.nn.Sequential(
            torch.nn.Linear(128, latent_space_dims)
        )

    def create_convolution_block(self, input_filter, containing_filters, kernel_size, stride, padding=1, num_conv_blocks=3):\
        
        layers = list()
        for conv_block_id in range(num_conv_blocks):
            if not layers:
                layer = torch.nn.Conv2d(in_channels=input_filter, out_channels=containing_filters,
                            kernel_size=kernel_size,
                            stride=stride, padding=padding)
            else:
                layer = torch.nn.Conv2d(in_channels=containing_filters, out_channels=containing_filters,
                            kernel_size=kernel_size,
                            stride=stride, padding=padding)
                
            activation = get_activation_function(self.activation_name)
            
            layers.extend([layer,activation])
        
        return torch.nn.Sequential(*layers)
                

    def forward(self, x):
        x = self.enc_conv(x)
        x = torch.mean(x, dim=(2, 3))
        latent_repr = self.enc_nn(x)

        return latent_repr


class Classifier(torch.nn.Module):
    def __init__(self, latent_space_dims, number_of_classes, activation_name="gelu"):
        super(Classifier, self).__init__()

        self.activation_name = activation_name
        
        self.cl_nn = torch.nn.Sequential(
            torch.nn.Linear(latent_space_dims, 512),
            get_activation_function(self.activation_name),
            torch.nn.Linear(512, number_of_classes)
        )

    def forward(self, x):
        return self.cl_nn(x)


class Decoder(torch.nn.Module):
    def __init__(self, input_shape, enc_out, activation_name="gelu", num_conv_blocks=3):
        super(Decoder, self).__init__()

        self.activation_name = activation_name
        
        self.num_conv_blocks = num_conv_blocks
        
        # save the input shape
        self.input_shape = input_shape

        # calculation of flattening 1d shape
        flatten_out_shape = (
            self.input_shape[0]//(2**4)) * (self.input_shape[1]//(2**4)) * 128

        # NN Module
        self.dec_nn = torch.nn.Sequential(
            torch.nn.Linear(enc_out, 128),
            get_activation_function(self.activation_name),
            torch.nn.Linear(128, flatten_out_shape),
            get_activation_function(self.activation_name)
        )

        # Reshape Layer
        self.unflatten = torch.nn.Unflatten(dim=1,
                                            unflattened_size=(128, self.input_shape[0]//(2**4), self.input_shape[1]//(2**4)))

        # NN module
        self.dec_conv = torch.nn.Sequential(

            # conv
            self.create_convolution_block(input_filter=128,
                                          containing_filters=128,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          num_conv_blocks=self.num_conv_blocks
                                          ),
            # upsample 16x16
            torch.nn.ConvTranspose2d(128, 128, (3, 3),
                                     stride=(2, 2),
                                     padding=1, output_padding=1),
            torch.nn.BatchNorm2d(128),
            get_activation_function(self.activation_name),

            # conv
            self.create_convolution_block(input_filter=128,
                                          containing_filters=128,
                                          kernel_size=(3, 3),
                                          stride=(1, 1),
                                          num_conv_blocks=self.num_conv_blocks
                                          ),
            # upsample 32x32
            torch.nn.ConvTranspose2d(128, 64, (3, 3),
                                     stride=(2, 2),
                                     padding=1, output_padding=1),
            torch.nn.BatchNorm2d(64),
            get_activation_function(self.activation_name),

            # conv
            self.create_convolution_block(input_filter=64,
                                          containing_filters=64,
                                          kernel_size=(5, 5),
                                          stride=(1, 1),
                                          padding=2,
                                          num_conv_blocks=self.num_conv_blocks
                                          ),
            # upsample 64x64
            torch.nn.ConvTranspose2d(64, 64, (5, 5),
                                     stride=(2, 2),
                                     padding=2, output_padding=1),
            torch.nn.BatchNorm2d(64),
            get_activation_function(self.activation_name),

            self.create_convolution_block(input_filter=64,
                                          containing_filters=64,
                                          kernel_size=(5, 5),
                                          stride=(1, 1),
                                          padding=2,
                                          num_conv_blocks=self.num_conv_blocks
                                          ),

            # upsample 128x128
            torch.nn.ConvTranspose2d(64, 64, (5, 5),
                                     stride=(2, 2),
                                     padding=2, output_padding=1),
            torch.nn.BatchNorm2d(64),
            get_activation_function(self.activation_name),

            self.create_convolution_block(input_filter=64,
                                          containing_filters=64,
                                          kernel_size=(5, 5),
                                          stride=(1, 1),
                                          padding=2,
                                          num_conv_blocks=self.num_conv_blocks
                                          ),

            # output layer
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=self.input_shape[-1],
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0),
            torch.nn.Sigmoid()
        )

    def create_convolution_block(self, input_filter, containing_filters, kernel_size, stride, padding=1, num_conv_blocks=3):\
        
        layers = list()
        for conv_block_id in range(num_conv_blocks):
            if not layers:
                layer = torch.nn.Conv2d(in_channels=input_filter, out_channels=containing_filters,
                            kernel_size=kernel_size,
                            stride=stride, padding=padding)
            else:
                layer = torch.nn.Conv2d(in_channels=containing_filters, out_channels=containing_filters,
                            kernel_size=kernel_size,
                            stride=stride, padding=padding)
            
            batch_norm = torch.nn.BatchNorm2d(containing_filters)
            activation = get_activation_function(self.activation_name)
            
            layers.extend([layer, batch_norm, activation])
        
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.dec_nn(x)
        x = self.unflatten(x)
        reconstructions = self.dec_conv(x)
        return reconstructions


class CSAE(torch.nn.Module):
    def __init__(self, encoder, decoder, classifier):
        super(CSAE, self).__init__()

        # the encoder network
        self.encoder = encoder

        # the decoder network
        self.decoder = decoder

        # the classifier network
        self.classifier = classifier

    def forward(self, x):
        # get the latent representations
        latent_repr = self.encoder(x)

        # do a prediction
        logits = self.classifier(latent_repr)

        # reconstruct the image from the latent representation
        reconstructions = self.decoder(latent_repr)

        return latent_repr, logits, reconstructions

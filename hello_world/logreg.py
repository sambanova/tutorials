r"""
This logreg example has minimal code that supports defining a model, compiling it, and 
doing a training/test run. 
In the accompanying tutorial at https://docs.sambanova.ai/developer/latest/getting-started.html, 
we show the commands and options. 
"""

import argparse
import sys
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision

import sambaflow.samba.utils as utils
from sambaflow import samba
from sambaflow.samba.utils.argparser import (parse_app_args,
                                             parse_yaml_to_args)
from sambaflow.samba.utils.dataset.mnist import dataset_transform
from sambaflow.samba.utils.pef_utils import get_pefmeta


class LogReg(nn.Module):
    """ Define the model architecture

    Define the model architecture i.e. the layers in the model and the
    number of features in each layer.

    Args:
        nlin_layer (ivar): Linear layer
        criterion (ivar): Cross Entropy loss layer
    """    
   
    def __init__(self, num_features: int, num_classes: int, bias: bool):
        """ Initialization function for this class 

        Args:
            num_features (int):  Number of input features for the model
            num_classes (int): Number of output labels the model classifies inputs
            bias (bool): _description_??
        """
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Linear layer for predicting target class of inputs
        self.lin_layer = nn.Linear(in_features=num_features, out_features=num_classes, bias=bias)

        # Cross Entropy layer for loss computation
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass of the model for the given inputs. 
        
        The forward pass predicts the class labels for the inputs
        and computes the loss between the correct and predicted class labels.

        Args:
            inputs (torch.Tensor):  Input samples in the dataset
            targets (torch.Tensor): correct labels for the inputs

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:The loss and predicted classes of the inputs
        """
        
        out = self.lin_layer(inputs)
        loss = self.criterion(out, targets)

        return loss, out


def add_args(parser: argparse.ArgumentParser) -> None:
    """ Add model-specific arguments. 
    
    By default, the compiler and the Samba framework support a set of arguments to compile() and run().
    The arguement parser supports adding application-specific arguments. 

    Args:
        parser (argparse.ArgumentParser): SambaNova argument parser. 
    """

    parser.add_argument('--lr', type=float, default=0.0015, help="Learning rate for training")
    parser.add_argument('--momentum', type=float, default=0.0, help="Momentum value for training")
    parser.add_argument('--weight-decay', type=float, default=3e-4, help="Weight decay for training")
    parser.add_argument('--num-epochs', '-e', type=int, default=1)
    parser.add_argument('--num-steps', type=int, default=-1)
    parser.add_argument('--num-features', type=int, default=784)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--yaml-config', default=None, type=str, help='YAML file used with launch_app.py')
    parser.add_argument('--data-dir',
                        '--data-folder',
                        type=str,
                        default='mnist_data',
                        help="The folder to download the MNIST dataset to.")
    parser.add_argument('--bias', action='store_true', help='Linear layer will learn an additive bias')
    # end args


def prepare_dataloader(args: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    """ Preparation of dataloader. 
    Prep work for training the logreg model with the `MNIST dataset <http://yann.lecun.com/exdb/mnist/>`__:
    We'll split the dataset into train and test sets and return the corresponding data loaders.
    
   
    # :param args: argument specifying the location of the dataset
    # :type args: argparse.Namespace
    # RK did not prompt to add arg. What's wrong with the function definition? 


    Returns:
       Tuple[torch.utils.data.DataLoader]: Train and test data loaders
    """

    # Get the train & test data (images and labels) from the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root=f'{args.data_dir}',
                                               train=True,
                                               transform=dataset_transform(vars(args)),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root=f'{args.data_dir}',
                                              train=False,
                                              transform=dataset_transform(vars(args)))

    # Get the train & test data loaders (input pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader


def train(args: argparse.Namespace, model: nn.Module, output_tensors: Tuple[samba.SambaTensor]) -> None:
    """ 
    Train the model. At the end of a training loop, the model will be able
    to correctly predict the class labels for any input, within a certain
    accuracy.

    Args:
        args (argparse.Namespace): Hyperparameter values and accuracy test behavior controls
        model (nn.Module): Model to be trained
        output_tensors (Tuple[samba.SambaTensor]): _description_

    Returns:
    # RK??       _type_: _description_
    """

    # Get data loaders for training and test data
    train_loader, test_loader = prepare_dataloader(args)

    # Total training steps (iterations) per epoch
    total_step = len(train_loader)

    hyperparam_dict = {"lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay}

    # Train and test for specified number of epochs
    for epoch in range(args.num_epochs):
        avg_loss = 0

        # Train the model for all samples in the train data loader
        for i, (images, labels) in enumerate(train_loader):
            global_step = epoch * total_step + i
            if args.num_steps > 0 and global_step >= args.num_steps:
                print('Maximum num of steps reached. ')
                return None

            sn_images = samba.from_torch_tensor(images, name='image', batch_dim=0)
            sn_labels = samba.from_torch_tensor(labels, name='label', batch_dim=0)
  
            loss, outputs = samba.session.run(input_tensors=[sn_images, sn_labels],
                                              output_tensors=output_tensors,
                                              hyperparam_dict=hyperparam_dict,
                                              data_parallel=args.data_parallel,
                                              reduce_on_rdu=args.reduce_on_rdu)

            # Sync the loss and outputs with host memory
            loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
            avg_loss += loss.mean()

            # Print loss per 10,000th sample in every epoch
            if (i + 1) % 10000 == 0 and args.local_rank <= 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_epochs, i + 1, total_step,
                                                                         avg_loss / (i + 1)))

        # Check the accuracy of the trained model for all samples in the test data loader
        # Sync the model parameters with host memory
        samba.session.to_cpu(model)
        test_acc = 0.0
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            for images, labels in test_loader:
                loss, outputs = model(images, labels)
                loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
                total_loss += loss.mean()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

            test_acc = 100.0 * correct / total

            if args.local_rank <= 0:
                print(f'Test Accuracy: {test_acc:.2f} Loss: {total_loss.item() / len(test_loader):.4f}')


        # if args.acc_test:
           # assert args.num_epochs == 1, "Accuracy test only supported for 1 epoch"
           # assert test_acc > 91.0 and test_acc < 92.0, "Test accuracy not within specified bounds."



def main(argv):
    """
    :param argv: Command line arguments (`compile`, `test`, `run`, `measure-performance` or `measure-sections`)
    """
    utils.set_seed(256)

    args_cli = parse_app_args(argv=argv, common_parser_fn=add_args)
    args_composed = parse_yaml_to_args(args_cli.yaml_config, args_cli) if args_cli.yaml_config else args_cli
    # _ = SambaConfig(args_composed, SNConfig).get_all_params()

    args = args_composed

    # when it is not distributed mode, local rank is -1.
    args.local_rank = dist.get_rank() if dist.is_initialized() else -1

    # Create random input and output for compilation
    ipt = samba.randn(args.batch_size, args.num_features, name='image', batch_dim=0).bfloat16().float()
    tgt = samba.randint(args.num_classes, (args.batch_size, ), name='label', batch_dim=0)
    inputs = (ipt, tgt)

    # RK>>This generates a warning, commenting out for now
    # ipt.host_memory = False
    # tgt.host_memory = False

    # Instantiate the model
    model = LogReg(args.num_features, args.num_classes, args.bias)

    # Sync model parameters with RDU memory
    samba.from_torch_model_(model)

    # Instantiate an optimizer if the model will be trained
    if args.inference:
        optimizer = None
    else:
        # We use the SGD optimizer to update the weights of the model
        optimizer = samba.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if args.command == "compile":
        #  Compile the model to generate a PEF (Plasticine Executable Format) binary
        samba.session.compile(model,
                              inputs,
                              optimizer,
                              name='logreg_torch',
                              app_dir=utils.get_file_dir(__file__),
                              config_dict=vars(args),
                              pef_metadata=get_pefmeta(args, model))
        
    else:
        assert args.command == "run"

        # Trace the compiled graph to initialize the model weights and input/output tensors
        # for execution on the RDU.
        # The PEF required for tracing is the binary generated during compilation
        traced_outputs = utils.trace_graph(model, inputs, optimizer, pef=args.pef)

        # Train the model on RDU. This is where the model will be trained
        # i.e. weights will be learned to fit the input dataset
        train(args, model, traced_outputs)



if __name__ == '__main__':
    main(sys.argv[1:])

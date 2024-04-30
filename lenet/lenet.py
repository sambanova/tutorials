import argparse
import os
from pathlib import Path
from typing import Tuple

import sambaflow.samba.utils as utils
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from mnist_utils import CustomMNIST, write_labels
from sambaflow import __version__ as sambaflow_version
from sambaflow import samba
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.pef_utils import get_pefmeta
from torch.utils.data.dataloader import DataLoader


class LeNet(nn.Module):
    """
    LeNet model for MNIST classification.

    Attributes:
        state: Dictionary to hold model's completed_steps and completed_epochs.
        conv1, conv2: Convolutional layers.
        maxpool1, maxpool2: Max pooling layers.
        fc1, fc2, fc3: Fully connected layers.
        criterion: Loss function.
    """

    def __init__(self, num_classes: int) -> None:
        super(LeNet, self).__init__()
        self.state = {"completed_steps": 0, "completed_epochs": 0}

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=(1, 1),
                               bias=False)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=(1, 1),
                               dilation=(1, 1),
                               bias=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        """Defines the forward propagation step."""
        x = self.conv1(inputs).relu()
        x = self.maxpool1(x)
        x = self.conv2(x).relu()
        x = self.maxpool2(x)
        x = torch.reshape(x, [x.shape[0], -1])
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        out = self.fc3(x)
        loss = self.criterion(out, labels)
        return loss, out


def get_inputs(params) -> Tuple[samba.SambaTensor, samba.SambaTensor]:
    """
    Creates input images and labels to set the model's shape for compilation.

    Args:
        params: A dictionary containing various parameters including 'batch_size' and 'num_classes'.

    Returns:
        A tuple of input images and labels.
    """
    images = samba.randn(params['batch_size'],
                         1,
                         28,
                         28,
                         name='image',
                         batch_dim=0)
    labels = samba.randint(params['num_classes'], (params['batch_size'],),
                           name='label',
                           batch_dim=0)
    return (images, labels)


def get_dataset(dataset_name: str, params):
    """Retrieves the specified dataset after applying necessary transformations.

    Args:
        dataset_name: The name of the dataset to retrieve.
        params: A dictionary containing various parameters including 'data_dir' and 'inference'.

    Returns:
        The requested dataset as a CustomMNIST object.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # norm by mean and var
        transforms.Normalize((0.1307,), (0.3081,)),
        # Reshape image to 1x28x28
        lambda x: x.reshape((1, 28, 28)),
    ])

    data_dir = Path(params['data_dir'])
    img_file = data_dir / (dataset_name + "-images-idx3-ubyte")
    if params['inference']:  # if running for inference there's no labels file
        lbl_file = None
    else:
        lbl_file = data_dir / (dataset_name + "-labels-idx1-ubyte")
    dataset = CustomMNIST(img_file, lbl_file, transform=transform)

    return dataset


def load_checkpoint(model, optimizer, init_ckpt_path: str):
    """
    Loads a checkpoint from a file and initialize the model and optimizer.

    Args:
        model (object): The model to be loaded.
        optimizer (object): The optimizer to be loaded.
        init_ckpt_path (str): The path to the checkpoint file.

    Returns:
        None
    """
    print(f"Loading checkpoint from file {init_ckpt_path}")
    ckpt = torch.load(init_ckpt_path)
    if model:
        print("Loading model...")
        model.load_state_dict(ckpt['model'])
        model.state['completed_steps'] = ckpt['completed_steps']
        model.state['completed_epochs'] = ckpt['completed_epochs']
    if optimizer:
        print("Loading optimizer...")
        optimizer.load_state_dict(ckpt['optimizer'])


def save_checkpoint(model, optimizer, completed_steps, completed_epochs,
                    ckpt_dir):
    """
    Saves the model checkpoint with the given parameters.

    Args:
        model (nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer to be saved.
        completed_steps (int): The number of completed steps.
        completed_epochs (int): The number of completed epochs.
        ckpt_dir (str): The directory in which to save the checkpoint.

    Returns:
        str: The path of the saved checkpoint.
    """
    ckpt_dir_path = Path(ckpt_dir)
    ckpt_dir_path.mkdir(parents=True, exist_ok=True)

    state_dict = {
        'completed_steps': completed_steps,
        'completed_epochs': completed_epochs,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    ckpt_path = ckpt_dir_path / (str(completed_steps) + ".pt")
    torch.save(state_dict, ckpt_path)
    return ckpt_path


def log_step(epoch, num_epochs, current_step, total_steps, loss):
    """
    Prints the current step information during training.

    Args:
        epoch (int): The current epoch number.
        num_epochs (int): The total number of epochs.
        current_step (int): The current step number.
        total_steps (int): The total number of steps.
        loss (float): The loss value for the current step.

    Returns:
        None
    """
    print(
        f"Epoch [{epoch}/{num_epochs}], Step [{current_step}/{total_steps}], Loss: {loss:.4f}"
    )


def prepare(model: nn.Module, optimizer, params):
    """
    Prepares the model by loading a checkpoint and tracing the graph.

    Args:
        model (nn.Module): The model to prepare.
        optimizer: The optimizer for the model.
        params: A dictionary of parameters.

    Returns:
        None
    """

    # We need to load the checkpoint first and then trace the graph to sync the weights from CPU to RDU
    if params['init_ckpt_path']:
        load_checkpoint(model, optimizer, params['init_ckpt_path'])
    else:
        print('[WARNING] No valid initial checkpoint has been provided')

    inputs = get_inputs(params)
    utils.trace_graph(model,
                      inputs,
                      optimizer,
                      pef=params['pef'],
                      mapping=params['mapping'])


def train(model: LeNet, optimizer, params) -> None:
    """
    Trains the given model using the specified optimizer and parameters.

    Args:
        model (LeNet): The model to be trained.
        optimizer: The optimizer to be used during training.
        params: A dictionary containing the parameters for training.

    Returns:
        None
    """
    if params['dataset_name'] is None:
        dataset_name = "train"
    else:
        dataset_name = params['dataset_name']
    data_dir = Path(params['data_dir'])
    print(f"Using dataset: {data_dir / dataset_name}")
    train_dataset = get_dataset(dataset_name, params)
    train_loader = DataLoader(train_dataset,
                              batch_size=params['batch_size'],
                              drop_last=True,
                              shuffle=True)

    # Train the model
    current_step = model.state['completed_steps']
    current_epoch = model.state['completed_epochs']
    total_steps = len(train_loader) * params['num_epochs']
    if current_epoch == params['num_epochs']:
        print(
            f"Epochs trained: {current_epoch} is equal to epochs requested: {params['num_epochs']}. Exiting..."
        )
        return
    print("=" * 30)
    print(f"Initial epoch: {current_epoch:3n}, initial step: {current_step:6n}")
    print(
        f"Target epoch:  {params['num_epochs']:3n}, target step:  {total_steps:6n}"
    )
    hyperparam_dict = {
        "lr": params['lr'],
        "momentum": params['momentum'],
        "weight_decay": params['weight_decay']
    }
    for epoch in range(current_epoch + 1, params['num_epochs'] + 1):
        avg_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            sn_images = samba.from_torch_tensor(images,
                                                name='image',
                                                batch_dim=0)
            sn_labels = samba.from_torch_tensor(labels,
                                                name='label',
                                                batch_dim=0)

            loss, outputs = samba.session.run(
                input_tensors=[sn_images, sn_labels],
                output_tensors=model.output_tensors,
                hyperparam_dict=hyperparam_dict,
                data_parallel=params['data_parallel'],
                reduce_on_rdu=params['reduce_on_rdu'])
            loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
            avg_loss += loss.mean()
            current_step += 1

            if (i + 1) % 100 == 0:
                log_step(epoch, params['num_epochs'], current_step, total_steps,
                         avg_loss / (i + 1))

    current_epoch = epoch

    samba.session.to_cpu(model)
    save_checkpoint(model, optimizer, current_step, current_epoch,
                    params['ckpt_dir'])


def test(model, dataset_name, params):
    """
    Calculates the test accuracy and loss for the given model and dataset.

    Parameters:
        model (object): The model to be tested.
        dataset_name (str): The name of the dataset to be used.
        params (dict): A dictionary of parameters.

    Returns:
        None
    """
    if dataset_name is None:
        dataset_name = "t10k"
    data_dir = Path(params['data_dir'])
    print(f"Using dataset: {data_dir / dataset_name}")
    test_dataset = get_dataset(dataset_name, params)
    test_loader = DataLoader(test_dataset,
                             drop_last=True,
                             batch_size=params['batch_size'])

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
        print('Test Accuracy: {:.2f}'.format(test_acc),
              ' Loss: {:.4f}'.format(total_loss.item() / (len(test_loader))))


def batch_predict(model, dataset_name: str, params):
    """
    Generates the predictions for a given model on a dataset.

    Args:
        model (object): The trained model to use for prediction.
        dataset_name (str): The name of the dataset to use for prediction.
        params (dict): Additional parameters for the prediction.

    Returns:
        None
    """
    if dataset_name is None:
        dataset_name = "inference"
    data_dir = Path(params['data_dir'])
    print(f"Using dataset: {data_dir / dataset_name}")
    dataset = get_dataset(dataset_name, params)

    loader = DataLoader(dataset,
                        batch_size=params.get('batch_size', 32),
                        drop_last=True,
                        shuffle=False)

    predicted_labels = []
    for _, (images, labels) in enumerate(loader):
        sn_images = samba.from_torch_tensor(images, name='image', batch_dim=0)
        sn_labels = samba.from_torch_tensor(labels, name='label', batch_dim=0)

        loss, predictions = samba.session.run(
            input_tensors=[sn_images, sn_labels],
            output_tensors=model.output_tensors,
            section_types=['fwd'])
        loss, predictions = samba.to_torch(loss), samba.to_torch(predictions)
        _, predicted_indices = torch.max(predictions, axis=1)  # type: ignore

        predicted_labels += predicted_indices.tolist()

    # write to the file in the same format labels are stored
    results_dir = Path(params['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    write_labels(predicted_labels,
                 str(results_dir / "prediction-labels-idx1-ubyte"))


def add_common_args(parser: argparse.ArgumentParser):
    """
    Adds common arguments to the given ArgumentParser object.

    Args:
        parser (argparse.ArgumentParser): The ArgumentParser object to add the arguments to.

    Returns:
        None
    """
    parser.add_argument('--num-classes',
                        type=int,
                        default=10,
                        help="Number of output classes (default=10)")
    parser.add_argument('--num-features',
                        type=int,
                        default=784,
                        help="Number of input features (default=784)")
    parser.add_argument('--lr',
                        type=float,
                        default=0.1,
                        help="Learning rate (default=0.1)")
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        default=32,
                        help="Batch size (default=32)")
    parser.add_argument('--momentum',
                        type=float,
                        default=0.0,
                        help="Momentum (default=0.0)")
    parser.add_argument('--weight-decay',
                        type=float,
                        default=0.01,
                        help="Weight decay (default=0.01)")
    parser.add_argument('--print-params',
                        action='store_true',
                        default=False,
                        help="Print the model parameters (default=False)")


def add_run_args(parser: argparse.ArgumentParser):
    """
    Add runtime arguments to the parser.

    Args:
        parser (argparse.ArgumentParser): The parser to which the arguments will be added.

    Returns:
        None
    """
    parser.add_argument('-e', '--num-epochs', type=int, default=1)
    parser.add_argument('--log-path', type=str, default='checkpoints')
    parser.add_argument('--test',
                        action="store_true",
                        help="Test the trained model")
    parser.add_argument('--init-ckpt-path',
                        type=str,
                        default='',
                        help='Path to load checkpoint')
    parser.add_argument('--ckpt-dir',
                        type=str,
                        default=os.getcwd(),
                        help='Path to save checkpoint')
    parser.add_argument('--data-dir',
                        type=str,
                        default='./data',
                        help="Directory containing datasets")
    parser.add_argument('--dataset-name',
                        type=str,
                        help="Dataset name: train, t10k, inference, etc.")
    parser.add_argument('--results-dir',
                        type=str,
                        default='./results',
                        help="Directory to store inference results")


def print_params(params):
    """
    Prints the parameters and their values when --print-params is passed.

    Args:
        params (dict): A dictionary containing the parameters and their values.

    Returns:
        None
    """
    for k in sorted(params.keys()):
        print(f"{k}: {params[k]}")


def main():
    args = parse_app_args(dev_mode=True,
                          common_parser_fn=add_common_args,
                          test_parser_fn=add_run_args,
                          run_parser_fn=add_run_args)
    utils.set_seed(42)
    params = vars(args)
    if args.print_params:
        print_params(params)

    model = LeNet(args.num_classes)
    samba.from_torch_model_(model)

    inputs = get_inputs(params)

    optimizer = samba.optim.SGD(model.parameters(),
                                lr=0.0) if not args.inference else None
    if args.command == "compile":
        pef_metadata=get_pefmeta(args, model)
        samba.session.compile(model,
                              inputs,
                              optimizer,
                              name='lenet',
                              app_dir=utils.get_file_dir(__file__),
                              squeeze_bs_dim=True,
                              config_dict=vars(args),
                              pef_metadata=pef_metadata)

    elif args.command == "test":
        print("Test is not implemented in this version.")
    elif args.command == "run":
        if args.inference:
            prepare(model, optimizer, params)
            batch_predict(model, params['dataset_name'], params)
        elif args.test:
            prepare(model, optimizer, params)
            test(model, params['dataset_name'], params)
        else:
            prepare(model, optimizer, params)
            train(model, optimizer, params)


if __name__ == '__main__':
    main()

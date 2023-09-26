import struct
from array import array
from typing import Callable, Optional, Tuple, List
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class CustomMNIST(Dataset):
    """Custom dataset that is compatible with MNIST format."""

    def __init__(self,
                 images_path: Path,
                 labels_path: Optional[Path],
                 transform: Optional[Callable] = None) -> None:
        """Initialize the dataset with images and labels path, and a transform.

        Args:
            images_path (str): Path to the images file.
            labels_path (str): Path to the labels file.
            transform (Optional[Callable]): Optional transform to be applied on an image.
        """
        self.images = self.read_images(images_path)
        if labels_path == None:  # in case of inference
            self.labels = [0] * len(self.images)
        else:
            self.labels = self.read_labels(labels_path)
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[np.array, int]:
        """Fetch an image-label pair by index.

        Args:
            idx (int): Index to the data.

        Returns:
            Tuple[np.array, int]: A tuple containing an image and its corresponding label.
        """
        label = self.labels[idx]
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self) -> int:
        """Get the size of the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.labels)

    @staticmethod
    def check_magic_number(magic: int, expected: int) -> None:
        """Check if the magic number matches the expected number.

        Args:
            magic (int): The actual magic number.
            expected (int): The expected magic number.
        """
        if magic != expected:
            raise ValueError(
                f"Magic number is wrong: expected {expected}, got {magic}")

    @staticmethod
    def read_labels(file_path: Path) -> List[int]:
        """Read labels from an MNIST-formatted file.

        Args:
            file_path (Path): Path to the file to read from.

        Returns:
            List[int]: List of labels.
        """
        file_data = file_path.read_bytes()
        magic, size = struct.unpack(">II", file_data[:8])
        CustomMNIST.check_magic_number(magic, 2049)
        return list(array("B", file_data[8:]))

    @staticmethod
    def read_images(file_path: Path) -> List[np.array]:
        """Read images from an MNIST-formatted file.

        Args:
            file_path (Path): Path to the file to read from.

        Returns:
            List[np.array]: List of images.
        """
        file_data = file_path.read_bytes()
        magic, size, rows, cols = struct.unpack(">IIII", file_data[:16])
        CustomMNIST.check_magic_number(magic, 2051)
        image_data = array("B", file_data[16:])

        return [
            np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(
                28, 28) for i in range(size)
        ]


def write_labels(labels: List[int], filename: str) -> None:
    """Write labels to an MNIST-formatted file.

    Args:
        labels (List[int]): List of labels.
        filename (str): Filename to write to.
    """
    with open(filename, 'wb') as f:
        f.write(struct.pack('>ii', 2049, len(labels)))
        f.writelines(struct.pack('B', label) for label in labels)

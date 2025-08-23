import os
from typing import Dict, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ISLVideoDataset(Dataset):
    """
    PyTorch Dataset for ISL video data stored in H5 format.
    
    Each H5 file contains:
    - hand_landmarks: (num_frames, 2, 21, 3) - 2 hands, 21 landmarks, 3D coords
    - world_landmarks: (num_frames, 2, 21, 3) - world coordinates
    - frame_metadata: (num_frames,) - metadata for each frame
    """

    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 use_world_landmarks: bool = False,
                 use_hand_landmarks: bool = True,
                 normalize: bool = True,
                 max_frames: Optional[int] = None):
        """
        Args:
            root_dir: Path to h5_output directory
            split: 'train', 'test', or 'validation'
            use_world_landmarks: Whether to include world landmarks
            use_hand_landmarks: Whether to include hand landmarks
            normalize: Whether to normalize landmark coordinates
            max_frames: Maximum number of frames to use (truncate with center crop if longer)
        """
        self.root_dir = root_dir
        self.split = split
        self.use_world_landmarks = use_world_landmarks
        self.use_hand_landmarks = use_hand_landmarks
        self.normalize = normalize
        self.max_frames = max_frames

        self.split_dir = os.path.join(root_dir, split)
        if not os.path.exists(self.split_dir):
            raise ValueError(f"Split directory {self.split_dir} does not exist")

        # Build file list and label mapping
        self.file_list = []
        self.labels = []
        self.parent_categories = []  # Track parent category for each file
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.parent_to_classes = {}  # Map parent category to list of classes
        self.class_to_parent = {}  # Map class to parent category

        self._build_file_list()

    def _build_file_list(self):
        """Build list of all H5 files and create label mappings with parent category hierarchy."""
        # Get parent categories (top-level directories in split)
        parent_categories = sorted([d for d in os.listdir(self.split_dir)
                                    if os.path.isdir(os.path.join(self.split_dir, d))])

        all_classes = []

        # Iterate through parent categories
        for parent_category in parent_categories:
            parent_dir = os.path.join(self.split_dir, parent_category)

            # Get class names within this parent category
            class_names = sorted([d for d in os.listdir(parent_dir)
                                  if os.path.isdir(os.path.join(parent_dir, d))])

            # Store parent-class relationship
            self.parent_to_classes[parent_category] = class_names

            # Map each class to its parent
            for class_name in class_names:
                self.class_to_parent[class_name] = parent_category
                all_classes.append(class_name)

        # Create class to index mapping (across all parent categories)
        for idx, class_name in enumerate(sorted(all_classes)):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name

        # Build file list with parent category hierarchy
        for parent_category in parent_categories:
            parent_dir = os.path.join(self.split_dir, parent_category)
            class_names = self.parent_to_classes[parent_category]

            for class_name in class_names:
                class_dir = os.path.join(parent_dir, class_name)
                h5_files = [f for f in os.listdir(class_dir) if f.endswith('.h5')]

                for h5_file in h5_files:
                    file_path = os.path.join(class_dir, h5_file)
                    self.file_list.append(file_path)
                    self.labels.append(self.class_to_idx[class_name])
                    self.parent_categories.append(parent_category)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Returns:
            data: torch.Tensor of shape (C, T, V, M) where:
                C = number of channels (3 for x,y,z coordinates)
                T = number of frames
                V = number of vertices/landmarks (21)
                M = number of hands (2)
            label: int, class index
            metadata: dict with additional information
        """
        file_path = self.file_list[idx]
        label = self.labels[idx]
        parent_category = self.parent_categories[idx]

        # Load H5 file
        with h5py.File(file_path, 'r') as f:
            # Load landmark data - prioritize hand_landmarks for now
            if self.use_hand_landmarks:
                data = f['hand_landmarks'][:]  # (T, M, V, C) = (num_frames, 2, 21, 3)
            elif self.use_world_landmarks:
                # data = f['world_landmarks'][:]  # (T, M, V, C) = (num_frames, 2, 21, 3)
                raise ValueError("World landmarks not supported yet.")
            else:
                raise ValueError("No features selected. Enable hand_landmarks or world_landmarks.")

            # Load metadata
            frame_metadata = f['frame_metadata'][:]

            # Get file metadata if available
            file_metadata = {}
            if 'file_metadata' in f:
                file_metadata = dict(f['file_metadata'].attrs)

        # Normalize if requested
        if self.normalize:
            data = self._normalize_landmarks(data)

        # Convert to torch tensor and reshape to (C, T, V, M)
        # From (T, M, V, C) to (C, T, V, M)
        data_tensor = torch.from_numpy(data).float()
        data_tensor = data_tensor.permute(3, 0, 2, 1)  # (C, T, V, M)

        # Apply max_frames limit with center cropping/padding
        data_tensor, frame_metadata = self._apply_frame_limit(data_tensor, frame_metadata)

        # Create metadata dict
        metadata = {
            'file_path': file_path,
            'class_name': self.idx_to_class[label],
            'parent_category': parent_category,
            'num_frames': data_tensor.shape[0],
            'frame_metadata': frame_metadata,
            'file_metadata': file_metadata
        }

        return data_tensor, label, metadata

    def _apply_frame_limit(self, data_tensor, frame_metadata):
        """
        Apply max_frames limit with center cropping/padding.
        
        Args:
            data_tensor: torch.Tensor of shape (C, T, V, M)
            frame_metadata: numpy array of frame metadata
            
        Returns:
            Tuple of (processed_data_tensor, processed_frame_metadata)
        """
        if not self.max_frames:
            return data_tensor, frame_metadata

        C, T, V, M = data_tensor.shape

        if T > self.max_frames:
            # Center crop along time dimension
            start = (T - self.max_frames) // 2
            data_tensor = data_tensor[:, start: start + self.max_frames, :, :]
            frame_metadata = frame_metadata[start: start + self.max_frames]

        elif T < self.max_frames:
            # Center padding along time dimension
            pad_num = self.max_frames - T
            left_pad = pad_num // 2
            right_pad = pad_num - left_pad

            # Create padding tensors
            left = torch.zeros(
                (C, left_pad, V, M),
                device=data_tensor.device,
                dtype=data_tensor.dtype,
            )
            right = torch.zeros(
                (C, right_pad, V, M),
                device=data_tensor.device,
                dtype=data_tensor.dtype,
            )

            data_tensor = torch.cat([left, data_tensor, right], dim=1)

            # Pad frame_metadata with zeros or appropriate values
            left_metadata = np.zeros((left_pad,) + frame_metadata.shape[1:], dtype=frame_metadata.dtype)
            right_metadata = np.zeros((right_pad,) + frame_metadata.shape[1:], dtype=frame_metadata.dtype)
            frame_metadata = np.concatenate([left_metadata, frame_metadata, right_metadata], axis=0)

        return data_tensor, frame_metadata

    def _normalize_landmarks(self, landmarks):
        """
        Normalize landmark coordinates.
        
        Args:
            landmarks: numpy array of shape (T, M, V, C)
        
        Returns:
            Normalized landmarks with same shape
        """
        # Simple min-max normalization per coordinate channel
        landmarks_norm = landmarks.copy()
        T, M, V, C = landmarks.shape

        # Normalize each coordinate channel independently
        for c in range(C):
            channel_data = landmarks[:, :, :, c]
            c_min = channel_data.min()
            c_max = channel_data.max()
            if c_max != c_min:
                landmarks_norm[:, :, :, c] = (channel_data - c_min) / (c_max - c_min)

        return landmarks_norm

    def get_class_names(self):
        """Return list of class names."""
        return [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

    def get_num_classes(self):
        """Return number of classes."""
        return len(self.class_to_idx)

    def get_parent_categories(self):
        """Return list of parent categories."""
        return sorted(self.parent_to_classes.keys())

    def get_classes_by_parent(self, parent_category):
        """Return list of classes for a given parent category."""
        return self.parent_to_classes.get(parent_category, [])


def collate_fn(batch):
    """
    Custom collate function for batching sequences of different lengths.
    
    Args:
        batch: List of (data, label, metadata) tuples where data has shape (C, T, V, M)
    
    Returns:
        data: Padded tensor of shape (batch_size, C, max_T, V, M)
        labels: Tensor of shape (batch_size,)
        lengths: Tensor of actual sequence lengths (time dimension)
        metadata: List of metadata dicts
    """
    data_list, labels, metadata = zip(*batch)

    # Get sequence lengths (time dimension)
    lengths = torch.tensor([data.shape[1] for data in data_list])  # T is at index 1

    # Find max time length
    max_time = max(data.shape[1] for data in data_list)

    # Get other dimensions (should be same for all samples)
    C, _, V, M = data_list[0].shape

    # Create padded batch tensor
    batch_size = len(data_list)
    padded_data = torch.zeros(batch_size, C, max_time, V, M)

    # Fill in the data
    for i, data in enumerate(data_list):
        T = data.shape[1]
        padded_data[i, :, :T, :, :] = data

    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_data, labels, lengths, list(metadata)


def create_data_loaders(root_dir: str,
                        batch_size: int = 32,
                        num_workers: int = 4,
                        use_world_landmarks: bool = True,
                        use_hand_landmarks: bool = True,
                        normalize: bool = True,
                        max_frames: Optional[int] = None,
                        shuffle_train: bool = True) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for train, test, and validation splits.
    
    Args:
        root_dir: Path to h5_output directory
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        use_world_landmarks: Whether to include world landmarks
        use_hand_landmarks: Whether to include hand landmarks
        normalize: Whether to normalize landmark coordinates
        max_frames: Maximum number of frames per sequence
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Dictionary containing DataLoaders for each split
    """
    dataloaders = {}

    # Available splits
    available_splits = []
    for split in ['train', 'test', 'val']:
        split_dir = os.path.join(root_dir, split)
        if os.path.exists(split_dir):
            available_splits.append(split)

    print(f"Found splits: {available_splits}")

    # Create datasets and dataloaders
    for split in available_splits:
        dataset = ISLVideoDataset(
            root_dir=root_dir,
            split=split,
            use_world_landmarks=use_world_landmarks,
            use_hand_landmarks=use_hand_landmarks,
            normalize=normalize,
            max_frames=max_frames
        )

        shuffle = shuffle_train if split == 'train' else False

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )

        dataloaders[split] = dataloader
        num_parent_categories = len(dataset.get_parent_categories())
        print(
            f"{split}: {len(dataset)} samples, {dataset.get_num_classes()} classes, {num_parent_categories} parent categories")

    return dataloaders


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    root_dir = "/Users/yashrb/Projects/isl_videos/landmarks"

    # Create dataloaders
    dataloaders = create_data_loaders(
        root_dir=root_dir,
        batch_size=8,
        num_workers=0,
        use_world_landmarks=False,
        use_hand_landmarks=True,
        normalize=True,
        max_frames=None  # Use all frames
    )

    # Test the dataloader
    for split, dataloader in dataloaders.items():
        print(f"\nTesting {split} dataloader:")
        for batch_idx, (data, labels, lengths, metadata) in enumerate(dataloader):
            print(f"  Batch {batch_idx}:")
            print(f"    Data shape: {data.shape} (N, C, T, V, M)")
            print(f"    Labels shape: {labels.shape}")
            print(f"    Lengths: {lengths}")
            print(f"    Sample classes: {[metadata[i]['class_name'] for i in range(min(3, len(metadata)))]}")
            print(
                f"    Sample parent categories: {[metadata[i]['parent_category'] for i in range(min(3, len(metadata)))]}")

            if batch_idx >= 2:  # Test first 3 batches
                break

import os
from typing import Dict, Optional
import random
import math

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
                 max_frames: Optional[int] = None,
                 augment: bool = False,
                 augment_prob: float = 0.3,
                 rotation_range: float = 15.0,
                 scale_range: tuple = (0.8, 1.2),
                 translation_range: float = 0.1,
                 noise_std: float = 0.02,
                 occlusion_prob: float = 0.1):
        """
        Args:
            root_dir: Path to h5_output directory
            split: 'train', 'test', or 'validation'
            use_world_landmarks: Whether to include world landmarks
            use_hand_landmarks: Whether to include hand landmarks
            normalize: Whether to normalize landmark coordinates (only applied to training split)
            max_frames: Maximum number of frames to use (truncate with center crop if longer)
            augment: Whether to apply data augmentations (only applied to training split)
            augment_prob: Probability for each augmentation to be applied
            rotation_range: Maximum rotation angle in degrees for rotation augmentation
            scale_range: (min_scale, max_scale) for scaling augmentation
            translation_range: Maximum translation offset as fraction of coordinate range
            noise_std: Standard deviation for Gaussian noise augmentation
            occlusion_prob: Probability of occluding individual landmarks
        """
        self.root_dir = root_dir
        self.split = split
        self.use_world_landmarks = use_world_landmarks
        self.use_hand_landmarks = use_hand_landmarks
        # Only normalize during training
        self.normalize = normalize and (split == 'train')
        self.max_frames = max_frames

        # Augmentation parameters (only apply during training)
        self.augment = augment and (split == 'train')
        self.augment_prob = augment_prob
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.noise_std = noise_std
        self.occlusion_prob = occlusion_prob
        self.EPS = 1e-8

        # Define semantic groups for 21-point hand model
        self._finger_groups = {
            'wrist': [0],
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }

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

        # Apply augmentations if requested (only during training)
        if self.augment:
            data_tensor = self._apply_augmentations(data_tensor)

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

    def _apply_augmentations(self, data_tensor):
        """
        Optimized augmentation pipeline with reduced tensor copies.
        Single clone at the start, then all augmentations work in-place.
        """
        # Decide which augmentations to apply upfront
        aug_decisions = {
            'rotate': random.random() < self.augment_prob,
            'scale': random.random() < self.augment_prob,
            'translate': random.random() < self.augment_prob,
            'noise': random.random() < self.augment_prob,
            'occlude': random.random() < self.augment_prob
        }

        # If no augmentations selected, return original
        if not any(aug_decisions.values()):
            return data_tensor

        # Single clone for all augmentations
        augmented = data_tensor.clone()

        # Apply spatial augmentations (rotation, scale, translation)
        # These work better when composed before clamping
        if aug_decisions['rotate']:
            augmented = self._aug_rotate(augmented, inplace=True)
        if aug_decisions['scale']:
            augmented = self._aug_scale(augmented, inplace=True)
        if aug_decisions['translate']:
            augmented = self._aug_translate(augmented, inplace=True)

        # Apply noise and occlusion
        if aug_decisions['noise']:
            augmented = self._aug_noise(augmented, inplace=True)
        if aug_decisions['occlude']:
            augmented = self._aug_occlude(augmented, inplace=True)

        return augmented

    def _aug_rotate(self, data_tensor, degrees=None, channels=(0, 1),
                    per_hand=True, clamp=True, inplace=False):
        """
        Rotate x,y plane around their center by a random angle.
        Now with per-hand rotation for better semantic preservation.

        Args:
            data_tensor: Input tensor (C, T, V, M)
            degrees: Rotation range in degrees (default: self.rotation_range)
            channels: Coordinate channels to rotate (default: (0,1) for x,y)
            per_hand: If True, rotate each hand around its own center
            clamp: If True, clamp coordinates to [0,1] after rotation
            inplace: If True, modify tensor in-place (assumes already cloned)
        """
        if degrees is None:
            degrees = float(self.rotation_range)

        C, T, V, M = data_tensor.shape
        device = data_tensor.device
        dtype = data_tensor.dtype

        # Check if we have the required channels
        ch = list(channels)
        if max(ch) >= C:
            return data_tensor

        # Generate rotation angle
        angle = random.uniform(-degrees, degrees) * math.pi / 180.0
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R = torch.tensor([[cos_a, -sin_a],
                          [sin_a, cos_a]], device=device, dtype=dtype)  # (2,2)

        # Work on existing tensor or clone
        out = data_tensor if inplace else data_tensor.clone()

        if per_hand:
            # Rotate each hand around its own center
            for m in range(M):
                # Extract coordinates for this hand
                xy_hand = out[ch, :, :, m].reshape(2, -1)  # (2, T*V)

                # Find center of this hand
                center = xy_hand.mean(dim=1, keepdim=True)  # (2, 1)

                # Center, rotate, and uncenter
                centered = xy_hand - center
                rotated = R @ centered
                rotated = rotated + center

                # Reshape back and update
                out[ch, :, :, m] = rotated.reshape(2, T, V)
        else:
            # Original implementation - rotate all landmarks around global center
            xy = out[ch, :, :, :].reshape(2, -1)  # (2, T*V*M)
            center = xy.mean(dim=1, keepdim=True)  # (2,1)

            centered = xy - center
            rotated = R @ centered
            rotated = (rotated + center).reshape(2, T, V, M)
            out[ch, :, :, :] = rotated

        if clamp:
            # Clamp only x,y to [0,1]
            out[ch, :, :, :] = out[ch, :, :, :].clamp(0.0, 1.0)

        return out

    def _aug_scale(self, data_tensor, scale_range=None, channels=(0, 1),
                   scale_z=False, inplace=False):
        """
        Uniform scaling around center in normalized coords.

        Args:
            inplace: If True, modify tensor in-place (assumes already cloned)
        """
        if scale_range is None:
            scale_range = tuple(self.scale_range)

        C, T, V, M = data_tensor.shape
        device = data_tensor.device
        dtype = data_tensor.dtype

        scale = random.uniform(scale_range[0], scale_range[1])

        ch = list(channels)
        if max(ch) >= C:
            return data_tensor

        out = data_tensor if inplace else data_tensor.clone()
        xy = out[ch, :, :, :]
        center = xy.mean(dim=(1, 2, 3), keepdim=True)  # (2,1,1,1)
        out[ch, :, :, :] = (xy - center) * scale + center

        # optionally scale z
        if scale_z and C > max(ch) + 1:
            z_idx = max(ch) + 1  # assume next channel is z if present
            z = out[z_idx:z_idx + 1, :, :, :]
            z_center = z.mean(dim=(1, 2, 3), keepdim=True)
            out[z_idx:z_idx + 1, :, :, :] = (z - z_center) * scale + z_center

        # keep coords in [0,1] for x,y
        out[ch, :, :, :] = out[ch, :, :, :].clamp(0.0, 1.0)

        return out

    def _aug_translate(self, data_tensor, translation_range=None, channels=(0, 1),
                       per_hand=False, per_frame=False, translate_z=False, inplace=False):
        """
        Apply translation.

        Args:
            inplace: If True, modify tensor in-place (assumes already cloned)
        """
        if translation_range is None:
            translation_range = float(self.translation_range)

        C, T, V, M = data_tensor.shape
        device = data_tensor.device
        dtype = data_tensor.dtype

        out = data_tensor if inplace else data_tensor.clone()
        ch = list(channels)
        if max(ch) >= C:
            return out

        # helper to sample offsets in [-r, r]
        def sample_offset(shape):
            return (torch.rand(shape, device=device, dtype=dtype) * 2.0 - 1.0) * translation_range

        if (not per_hand) and (not per_frame):
            # one offset for sample
            offset = sample_offset((len(ch),))  # (2,)
            offset = offset.view(len(ch), 1, 1, 1)
            out[ch, :, :, :] = out[ch, :, :, :] + offset
        elif per_hand and not per_frame:
            # offset per-hand: (2,1,1,M)
            offset = sample_offset((len(ch), M)).view(len(ch), 1, 1, M)
            out[ch, :, :, :] = out[ch, :, :, :] + offset
        elif per_frame and not per_hand:
            # offset per-frame: (2,T,1,1)
            offset = sample_offset((len(ch), T)).view(len(ch), T, 1, 1)
            out[ch, :, :, :] = out[ch, :, :, :] + offset
        else:
            # per-frame per-hand: (2,T,1,M)
            offset = sample_offset((len(ch), T, M)).view(len(ch), T, 1, M)
            out[ch, :, :, :] = out[ch, :, :, :] + offset

        # optionally translate z channel too
        if translate_z and C > max(ch) + 1:
            z_idx = max(ch) + 1
            if (not per_hand) and (not per_frame):
                z_offset = sample_offset((1,)).view(1, 1, 1, 1)
                out[z_idx:z_idx + 1, :, :, :] = out[z_idx:z_idx + 1, :, :, :] + z_offset
            else:
                z_offset = sample_offset((1, T if per_frame else 1, M if per_hand else 1))
                z_offset = z_offset.view(1, T if per_frame else 1, 1, M if per_hand else 1)
                out[z_idx:z_idx + 1, :, :, :] = out[z_idx:z_idx + 1, :, :, :] + z_offset

        # clamp x,y to [0,1]
        out[ch, :, :, :] = out[ch, :, :, :].clamp(0.0, 1.0)
        return out

    def _aug_noise(self, data_tensor, noise_std=None, channels=(0, 1),
                   per_frame=True, scale_by_range=False, include_z=False,
                   clamp=True, inplace=False):
        """
        Add Gaussian noise.
        Fixed: per_frame now defaults to True for temporal variation.

        Args:
            noise_std: Standard deviation in normalized units
            per_frame: If True, different noise each frame (default: True)
            scale_by_range: If True, scale noise by coordinate range
            include_z: Whether to add noise to z channel
            clamp: Whether to clamp values to [0,1]
            inplace: If True, modify tensor in-place
        """
        if noise_std is None:
            noise_std = float(self.noise_std)

        C, T, V, M = data_tensor.shape
        device = data_tensor.device
        dtype = data_tensor.dtype

        out = data_tensor if inplace else data_tensor.clone()
        ch = list(channels)
        if max(ch) >= C:
            return out

        if scale_by_range:
            # compute per-channel range across whole sample
            flat = out.reshape(C, -1)
            ch_mins = flat.min(dim=1)[0]
            ch_maxs = flat.max(dim=1)[0]
            ranges = (ch_maxs - ch_mins).clamp(min=self.EPS)  # (C,) - Now EPS is defined!
        else:
            ranges = torch.ones(C, device=device, dtype=dtype)

        # Generate noise
        if per_frame:
            # Different noise per frame (more realistic for temporal data)
            noise_shape = (len(ch), T, V, M)
        else:
            # Same noise pattern across all frames
            noise_shape = (len(ch), 1, V, M)

        noise = torch.randn(noise_shape, device=device, dtype=dtype)
        noise = noise * (noise_std * ranges[ch].view(len(ch), 1, 1, 1))

        # Apply noise
        if not per_frame:
            # Broadcast same noise to all frames
            noise = noise.expand(len(ch), T, V, M)

        out[ch, :, :, :] = out[ch, :, :, :] + noise

        # Optionally add noise to z channel
        if include_z and C > max(ch) + 1:
            z_idx = max(ch) + 1
            z_noise_shape = (1, T, V, M) if per_frame else (1, 1, V, M)
            z_noise = torch.randn(z_noise_shape, device=device, dtype=dtype)
            z_noise = z_noise * noise_std * (ranges[z_idx] if scale_by_range else 1.0)

            if not per_frame:
                z_noise = z_noise.expand(1, T, V, M)

            out[z_idx:z_idx + 1, :, :, :] = out[z_idx:z_idx + 1, :, :, :] + z_noise

        if clamp:
            out[ch, :, :, :] = out[ch, :, :, :].clamp(0.0, 1.0)

        return out

    def _aug_occlude(self, data_tensor, occlusion_prob=None, per_frame=False,
                     group_occlusion=True, semantic_groups=True, inplace=False):
        """
        Randomly zero-out (occlude) landmarks with improved semantic grouping.

        Args:
            occlusion_prob: Probability to occlude each landmark/group
            per_frame: If True, occlusion varies per frame
            group_occlusion: If True, occlude contiguous groups
            semantic_groups: If True, use finger-based semantic groups (overrides group_occlusion)
            inplace: If True, modify tensor in-place
        """
        if occlusion_prob is None:
            occlusion_prob = float(self.occlusion_prob)

        C, T, V, M = data_tensor.shape
        out = data_tensor if inplace else data_tensor.clone()

        if semantic_groups and V == 21:  # Check if we have 21-point hand model
            # Use semantic finger groups
            mask = torch.zeros((V, M), dtype=torch.bool)

            # Decide which finger groups to occlude
            for group_name, indices in self._finger_groups.items():
                if random.random() < occlusion_prob:
                    # Occlude this entire finger/group
                    for idx in indices:
                        mask[idx, :] = True

            # Optionally make occlusion hand-specific
            if random.random() < 0.3:  # 30% chance to have different occlusion per hand
                for m in range(M):
                    mask[:, m] = torch.zeros(V, dtype=torch.bool)
                    for group_name, indices in self._finger_groups.items():
                        if random.random() < occlusion_prob:
                            for idx in indices:
                                mask[idx, m] = True

        elif group_occlusion:
            # Fallback to simple contiguous group occlusion
            mask = torch.zeros((V, M), dtype=torch.bool)
            num_groups = max(1, int(V * occlusion_prob * 0.3))  # Reduce number of groups

            for _ in range(num_groups):
                # Pick a starting point
                start_idx = random.randint(0, V - 1)
                # Occlude a contiguous group of 2-4 landmarks
                group_size = random.randint(2, min(4, V - start_idx))
                for i in range(start_idx, min(start_idx + group_size, V)):
                    mask[i, :] = True
        else:
            # Random individual landmark occlusion
            mask = torch.rand((V, M)) < occlusion_prob

        # Apply occlusion
        if per_frame:
            # Different occlusion pattern per frame
            for t in range(T):
                if semantic_groups and V == 21:
                    # Re-generate semantic occlusion per frame
                    frame_mask = torch.zeros((V, M), dtype=torch.bool)
                    for group_name, indices in self._finger_groups.items():
                        if random.random() < occlusion_prob:
                            for idx in indices:
                                frame_mask[idx, :] = True
                else:
                    # Random occlusion per frame
                    frame_mask = torch.rand((V, M)) < occlusion_prob

                out[:, t, frame_mask] = 0.0
        else:
            # Consistent occlusion across all frames
            out[:, :, mask] = 0.0

        return out

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
                        shuffle_train: bool = True,
                        augment: bool = False,
                        augment_prob: float = 0.3,
                        rotation_range: float = 15.0,
                        scale_range: tuple = (0.8, 1.2),
                        translation_range: float = 0.1,
                        noise_std: float = 0.02,
                        occlusion_prob: float = 0.1) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for train, test, and validation splits.

    Args:
        root_dir: Path to h5_output directory
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        use_world_landmarks: Whether to include world landmarks
        use_hand_landmarks: Whether to include hand landmarks
        normalize: Whether to normalize landmark coordinates (only applied to training)
        max_frames: Maximum number of frames per sequence
        shuffle_train: Whether to shuffle training data
        augment: Whether to apply data augmentations (only applied to training)
        augment_prob: Probability for each augmentation to be applied
        rotation_range: Maximum rotation angle in degrees for rotation augmentation
        scale_range: (min_scale, max_scale) for scaling augmentation
        translation_range: Maximum translation offset as fraction of coordinate range
        noise_std: Standard deviation for Gaussian noise augmentation
        occlusion_prob: Probability of occluding individual landmarks

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
            max_frames=max_frames,
            augment=augment,
            augment_prob=augment_prob,
            rotation_range=rotation_range,
            scale_range=scale_range,
            translation_range=translation_range,
            noise_std=noise_std,
            occlusion_prob=occlusion_prob
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
    root_dir = "/Users/yashrb/Projects/isl-gnn/landmarks"

    # Create dataloaders with improved augmentations
    dataloaders = create_data_loaders(
        root_dir=root_dir,
        batch_size=8,
        num_workers=0,
        use_world_landmarks=False,
        use_hand_landmarks=True,
        normalize=True,
        max_frames=None,  # Use all frames
        augment=True,  # Enable augmentations for training
        augment_prob=0.3,  # 30% chance for each augmentation
        rotation_range=15.0,  # ±15 degrees rotation
        scale_range=(0.8, 1.2),  # 80% to 120% scaling
        translation_range=0.1,  # 10% translation range
        noise_std=0.02,  # 2% noise standard deviation
        occlusion_prob=0.1  # 10% chance to occlude each landmark/group
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
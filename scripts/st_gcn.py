import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))
        return x.contiguous(), A


class HandGraph:
    def __init__(self, strategy="spatial", connect_hands=True):
        self.num_nodes = 42  # 21 landmarks per hand * 2 hands
        self.strategy = strategy
        self.connect_hands = connect_hands
        self.A = self._get_adjacency()

    def _get_adjacency(self):
        if self.strategy == "spatial":
            # MediaPipe hand landmark connections for one hand
            single_hand_edges = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # index finger
                (0, 9), (9, 10), (10, 11), (11, 12),  # middle finger
                (0, 13), (13, 14), (14, 15), (15, 16),  # ring finger
                (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
                (5, 9), (9, 13), (13, 17),  # palm connections
            ]

            # Initialize adjacency matrix for 42 nodes (2 hands)
            A = np.zeros((1, self.num_nodes, self.num_nodes))

            # Add edges for first hand (nodes 0-20) and second hand (nodes 21-41)
            for i, j in single_hand_edges:
                A[0, i, j] = 1
                A[0, j, i] = 1
                A[0, i + 21, j + 21] = 1
                A[0, j + 21, i + 21] = 1

            # Optionally connect the two hands (wrist to wrist)
            if self.connect_hands:
                A[0, 0, 21] = 1  # Connect left wrist to right wrist
                A[0, 21, 0] = 1  # Connect right wrist to left wrist

            # Add self-loops
            A[0] = A[0] + np.eye(self.num_nodes)

            # Normalize adjacency matrix
            D = np.sum(A[0], axis=1)
            D_inv = np.power(D, -0.5)
            D_inv[np.isinf(D_inv)] = 0
            D_inv = np.diag(D_inv)
            A[0] = D_inv @ A[0] @ D_inv

            return A


class st_gcn(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dropout=-1, residual=True
    ):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(
            in_channels,
            out_channels,
            kernel_size[1],
            t_kernel_size=1,
            t_stride=1,
            t_padding=0,
            t_dilation=1,
            bias=True,
        )

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=False),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


class HandSTGCN(nn.Module):
    """Spatial Temporal Graph Convolutional Networks for MediaPipe hand landmarks.

    Args:
        in_channels (int): Number of channels in the input data (usually 2 or 3 for x,y or x,y,z coordinates)
        num_class (int): Number of classes for the classification task
        edge_importance_weighting (bool): If ``True``, adds a learnable importance weighting
                                         to the edges of the graph
        dropout (float): Dropout rate used in the network

    Shape:
        - Input: (N, C, T, V, M) where
            N is batch size
            C is number of channels (coordinates)
            T is length of input sequence (frames)
            V is number of nodes (21 hand landmarks per hand)
            M is number of hands (2)
        - Output: (N, num_class)
    """

    def __init__(
        self, in_channels=2, num_class=262, edge_importance_weighting=True, dropout=0.4
    ):
        super().__init__()

        # Load hand graph
        self.graph = HandGraph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        # Build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # Data normalization
        self.data_bn = nn.BatchNorm1d(in_channels * 42)

        # Define ST-GCN layers
        self.st_gcn_networks = nn.ModuleList(
            (
                st_gcn(in_channels, 32, kernel_size, 1, dropout=dropout, residual=True),
                st_gcn(32, 64, kernel_size, 2, dropout=dropout, residual=True),
                st_gcn(64, 64, kernel_size, 2, dropout=dropout, residual=True),
                st_gcn(64, 64, kernel_size, 2, dropout=dropout, residual=True),
                st_gcn(64, 64, kernel_size, 2, dropout=dropout, residual=True),
                st_gcn(64, 64, kernel_size, 2, dropout=dropout, residual=True),
            )
        )

        # Edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # FCN for prediction
        self.fcn = nn.Conv2d(64, num_class, kernel_size=1)

    def forward(self, x):
        """
        x: Input tensor with shape (N, C, T, V, M)
            N: batch size
            C: number of channels (x,y or x,y,z coordinates)
            T: number of frames
            V: number of landmarks (21 for MediaPipe hand)
            M: number of hands (2)
        """
        N, C, T, V, M = x.size()
        
        # Combine both hands into a single graph with 42 nodes
        # Reshape from (N, C, T, 21, 2) to (N, C, T, 42)
        x = x.permute(0, 1, 2, 4, 3).contiguous()  # (N, C, T, M, V)
        x = x.view(N, C, T, M * V)  # (N, C, T, 42)
        
        # Reshape for batch normalization
        # We need (N, C*V, T) for BatchNorm1d
        x = x.permute(0, 1, 3, 2).contiguous()  # (N, C, 42, T)
        x = x.view(N, C * M * V, T)  # (N, 84, T) for C=2 or (N, 126, T) for C=3
        x = self.data_bn(x)
        x = x.view(N, C, M * V, T)  # (N, C, 42, T)
        x = x.permute(0, 1, 3, 2).contiguous()  # (N, C, T, 42)

        # Forward through ST-GCN networks
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # Global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)

        # Prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):
        """
        Extract features from the network for visualization or other purposes

        x: Input tensor with shape (N, C, T, V, M)
        """
        N, C, T, V, M = x.size()
        
        # Combine both hands into a single graph with 42 nodes
        x = x.permute(0, 1, 2, 4, 3).contiguous()  # (N, C, T, M, V)
        x = x.view(N, C, T, M * V)  # (N, C, T, 42)
        
        # Reshape for batch normalization
        x = x.permute(0, 1, 3, 2).contiguous()  # (N, C, 42, T)
        x = x.view(N, C * 42, T)
        x = self.data_bn(x)
        x = x.view(N, C, 42, T)
        x = x.permute(0, 1, 3, 2).contiguous()  # (N, C, T, 42)

        # Forward through ST-GCN networks
        features = []
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
            features.append(x)

        # Get final feature
        _, c, t, v = x.size()
        feature = x.view(N, c, t, v)

        # Prediction
        x = self.fcn(x)
        output = x.view(N, -1, t, v)

        return output, feature, features


# Example usage
def test_model():
    # Create sample data: [batch_size, coordinates, frames, landmarks, hands]
    # Example with batch_size=4, 2 coordinates (x,y), 30 frames, 21 landmarks, 2 hands
    sample_data = torch.randn(4, 2, 30, 21, 2)

    # Initialize model
    model = HandSTGCN(in_channels=2, num_class=262, dropout=0.4)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Check adjacency matrix size
    print(f"Adjacency matrix shape: {model.A.shape}")
    print(f"Edge importance shape: {model.edge_importance[0].shape}")
    
    # Forward pass
    output = model(sample_data)

    print(f"Input shape: {sample_data.shape}")
    print(f"Output shape: {output.shape}")

    # Extract features
    output_features, final_feature, all_features = model.extract_feature(sample_data)
    print(f"Extracted feature shape: {final_feature.shape}")
    print(f"Number of intermediate features: {len(all_features)}")


if __name__ == "__main__":
    test_model()
import torch.nn as nn


class CNN1D(nn.Module):
    """
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Pararmetes:
        n_targets: number of classes

    """

    def __init__(
        self,
        n_features,
        n_targets,
        window_size,
        feature_dim=10,
        kernel_size=6,
        stride=1,
        activation=nn.ReLU(),
        dropout=0.2,
    ):
        super(CNN1D, self).__init__()

        self.n_features = n_features
        self.n_targets = n_targets
        self.activation = activation
        self.window_size = window_size
        self.pooling = nn.MaxPool1d

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                self.n_features,
                feature_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                bias=True,
            ),
            self.pooling(2),
            nn.BatchNorm1d(feature_dim, feature_dim),
            self.activation,
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                feature_dim,
                2 * feature_dim,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=True,
            ),
            self.pooling(2),
            nn.BatchNorm1d(2 * feature_dim, 2 * feature_dim),
            self.activation,
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(
                2 * feature_dim,
                2 * feature_dim,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=True,
            ),
            self.pooling(2),
            nn.BatchNorm1d(2 * feature_dim, 2 * feature_dim),
            self.activation,
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(
                2 * feature_dim,
                2 * feature_dim,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=True,
            ),
            self.pooling(2),
            nn.BatchNorm1d(2 * feature_dim, 2 * feature_dim),
            self.activation,
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(
                2 * feature_dim,
                4 * feature_dim,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=True,
            ),
            self.pooling(2),
            nn.BatchNorm1d(4 * feature_dim, 4 * feature_dim),
            self.activation,
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(
                4 * feature_dim,
                self.window_size * feature_dim,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=True,
            ),
            self.pooling(2),
            nn.BatchNorm1d(
                self.window_size * feature_dim, self.window_size * feature_dim
            ),
            self.activation,
        )

        self.dropout = nn.Dropout(dropout, inplace=False)

        self.fc1 = nn.Sequential(
            nn.Linear(self.window_size * feature_dim, 8 * feature_dim, bias=True),
            self.activation,
            self.dropout,
        )

        self.fc2 = nn.Sequential(
            nn.Linear(8 * feature_dim, 8 * feature_dim, bias=True),
            self.activation,
            self.dropout,
        )

        self.output = nn.Linear(8 * feature_dim, self.n_targets, bias=True)

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight, 0.1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # CNNs
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)

        # Dimension Fitting
        out = out.mean(-1)
        out = out.view(out.size(0), -1)

        # Dense Layers
        out = self.fc1(out)
        out = self.fc2(out)

        # Output
        output = self.output(out)

        return output


if __name__ == "__main__":
    pass

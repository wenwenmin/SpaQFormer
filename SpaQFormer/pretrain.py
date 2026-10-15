import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, act=F.relu):
        super(Encoder, self).__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one dimension.")

        self.layers = nn.ModuleList()
        self.act = act
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)


class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, z):
        return self.fc(z)


class Decoder(nn.Module):
    def __init__(self, hidden_dims, input_dim, act=F.relu):
        super(Decoder, self).__init__()
        reversed_dims = hidden_dims[::-1]
        if len(reversed_dims) < 1:
            raise ValueError("hidden_dims must contain at least one dimension.")

        self.layers = nn.ModuleList()
        self.act = act
        if len(reversed_dims) > 1:
            for i in range(len(reversed_dims) - 1):
                self.layers.append(nn.Linear(reversed_dims[i], reversed_dims[i + 1]))
        self.layers.append(nn.Linear(reversed_dims[-1], input_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)


class PretrainModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super(PretrainModel, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims)
        self.cls = Classifier(hidden_dims[-1], num_classes)
        self.decoder = Decoder(hidden_dims, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        class_logits = self.cls(z)
        recon_x = self.decoder(z)
        return z, class_logits, recon_x


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        weight = self.weight.to(inputs.device) if self.weight is not None else None
        ce_loss = nn.CrossEntropyLoss(weight=weight, reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class SingleCellDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

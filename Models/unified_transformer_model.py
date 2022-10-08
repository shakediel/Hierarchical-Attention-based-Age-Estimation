import torch
import torch.nn as nn
import torch.nn.functional as F


class AgeTransformer(nn.Module):
    def __init__(self, backbone, transformer, backbone_feature_size, transformer_feature_size):#, num_queries, aux_loss=False):
        super().__init__()

        self.backbone = backbone
        self.transformer = transformer
        self.linear = nn.Linear(backbone_feature_size, transformer_feature_size)

        self.batch_norm = nn.BatchNorm1d(transformer_feature_size)

        # self.num_classes = num_classes
        # self.num_queries = num_queries
        # self.aux_loss = aux_loss

    def FreezeBaseCnn(self, OnOff):
        for param in self.backbone.parameters():
            param.requires_grad = not OnOff

    def forward(self, input):
        unpacked_input = input.view(input.shape[0] * input.shape[1], input.shape[2], input.shape[3], input.shape[4])
        embs = self.backbone(unpacked_input)

        embs = self.linear(embs)
        embs = F.leaky_relu(embs)

        # embs = self.batch_norm(embs)

        embs = F.normalize(embs, dim=1, p=2)

        packed_embs = embs.view(input.shape[0], input.shape[1], -1)
        # packed_embs = embs.reshape((int(embs.shape[0] / input.shape[1]), int(input.shape[1]), embs.shape[1]))

        output = self.transformer(packed_embs)

        return output

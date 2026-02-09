import torch
import torch.nn as nn
from torchvision import models
from models.networks.GNN import GNN, SpatialGNN
import datasets.util.viz_utils as viz_utils


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True),
    )

import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch.nn.functional as F
class Visualizer:
    def __init__(self):
        pass

    def save_grid_semantic_matrix(self, A, full_path="adj_matrix"):
        """
        可视化并保存邻接矩阵 A (Adjacency Matrix)
        """
        # A 形状为 [BT, N, N]
        matrices = A.detach().cpu().numpy()
        num_samples = matrices.shape[0]
        for i in range(num_samples):
            adj = matrices[i]

            plt.figure(figsize=(10, 8))
            # 使用绘图顶刊常用的色彩映射 'rocket' 或 'viridis'
            sns.heatmap(adj, cmap='viridis', robust=True, square=True)

            plt.title(f"Dynamic Semantic Correlation adj_matrix")
            plt.xlabel("Spatial Node Index")
            plt.ylabel("Spatial Node Index")

            # 同时保存 png 和 pdf (pdf方便论文无限放大)
            file_name = f"class_grid__matrix_{i:03d}"

            # 保存 PNG (预览快)
            plt.savefig(os.path.join(full_path, f"{file_name}.png"), dpi=300, bbox_inches='tight')
            # plt.savefig(os.path.join(self.save_dir, f"{name}.pdf"), bbox_inches='tight')
            plt.close()

    def save_output(self, out, full_path="prediction"):
        """
        可视化预测的语义地图（取通道最大值）
        """
        # out 形状为 [BT, 27, 64, 64]
        # pred = torch.argmax(out[-1], dim=0).detach().cpu().numpy()
        preds = out.argmax(dim=1).detach().cpu().numpy()

        num_samples = preds.shape[0]
        print(f"Saving {num_samples} semantic maps to {full_path}...")

        for i in range(num_samples):
            # 2. 调用工具函数编码颜色
            # 假设 viz_utils.colorEncode 接受的是 [H, W] 的类索引矩阵
            color_step_grid27 = viz_utils.colorEncode(preds[i])

            plt.figure(figsize=(6, 6))
            plt.imshow(color_step_grid27, cmap='tab20')  # 使用离散颜色表表示不同类别
            plt.axis('off')
            plt.title("Semantic Map Prediction")
            file_name = f"pred_{i:04d}.png"
            plt.savefig(os.path.join(full_path, file_name), dpi=300)
            plt.close()

    def save_class_semantic_matrix(self, sigma, full_path="sigma_matrix"):
        """
        可视化并保存类别相关性矩阵 sigma [BT, C, C]
        """
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        matrices = sigma.detach().cpu().numpy()
        num_samples = matrices.shape[0]

        for i in range(num_samples):
            mat = matrices[i]

            plt.figure(figsize=(10, 8))
            # 使用 'magma' 或 'rocket' 色系，这类色系在表达激活强度时视觉对比度更高
            sns.heatmap(mat, cmap='viridis', robust=True, square=True)

            plt.title(f"Dynamic Class Correlation (Sigma) - Sample {i}")
            plt.xlabel("Category Index")
            plt.ylabel("Category Index")

            file_name = f"class_semantic_matrix_{i:04d}.png"
            plt.savefig(os.path.join(full_path, file_name), dpi=300, bbox_inches='tight')
            plt.close()

    def save_grid_spatial_matrix(self, A_static, full_path="spatial_prior"):
        """
        可视化并保存静态的空间邻接矩阵 (Spatial Prior)
        A_static 形状为 [N, N]
        """
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        # 1. 转换并处理
        adj = A_static.detach().cpu().numpy()

        plt.figure(figsize=(10, 8))
        # 空间矩阵通常数值范围较大，使用 robust=True 自动调整色彩范围
        # sns.heatmap(adj, cmap='coolwarm', robust=True, square=True)  viridis
        sns.heatmap(adj, cmap='viridis', robust=True, square=True)

        plt.title("Static Spatial Distance Prior (Manhattan/Euler)")
        plt.xlabel("Spatial Node Index")
        plt.ylabel("Spatial Node Index")

        # 2. 保存
        plt.savefig(os.path.join(full_path, "class_spatial_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()


class ResNetUNet(nn.Module):
    def __init__(self, n_channel_in, n_class_out):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(n_channel_in, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(n_channel_in, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class_out, 1)

    def forward(self, input):
        B, T, C, cH, cW = input.shape
        input = input.view(B*T, C, cH, cW)

        x_original = self.conv_original_size0(input)  # [B*T, 64, 64, 64 ]
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)  # [B*T, 64, 32, 32 ]
        layer1 = self.layer1(layer0)  # [B*T, 64, 16, 16 ]
        layer2 = self.layer2(layer1)   # [B*T, 128, 8, 8 ]
        layer3 = self.layer3(layer2)  # [B*T, 256, 4, 4 ]
        layer4 = self.layer4(layer3)  # [B*T, 512, 2, 2 ]

        layer4 = self.layer4_1x1(layer4)  # [B*T, 512, 2, 2 ]
        x = self.upsample(layer4)  # [B*T, 512, 4, 4 ]

        layer3 = self.layer3_1x1(layer3)  # [B*T, 256, 4, 4 ]
        x = torch.cat([x, layer3], dim=1)  # [B*T, 768, 4, 4 ]
        x = self.conv_up3(x)   # [B*T, 512, 4, 4 ]

        x = self.upsample(x)  # [B*T, 512, 8, 8 ]
        layer2 = self.layer2_1x1(layer2)  # [B*T, 128, 8, 8 ]
        x = torch.cat([x, layer2], dim=1)  # [B*T, 640, 8, 8 ]
        x = self.conv_up2(x)  # [B*T, 256, 8, 8 ]

        x = self.upsample(x)  # [B*T, 256, 16, 16 ]
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)  # [B*T, 320, 16, 16 ]
        x = self.conv_up1(x)  # [B*T, 256, 16, 16 ]

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)  # [B*T, 128, 32, 32 ]

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)  # [B*T, 64, 64, 64 ]

        out = self.conv_last(x)  # [B*T, 27, 64, 64 ]

        return out


class AE(nn.Module):
    def __init__(self, N, C, in_channels, inter_channels, out_channels, pool=(2, 2), factor=2):
        super().__init__()
        self.C = C
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(pool)
        self.pool_sp = nn.MaxPool2d((2, 2))
        self.linearKC = nn.Linear(inter_channels, C)
        self.linearNC = nn.Linear(N*N, C)
        self.gnn = GNN(inter_channels)
        self.spatialgnn = SpatialGNN(inter_channels, N, N)

        self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
        self.back = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, SP):
        '''x: bs, f, h, w
            SP: bs, C, h, w
        '''
        SP = torch.softmax(SP, dim=1)
        SP = self.pool(SP)  # bs, C, h/2, w/2
        SP = SP.reshape(SP.shape[0], SP.shape[1], -1)  # bs, C, n

        t = self.trans(x)  # bs, k, h/4, w/4
        y = t
        # y = self.pool(t)  # bs, k, h/2, w/2
        size = y.shape
        y = y.reshape(y.shape[0], y.shape[1], -1)  # bs, k, n

        # object
        # no object
        # A = torch.matmul(y.permute(0, 2, 1), y)

        # with object
        sigma = self.linearKC(self.linearNC(y).permute(0, 2, 1))  # bs, c, c
        A = torch.matmul(SP.permute(0, 2, 1), torch.matmul(sigma, SP))  # bs, n, n

        y = y.permute(0, 2, 1)  # bs, n, k

        # y = self.gnn(A, y) + y
        se_y = self.gnn(A, y) + y


        # no spatial
        # y = self.spatialgnn(y) + y
        sp_y = self.spatialgnn(y) + y

        # gate
        # y = self.gate(se_y, sp_y)
        y = se_y+sp_y

        y = self.dropout(self.up(y.permute(0, 2, 1).reshape(size))) + t
        y = self.back(y)
        return self.dropout(y)


class AM(nn.Module):
    def __init__(self, N, C, in_channels, inter_channels, out_channels, pool=(2, 2), factor=2):
        super().__init__()
        self.C = C
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(pool)
        self.pool_sp = nn.MaxPool2d((2, 2))
        self.linearKC = nn.Linear(inter_channels, C)
        self.linearNC = nn.Linear(N*N, C)
        self.gnn = GNN(inter_channels)
        self.spatialgnn = SpatialGNN(inter_channels, N, N)

        self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
        self.back = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, SP):
        '''x: bs, f, h, w
            SP: bs, C, h, w
        '''
        SP = torch.softmax(SP, dim=1)
        SP = self.pool(SP)  # bs, C, h/2, w/2
        SP = SP.reshape(SP.shape[0], SP.shape[1], -1)  # bs, C, n

        t = self.trans(x)  # bs, k, h/4, w/4
        y = t
        # y = self.pool(t)  # bs, k, h/2, w/2
        size = y.shape
        # 【论文】将X展平为X'
        y = y.reshape(y.shape[0], y.shape[1], -1)  # bs, k, n

        # no object
        # A = torch.matmul(y.permute(0, 2, 1), y)

        # 【论文】sigma是类别关系矩阵 ˆA_Obj = W1 * X' * W2
        sigma = self.linearKC(self.linearNC(y).permute(0, 2, 1))  # bs, c, c
        # 【论文】语义关系邻接矩阵
        A = torch.matmul(SP.permute(0, 2, 1), torch.matmul(sigma, SP))  # bs, n, n

        y = y.permute(0, 2, 1)  # bs, n, k
        # y = self.gnn(A, y) + y
        sm_y = self.gnn(A, y) + y

        # y = self.spatialgnn(y) + y
        # 【论文】空间关系邻接矩阵
        sp_y = self.spatialgnn(y) + y

        # gate
        # y = self.gate(sm_y, sp_y)
        y = sm_y + sp_y

        y = self.dropout(self.up(y.permute(0, 2, 1).reshape(size))) + t
        y = self.back(y)
        return self.dropout(y), A, sigma


class ResNetUNetDAMLastLayerv2(nn.Module):
    def __init__(self, n_channel_in, n_class_out):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(n_channel_in, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer0_ae = AE(32, 27, 64, 64, 64, pool=(2, 2), factor=1)

        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer1_ae = AE(16, 27, 64, 64, 64, pool=(4, 4), factor=1)

        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer2_ae = AE(8, 27, 128, 128, 128, pool=(8, 8), factor=1)

        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer3_ae = AE(4, 27, 256, 256, 256, pool=(16, 16), factor=1)

        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        self.layer4_ae = AE(2, 27, 512, 512, 512, pool=(32, 32), factor=1)

        # FIXME:（U-Net的创新点1）以上为编码器部分
        # FIXME:（U-Net的创新点2）以下为解码器部分

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)   # 解码器中的 conv_up3 = convrelu(768, 512, 3, 1) 负责将这两个来自不同深度、不同性质的特征图融合并压缩回标准通道数。
        self.up3_ae = AE(4, 27, 512, 512, 512, pool=(16, 16), factor=1)

        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.up2_ae = AE(8, 27, 256, 256, 256, pool=(8, 8), factor=1)

        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.up1_ae = AE(16, 27, 256, 256, 256, pool=(4, 4), factor=1)

        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        self.up0_ae = AE(32, 27, 128, 128, 128, pool=(2, 2), factor=1)

        self.conv_original_size0 = convrelu(n_channel_in, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        self.up0_last = AM(64, 27, 64, 64, 64, pool=(1, 1), factor=1)

        self.conv_last = nn.Conv2d(64, n_class_out, 1)

        # 可视化
        self.viz = Visualizer()
        # self.register_buffer('step_counter', torch.tensor(0))  # 使用 buffer 保证能随模型保存/加载


    def forward(self, input, step_name="1"):
        B, T, C, cH, cW = input.shape
        input = input.view(B*T, C, cH, cW)

        x_original = self.conv_original_size0(input)  # [B*T, 64, 64, 64 ]
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)  # [B*T, 64, 32, 32 ]
        layer0 = self.layer0_ae(layer0, input)

        layer1 = self.layer1(layer0)  # [B*T, 64, 16, 16 ]
        layer1 = self.layer1_ae(layer1, input)

        layer2 = self.layer2(layer1)   # [B*T, 128, 8, 8 ]
        layer2 = self.layer2_ae(layer2, input)

        layer3 = self.layer3(layer2)  # [B*T, 256, 4, 4 ]
        layer3 = self.layer3_ae(layer3, input)

        layer4 = self.layer4(layer3)  # [B*T, 512, 2, 2 ]
        layer4 = self.layer4_ae(layer4, input)

        layer4 = self.layer4_1x1(layer4)  # [B*T, 512, 2, 2 ]
        x = self.upsample(layer4)  # [B*T, 512, 4, 4 ]

        layer3 = self.layer3_1x1(layer3)  # [B*T, 256, 4, 4 ]
        x = torch.cat([x, layer3], dim=1)  # [B*T, 768, 4, 4 ]     # FIXME:（U-Net的创新点3）: 融合跳跃连接
        x = self.conv_up3(x)   # [B*T, 512, 4, 4 ]
        x = self.up3_ae(x, input)

        x = self.upsample(x)  # [B*T, 512, 8, 8 ]
        layer2 = self.layer2_1x1(layer2)  # [B*T, 128, 8, 8 ]
        x = torch.cat([x, layer2], dim=1)  # [B*T, 640, 8, 8 ]     # FIXME:（U-Net的创新点3）: 融合跳跃连接
        x = self.conv_up2(x)  # [B*T, 256, 8, 8 ]
        x = self.up2_ae(x, input)

        x = self.upsample(x)  # [B*T, 256, 16, 16 ]
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)  # [B*T, 320, 16, 16 ]     # FIXME:（U-Net的创新点3）: 融合跳跃连接
        x = self.conv_up1(x)  # [B*T, 256, 16, 16 ]
        x = self.up1_ae(x, input)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)  # [B*T, 128, 32, 32 ]
        x = self.up0_ae(x, input)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)  # [B*T, 64, 64, 64 ]
        x, up0_am, sigma = self.up0_last(x, input)

        # 可视化
        # self.viz.save_grid_semantic_matrix(up0_am, full_path=step_name)
        # self.viz.save_class_semantic_matrix(sigma, full_path=step_name)
        # self.viz.save_grid_spatial_matrix(self.up0_last.spatialgnn.A,  full_path=step_name)

        out = self.conv_last(x)  # [B*T, 27, 64, 64 ]
        # self.viz.save_output(out, full_path=step_name)

        return out, up0_am


if __name__ == "__main__":

    x = torch.rand(4, 10, 27, 64, 64)
    x5 = torch.rand(4, 27, 128, 128)
    rgb = torch.rand(4, 10, 3, 128, 128)

    model = ResNetUNetDAMLastLayerv2(n_channel_in=27, n_class_out=27)


    # y = model(rgb)
    for i in range(5):
        y = model(x)
        print('y', y[0].shape)


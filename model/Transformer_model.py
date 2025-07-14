import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = rearrange(x, 'b 1 n l -> b n l')
        # [batch_size, num_patches+1, inner_dim*3] --> ([batch_size, num_patches+1, inner_dim], -->(q,k,v)
        #                                               [batch_size, num_patches+1, inner_dim],
        #                                               [batch_size, num_patches+1, inner_dim])
        #将x转变为qkv
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 对tensor进行分块

        q, k, v = \
            map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = rearrange(out, 'b n l -> b 1 n l')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.pooling = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNorm(dim, ResNetBlock()),
                # PreNorm(dim//(2**i), ConvAttention(dim//(2**i), heads, dim_head//(2**i), dropout)),
                # PreNorm(dim//(2**(i+1)), FeedForward(dim//(2**(i+1)), mlp_dim, dropout))
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x


class TimeTransformer(nn.Module):
    def __init__(self, *, input_dim,  num_patches=16, dim, depth, heads, mlp_dim,
                 pool='cls', channels=1, dim_head, emb_dropout=0., dropout=0.):
        super(TimeTransformer, self).__init__()

        # self.to_patch_embedding = Embedding(input_dim, dim)
        self.to_patch_embedding = self.to_patch_embedding = nn.Sequential(
            Rearrange('b 1 (n d) -> b 1 n d', n=num_patches),
            nn.Linear(input_dim//num_patches, dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))  # [1, 1, 1, dim] 随机数
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # [1, num_patches+1, dim] 随机数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()  # 这个恒等函数，就如同名字占位符，并没有实际操作

    def forward(self, rawdata):
        TimeSignals = rawdata   # Get Time Domain Signals
        # TimeSignals = rearrange(TimeSignals, 'b l -> b 1 l')
        # print(TimeSignals.shape, rawdata.shape)

        x = self.to_patch_embedding(TimeSignals)
        b, _, n, _ = x.shape      # x: [batch_size, channels, num_patches, dim]

        cls_tokens = repeat(self.cls_token, '() c n d -> b c n d', b=b)  # cls_tokens: [batch_size, c, num_patches, dim]
        x = torch.cat((cls_tokens, x), dim=2)  # x: [batch_size, c, num_patches+1, dim]
        # print(x.shape)
        #x += self.pos_embedding[:, :(n + 1)]  # 添加位置编码：x: [batch_size, c, num_patches+1, dim]
        x = self.dropout(x)

        x = self.transformer(x)     # x: [batch_size, c, num_patches+1, dim]
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, :, 0]  # x: [batch_size, c, 1, dim]
        x = self.to_latent(x)
        return x


class DSCTransformer(nn.Module):
    def __init__(self, *, input_dim, dim, depth, heads, mlp_dim, pool='cls',
                 num_classes, channels=1, dim_head, emb_dropout=0., dropout=0.):
        super(DSCTransformer, self).__init__()
        self.in_dim_time = input_dim
        self.TimeTrans = TimeTransformer(input_dim=self.in_dim_time, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim,
                                         pool=pool, dim_head=dim_head, emb_dropout=emb_dropout, dropout=dropout)

        self.mlp_head = nn.Sequential(
            Rearrange('b c l -> b (l c)'),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.dim = 256

    def forward(self, x):
        TimeSignals = x
        TimeFeature = self.TimeTrans(TimeSignals)
        y = self.mlp_head(TimeFeature)

        return y  # [batch_size, 1, num_classes]
    def get_embedding(self, x):
        return self.forward(x)
    def output_num(self):
        return self.dim


class Conv1DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv1DLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class NetworkModel(nn.Module):
    def __init__(self,pretrained=False):
        super(NetworkModel, self).__init__()
        self.conv1 = Conv1DLayer(1, 16, 64, 2, 1)
        self.pool1 = nn.MaxPool1d(2,2)

        self.conv2 = Conv1DLayer(16, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = Conv1DLayer(32, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = Conv1DLayer(64, 128, 3, 1, 1)
        self.pool4 = nn.MaxPool1d(2)

        self.position_embedding = nn.Embedding(1024, 128)  # Adjust the max length as needed
        self.transformer1 = TransformerLayer(128, 8, 128)
        self.transformer2 = TransformerLayer(128, 8, 128)

        self.fc1 = nn.Linear(128, 512)
        self.norm1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.norm2 = nn.BatchNorm1d(256)
        self.dim=256
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.pool4(x)

        x = x.permute(2, 0, 1)  # (N, C, L) -> (L, N, C)
        positions = torch.arange(0, x.size(0)).unsqueeze(1).expand(x.size(0), x.size(1)).to(x.device)
        x = x + self.position_embedding(positions)

        x = self.transformer1(x)
        x = self.transformer2(x)

        x = x.permute(1, 0, 2)  # (L, N, C) -> (N, L, C)
        x = x.mean(dim=1)  # Global average pooling

        # x = F.relu(self.norm1(self.fc1(x)))
        # x = F.relu(self.norm2(self.fc2(x)))
        x= self.fc1(x)
        x = self.fc2(x)
        return x
    def get_embedding(self, x):
        return self.forward(x)
    def output_num(self):
        return self.dim

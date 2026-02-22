import argparse
import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        laplacian = calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        self.laplacian = laplacian.cuda()
        weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.weights = weights.cuda()
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + 1) * batch_size)
        )
        a_times_concat = self.laplacian @ concatenation
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + 1, batch_size)
        )
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        outputs = a_times_concat @ self.weights
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }

class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim * 2, bias=0.05
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state):
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}

class ChebyKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        self.epsilon = 1e-7
        self.pre_mul = True
        self.post_mul = True
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        # View and repeat input degree + 1 times
        b,c_in = x.shape
        if self.pre_mul:
            mul_1 = x[:,::2]
            mul_2 = x[:,1::2]
            mul_res = mul_1 * mul_2
            x = torch.cat([x[:, :x.shape[1]//2], mul_res], dim=1)
        x = x.view((b, c_in, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        x = torch.tanh(x)
        x = torch.acos(x)
        x = x* self.arange
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        if self.post_mul:
            mul_1 = y[:,::2]
            mul_2 = y[:,1::2]
            mul_res = mul_1 * mul_2
            y = torch.cat([y[:, :y.shape[1]//2], mul_res], dim=1)

        return y

class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features,order):
        super().__init__()
        self.fc1 = ChebyKANLinear(
                            in_features,
                            out_features,
                            order)
    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,-1).contiguous()
        return x

class BasicConv(nn.Module):
    def __init__(self,c_in,c_out, kernel_size, degree,stride=1, padding=0, dilation=1, groups=1, act=False, bn=False, bias=False,dropout=0.):
        super(BasicConv, self).__init__()
        self.out_channels = c_out
        self.conv = nn.Conv1d(c_in,c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x.transpose(-1,-2)).transpose(-1,-2)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class Embedding(nn.Module):
    def __init__(self, linear: nn.Module, in_norm: nn.Module):
        super().__init__()
        self.linear = linear
        self.in_norm = in_norm

    def forward(self, case):
        B, T, N = case.shape
        case_all = case.view(B * T, N, 1)
        x2 = self.linear(case_all).squeeze(-1)
        return self.in_norm(x2)

class Dual_layer_T_GCN_Aggregation_block(nn.Module):

    def __init__(self, tgcn_cell1: nn.Module, tgcn_cell2: nn.Module,
                  hidden_dim: int):
        super().__init__()
        self.tgcn_cell1 = tgcn_cell1
        self.tgcn_cell2 = tgcn_cell2
        self._hidden_dim = hidden_dim

    def forward(self, inputs: torch.Tensor, B: int, T: int, N: int) -> torch.Tensor:
        D = self._hidden_dim
        h0 = torch.zeros(B * T, N * D, device=inputs.device)

        layer_embeds = []

        # TGCN 1
        out1, h1 = self.tgcn_cell1(inputs, h0)
        z1 = out1.view(B, T, N, D)
        layer_embeds.append(z1)

        x2nn = z1[..., 0].contiguous().view(B * T, N)  # [B*T, N]

        # TGCN 2
        out2, h2 = self.tgcn_cell2(x2nn, h1)
        out2 = out2.view(B, T, N, D)
        layer_embeds.append(out2)

        # AGG
        spatial_feats = torch.max(
            torch.stack(layer_embeds, dim=0),
            dim=0
        ).values
        spatial_last = spatial_feats[:, -1]  # [B, N, D]

        return spatial_last

class TimeKANblock(nn.Module):
    def __init__(self,d_model,seq_len,order):
        super().__init__()
        self.channel_mixer = nn.Sequential(
            ChebyKANLayer(d_model, d_model,order)
        )
        self.conv = BasicConv(d_model,d_model,kernel_size=3,degree=order,groups=d_model)
    def forward(self,x):
        x1 = self.channel_mixer(x)
        x2 = self.conv(x)
        out  = x1 + x2
        return out

class ORGF(nn.Module):
    def __init__(self, feature_dim: int,
                 rank_ratio: int = 16,
                 max_gamma: float = 0.2,
                 warmup_steps: int = 400,
                 eps: float = 1e-6):
        super().__init__()
        D = feature_dim
        r = max(4, D // rank_ratio)

        self.eps = eps
        self.max_gamma = float(max_gamma)
        self.warmup_steps = max(1, int(warmup_steps))

        # baseline attention
        self.attn = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.GELU(),
            nn.Linear(D, 2),
            nn.Softmax(dim=-1),
        )

        # residual branch (orth + low-rank)
        self.norm_s = nn.LayerNorm(D)
        self.norm_t = nn.LayerNorm(D)
        self.orth_proj = nn.Linear(D, r, bias=False)
        self.orth_out = nn.Linear(r, D, bias=False)

        # simple confidence + scale
        self.conf = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))
        self.delta = nn.Parameter(torch.tensor(0.5))

        self.register_buffer("_updates", torch.zeros((), dtype=torch.long))

        self._init_params()

    def _init_params(self):
        # attention normal init
        for m in self.attn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # residual path zero init: start == pure attention
        nn.init.zeros_(self.orth_proj.weight)
        nn.init.zeros_(self.orth_out.weight)
        nn.init.zeros_(self.conf[1].weight)
        nn.init.zeros_(self.conf[1].bias)

    def _proj(self, u, v):
        # project v onto u: (v·u)/(u·u+eps) * u
        coef = (v * u).sum(dim=-1, keepdim=True) / ((u * u).sum(dim=-1, keepdim=True) + self.eps)
        coef = coef.clamp(-10.0, 10.0)  # 防极端梯度
        return coef * u

    def _gamma(self, ref_tensor: torch.Tensor):
        if self.training:
            self._updates += 1
            u = min(1.0, float(self._updates.item()) / float(self.warmup_steps))
        else:
            u = 1.0
        g = self.max_gamma * 0.5 * (1.0 - math.cos(math.pi * u))
        return ref_tensor.new_tensor(g)

    def forward(self, spatial_feat, temporal_feat):
        s, t = spatial_feat, temporal_feat  # [..., D]

        # baseline attention fusion
        w = self.attn(torch.cat([s, t], dim=-1))  # [..., 2]
        Zbase = w[..., :1] * s + w[..., 1:] * t

        # orth residual
        sn, tn = self.norm_s(s), self.norm_t(t)
        orth = (tn - self._proj(sn, tn)) + (sn - self._proj(tn, sn))
        resid = self.orth_out(self.orth_proj(orth))

        # lightweight safety
        resid = torch.where(torch.isfinite(resid), resid, torch.zeros_like(resid))
        conf = torch.sigmoid(self.conf(resid))           # [..., 1]

        Zres = conf * resid
        out = Zbase + Zres
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out

def _same_pad(k, d):
    return ((k - 1) * d) // 2

class MultiscaleDConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=(1, 3, 5),
                 dropout=0.0):
        super().__init__()
        self.dilation_rates = dilation_rates

        def branch(kernel, dilation, padding):
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel,
                          dilation=dilation, padding=padding, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.conv1 = branch(1, 1, 0)
        self.conv3 = branch(3, dilation_rates[1], _same_pad(3, dilation_rates[1]))
        self.conv5 = branch(5, dilation_rates[2], _same_pad(5, dilation_rates[2]))

    def forward(self, x):
        # x: [B, N, F, T]
        B, N, F, T = x.shape
        x = x.view(B * N, F, T)

        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)

        D = x1.size(1)
        x1 = x1.view(B, N, D, T)
        x3 = x3.view(B, N, D, T)
        x5 = x5.view(B, N, D, T)
        return x1, x3, x5

class EngergyGated(nn.Module):
    def __init__(self, tau=0.9, energy="l2"):
        super().__init__()
        self.tau = tau
        self.energy = energy.lower()

    def _energy(self, x):
        # x: [B, N, D, T] → [B, N, 1, T]
        if self.energy == "l2":
            return (x ** 2).mean(dim=2, keepdim=True)
        elif self.energy == "l1":
            return x.abs().mean(dim=2, keepdim=True)
        else:  # max
            return x.abs().amax(dim=2, keepdim=True)

    def forward(self, x1, x3, x5):
        # Energy maps
        e1 = self._energy(x1)
        e3 = self._energy(x3)
        e5 = self._energy(x5)

        # Stack on branch dimension
        logits = torch.cat([e1, e3, e5], dim=2)  # [B, N, 3, T]
        # Centering on branch dimension (stable)
        logits = logits - logits.mean(dim=2, keepdim=True)

        # Softmax over branches
        w = torch.softmax(logits / self.tau, dim=2)  # [B, N, 3, T]

        return w

class WeightedSum(nn.Module):

    def __init__(self, check_shapes: bool = True):
        super().__init__()

    def forward(self, w: torch.Tensor, x1: torch.Tensor, x3: torch.Tensor, x5: torch.Tensor) -> torch.Tensor:

        y = (
            w[:, :, 0:1, :] * x1 +
            w[:, :, 1:2, :] * x3 +
            w[:, :, 2:3, :] * x5
        )
        return y

class BiLSTMXL(nn.Module):
    def __init__(self, hidden_dim, pre_len):
        super(BiLSTMXL, self).__init__()
        self.BiLSTM = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=pre_len,  # Using pre_len for the number of layers
            bidirectional=True,  # Bidirectional LSTM
            batch_first=True  # Input and output tensors are in (batch, seq, feature) format
        )

    def forward(self, x):
        B, N, F, T = x.shape

        # Permute and reshape inside the class
        x = x.permute(0, 1, 3, 2).contiguous()  # [B, N, F, T] -> [B, N, T, D]
        x = x.view(B * N, T, F)  # Reshape to [B * N, T, D] (F is the feature dimension here)

        # BiLSTM processing
        w_out, _ = self.BiLSTM(x)  # BiLSTM output
        w_out = w_out[:, -1, :]  # Take the last time step's output
        w_out = w_out.view(B, N, -1)  # Reshape back to [B, N, 2D]

        return w_out

class _Scale(nn.Module):
    def __init__(self, s: float): super().__init__(); self.s = float(s)
    def forward(self, x): return self.s * x

class Encoder(nn.Module):
    def __init__(self, adj, input_dim, hidden_dim, seq_len, pre_len):
        super(Encoder, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.MultiscaleDConv = MultiscaleDConv(in_channels=5, out_channels=hidden_dim)
        self.EngergyGated = EngergyGated(
            tau=0.9,
            energy="l2"
        )
        self.BiLSTMXL = BiLSTMXL(hidden_dim, pre_len)
        self.hpool = nn.AvgPool1d(kernel_size=1, stride=2, padding=0)
        self.TimeKANblock = TimeKANblock(
            d_model=hidden_dim,
            seq_len=seq_len,
            order=3
        )
        self.ORGF = ORGF(
            feature_dim=hidden_dim,
            rank_ratio=16,
            max_gamma=0.2, #0.2
            warmup_steps=400 #400
        )
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim) if hidden_dim * 2 != hidden_dim else nn.Identity()
        self.ablate_fusion = getattr(self, "ablate_fusion", True)
        N = self.adj.size(0)
        self.in_norm = nn.LayerNorm(N)
        self.tgcn_cell1 = TGCNCell(self.adj, self._input_dim, self._hidden_dim)
        self.tgcn_cell2 = TGCNCell(self.adj, self._input_dim, self._hidden_dim)
        self.linear = nn.Linear(1, 1)
        self.weighted_sum = WeightedSum()
        self.embedding = Embedding(self.linear, self.in_norm)
        self.dual_tgcn_agg = Dual_layer_T_GCN_Aggregation_block(
            tgcn_cell1=self.tgcn_cell1,
            tgcn_cell2=self.tgcn_cell2,
            hidden_dim=self._hidden_dim
        )

    def forward(self, case, combine):

        B, T, N = case.shape
        # ============================================================
        # AF-SB (Spatial branch)
        # ============================================================
        # Embedding
        inputs = self.embedding(case)  # [B*T, N]

        # Dual-layer_T-GCN_Aggregation_block
        spatial_last = self.dual_tgcn_agg(inputs, B=B, T=T, N=N)  # [B, N, D]

        # TimeKAN block
        temporal_last = self.TimeKANblock(spatial_last)  # [B, N, D]

        # ORGF
        output = self.ORGF(spatial_last, temporal_last) # [B, N, D]

        # ============================================================
        # EGM-TB (Temporal branch)
        # ============================================================
        x3 = combine.permute(0, 2, 3, 1).contiguous() # [B, N, F, T]

        # Multi-scale D-Conv
        e1,e2,e3 = self.MultiscaleDConv(x3)  # [B, N, D, T]

        # Energy-Gated
        w = self.EngergyGated(e1, e2, e3)

        # Weighted Sum
        WS = self.weighted_sum(w, e1, e2, e3)

        # BiLSTM × L
        w_out = self.BiLSTMXL(WS)

        # AvgPool1d
        J = self.hpool(w_out)  # [B, N, D]

        return J + output

class Decoder(nn.Module):
    def __init__(self, hidden_dim, pre_len):
        super(Decoder, self).__init__()
        self.Predictors = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, 1) for i in range(pre_len)])
        self.Decoders = nn.ModuleList(
            [nn.LSTM(
                hidden_dim if i == 0 else hidden_dim * 2 + 1,
                hidden_dim,
                num_layers=1,
                bidirectional=True,
            ) for i in range(pre_len)]
        )

    def forward(self, w_out):
        batch_size, num_nodes, _ = w_out.shape
        y = torch.zeros(batch_size, num_nodes, 1).to(w_out.device)
        y_list = torch.zeros(batch_size, num_nodes, 1).to(w_out.device)
        for i, predictor in enumerate(self.Predictors):
            dh = w_out if i == 0 else torch.cat((do, y), dim=2)
            do, dh = self.Decoders[i](dh)
            y = predictor(do)
            y_list = y if i == 0 else torch.cat((y_list, y), dim=2)
        return torch.tanh(y_list)

class STGMixer(nn.Module):
    def __init__(self, adj, hidden_dim, pre_len, seq_len: int, num_heads=8, **kwargs):
        super(STGMixer, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self._adj = adj
        self.pre_len = pre_len
        self.encoder = Encoder(self.adj, self._input_dim, self._hidden_dim, seq_len,pre_len)
        self.decoder = Decoder(self._hidden_dim, pre_len)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes, _ = inputs.shape
        split_tensors = torch.split(inputs, [1, 5], dim=3)
        case = split_tensors[0].reshape((batch_size, seq_len, num_nodes))
        combine = split_tensors[1]
        assert self._input_dim == num_nodes
        w_out = self.encoder(case, combine)
        y_list = self.decoder(w_out)
        return y_list


    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=128)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
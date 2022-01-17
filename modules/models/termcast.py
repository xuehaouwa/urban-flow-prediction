from torch import nn
import math
import torch


class T(nn.Module):
    def __init__(self, cfg_params):
        super(T, self).__init__()
        cfg_params.copyAttrib(self)
        self.num = self.grid_c * self.grid_h * self.grid_w
        self.close = resunit(self.grid_c * self.obs_len, self.grid_c)

        self.w1 = nn.Parameter(-torch.ones((2, self.grid_h, self.grid_w), requires_grad=True))
        self.w2 = nn.Parameter(-torch.ones((2, self.grid_h, self.grid_w), requires_grad=True))
        self.w3 = nn.Parameter(-torch.ones((2, self.grid_h, self.grid_w), requires_grad=True))

        self.transformer = TransformerBlock(input_seq_len=self.obs_len, hidden_dim=self.hidden_dim)
        self.relation_layer = nn.Sequential(
            nn.Linear(3 * self.num, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.ext = nn.Sequential(
            nn.Linear(in_features=self.interval+7, out_features=self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_out),
            nn.Linear(in_features=self.hidden_dim, out_features=self.num))

        self.relation_predict = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_out),
            nn.Linear(in_features=self.hidden_dim, out_features=self.num))

    def encode_relation(self, x_c, x_p, x_t):
        relations = []  # need to be Batch_size x len X hidden_dim

        for i in range(self.obs_len):
            relation_in = torch.cat((x_c[:, i, :, :, :].view(-1, self.num), x_p[:, i, :, :, :].view(-1, self.num),
                                     x_t[:, i, :, :, :].view(-1, self.num)), dim=1)
            relations.append(self.relation_layer(relation_in))

        out = torch.stack(relations, dim=1)

        return out

    def forward(self, close_data, period_data, trend_data, extra):
        extra_p = self.ext(extra).view(-1, self.grid_c, self.grid_h, self.grid_w)
        close_out = self.close(close_data.view(-1, self.grid_c * self.obs_len, self.grid_h, self.grid_w))
        observed_relations = self.encode_relation(close_data, period_data, trend_data)

        predict_relation = self.transformer(observed_relations)
        relation_out = self.relation_predict(predict_relation).view(-1, self.grid_c, self.grid_h, self.grid_w)

        main_out = torch.mul(close_out, self.w1) + torch.mul(relation_out, self.w2) + torch.mul(extra_p, self.w3)

        infer_relation = self.relation_layer(torch.cat((main_out.view(-1, self.num),
                                                        period_data[:, -1, :, :, :].view(-1, self.num),
                                                        trend_data[:, -1, :, :, :].view(-1, self.num)), dim=1))

        return main_out, predict_relation, infer_relation


class TransformerBlock(nn.Module):
    def __init__(self, input_seq_len, hidden_dim=256, nheads=8, dropout=0.2,
                 num_encoder_layers=6, num_decoder_layers=6):
        super(TransformerBlock, self).__init__()

        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.query_pos = nn.Parameter(torch.rand(1, hidden_dim))
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len=input_seq_len)

    def forward(self, x):
        """

        :param x:  Batch_size x len X hidden_dim
        :return: Batch_size x hidden_dim
        """
        batch_size = x.size()[0]
        h = x.permute(1, 0, 2)
        # sequence to the encoder
        transformer_input = self.pos_encoder(h)
        transformer_out = self.transformer(transformer_input, self.query_pos.unsqueeze(1).repeat(1, batch_size, 1))

        return transformer_out.squeeze(0)


class resunit(nn.Module):
    def __init__(self, in_flow, out_flow):
        super(resunit, self).__init__()
        self.unit = nn.Sequential(
            nn.Conv2d(in_flow, 32, kernel_size=3, stride=1, padding=1, bias=False),
            residual(),
            residual(),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_flow, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.unit(x)


class residual(nn.Module):
    def __init__(self, in_flow=32, out_flow=32):
        super(residual, self).__init__()
        self.left = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_flow, out_flow, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.left(x)
        out = self.left(out)
        out = out + x
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

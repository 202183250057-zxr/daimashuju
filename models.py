import torch
import torch.nn as nn
import math


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, kernel_size, lstm_hidden_dim, output_dim):
        super(CNNLSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_out_channels, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output


class MultiScaleMultiTaskLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim_short, hidden_dim_long, fusion_dim, task_hidden_dim, num_tasks):
        super(MultiScaleMultiTaskLSTM, self).__init__()
        # 多尺度LSTM层
        self.short_lstm = nn.LSTM(input_dim, hidden_dim_short, batch_first=True)
        self.long_lstm = nn.LSTM(input_dim, hidden_dim_long, batch_first=True)

        # 融合层
        self.fusion = nn.Linear(hidden_dim_short + hidden_dim_long, fusion_dim)
        self.relu = nn.ReLU()

        # 3. 多任务学习层 (每个任务有独立的输出头)
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, task_hidden_dim),
                nn.ReLU(),
                nn.Linear(task_hidden_dim, 1)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x_short, x_long):
        # x_short, x_long 来自多通道输入层（数据预处理阶段）
        _, (h_short, _) = self.short_lstm(x_short)
        _, (h_long, _) = self.long_lstm(x_long)

        # 融合LSTM的最后隐藏状态
        combined_feat = torch.cat([h_short.squeeze(0), h_long.squeeze(0)], dim=1)
        fused = self.relu(self.fusion(combined_feat))

        # 通过各自的头进行预测
        outputs = [head(fused) for head in self.task_heads]

        return outputs
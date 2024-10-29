import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=128):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.lstm1 = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )

        self.lstm2 = nn.LSTM(
          input_size=self.hidden_dim,
          hidden_size=embedding_dim,
          num_layers=2,
          dropout=0.5,
          batch_first=True
        )

    def forward(self, x):
        #print('enc:', x.shape)

        x, (hidden_n, cell_n) = self.lstm1(x)
        x, (hidden_n, cell_n) = self.lstm2(x)

        return x

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=128, output_dim=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, output_dim

        self.lstm1 = nn.LSTM(
          input_size=input_dim,
          hidden_size=input_dim,
          num_layers=2,
          dropout=0.5,
          batch_first=True
        )

        self.lstm2 = nn.LSTM(
          input_size=input_dim,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )
        self.dense_layers = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        #print('dec1:', x.shape)
        x, (hidden_n, cell_n) = self.lstm1(x)
        x, (hidden_n, cell_n) = self.lstm2(x)
        x = x[:,-1,:]
        #print('dec2:', x.shape)

        return self.dense_layers(x)
    
class LSTMAutoencoder(nn.Module):
    def __init__(self, device, seq_len, n_features, embedding_dim=128, output_dim=1):
        super(LSTMAutoencoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, output_dim).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.unsqueeze(-1)
        # print('LSTMAE: ', x.shape)

        return x
import torch
import torch.nn as nn


def get_network(name):
    mapping = {"FF": net_1, "RNN": net_2, "LSTM": net_3, "Trans": net_4}
    return mapping[name]


class net_1(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(net_1, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc_block1 = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU()
        )
        self.fc_block2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.output_block = nn.Sequential(
            nn.Linear(16, num_class),
            nn.Sigmoid()
        )
    
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        outs = self.fc_block1(embedded)
        outs = self.fc_block2(outs)
        outs = self.output_block(outs)
        return outs


class net_2(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(net_2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.rnn = nn.RNN(embed_dim, num_class, 2, batch_first = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, offsets):
        offsets = torch.cat((offsets,torch.tensor([len(text)]).to("cuda")), 0).tolist()
        offsets = [offsets[i] - offsets[i-1] for i in range(1, len(offsets))]
        text = torch.split(text, offsets)
        text = torch.nn.utils.rnn.pad_sequence([p for p in text], batch_first=True)
        embedded = self.embedding(text)
        outs, _ = self.rnn(embedded)
        outs = outs[:,-1,:]
        outs = self.sigmoid(outs)
        return outs


class net_3(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(net_3, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.lstm = nn.LSTM(embed_dim, num_class, 2, batch_first = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, offsets):
        offsets = torch.cat((offsets,torch.tensor([len(text)]).to("cuda")), 0).tolist()
        offsets = [offsets[i] - offsets[i-1] for i in range(1, len(offsets))]
        text = torch.split(text, offsets)
        text = torch.nn.utils.rnn.pad_sequence([p for p in text], batch_first=True)
        embedded = self.embedding(text)
        outs, _ = self.lstm(embedded)
        outs = outs[:,-1,:]
        outs = self.sigmoid(outs)
        return outs


class net_4(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(net_4, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.output_block = nn.Sequential(
            nn.Linear(embed_dim, num_class),
            nn.Sigmoid()
        )

    def forward(self, text, offsets):
        offsets = torch.cat((offsets,torch.tensor([len(text)]).to("cuda")), 0).tolist()
        offsets = [offsets[i] - offsets[i-1] for i in range(1, len(offsets))]
        text = torch.split(text, offsets)
        text = torch.nn.utils.rnn.pad_sequence([p for p in text], batch_first=True)
        embedded = self.embedding(text)
        
        keys = self.k(embedded)
        values = self.v(embedded)
        querys = self.q(embedded)
        
        scores = self.get_scores(querys, keys)
        outs = torch.matmul(scores.to('cuda'), values) + embedded
        outs = nn.functional.normalize(outs, dim=2)

        outs = outs + self.fc(outs)
        outs = nn.functional.normalize(outs, dim=2)
        outs = outs[:, -1, :]

        outs = self.output_block(outs)

        return outs
    
    def get_scores(self, query, keys):
        scores = []
        for q, k in zip(query, keys):
            mat = torch.matmul(q, k.T).tolist()
            for i in range(len(mat)):
                for j in range(len(mat)):
                    if j>i:
                        mat[i][j] = -float("inf")
            scores.append(mat)
        scores = torch.tensor(scores)/query.shape[2]**0.5
        scores = nn.functional.softmax(scores, 2)
        return scores


if __name__ == "__main__":

    vocab_size = 86663
    embed_dim = 128
    num_class = 6
    model = net_4(vocab_size, embed_dim, num_class)

    trial_text = torch.tensor([  816,   174,     9,     8,    60,  1809,  1555,     4,  3291,    80,
            3,    35,  3201, 14391,    22,  2049,     5,    46,     4,    29,
          120,    16,    58,   302,    31,     6,    14,     8,    10,     7,
        10406,     8,    11,  4248, 11671,  3113,  2313,  7852,  2368,  1054,
         2736, 11671, 10373,   383,  1549,   276,     5,   179,     2,    40,
          122,     6,    13,    21,   308,     2,    14,   217,   318,    23,
            6,   644,     9,     6,   114,   111,   277,    78,   364,   111,
          495,   196,    37,    15,    29,    44,    28,    26,   183,    15,
           21,    44,    28,     5,   196,    21,   204,    42,    47,   524,
            2,   460,    21,   108,    15,    44,   109,    31,   197,   456,
          751,    26,    31,  1408,    23,  1031,    14,    33,   722,   844,
           21,   598,     5,     1,   408,    65,    47,    20,    21,   210,
            2,   342,  1692,    12,     1,    74,   463,   865,    22,    21,
          120,   495,    17,    60,   569,   209,     2,  2881,    21,  1721,
          420,   115,  1353,  3428,  1682,  2172,  1054,  3183,  1937,  1583,
          364,   439,     7,   894,   171,   584,   843,   364,  6603,    71,
            2,    74,     7,    28,  1328,     5,    81,   319,   424,    21,
          110,    25,  5807,  1138,     2,    40,    25,  1024,  2572,   364,
          111,  1652,   535,   244,  2584,   121,     2,   196,   244,    26,
           94,   211,   288,   588,   666,   111,  1997,     7,   216, 12072,
           28,     2,   196,   244,    15,   111,  2563,     7,  3171, 18385,
          111,  1495,    28,   476,     5,   353,   480,   143,     4,   298,
           49,   268,   357,  1815,   253,   105,  1374,   105,    41,    40,
            8,    10,  1212,     4,   752,   999,   164,     4,   476,  2637,
          614,   891,  2913,  2935,   214,    13,  1246,   168,   147,    81,
          214,   656,   209,  1065,   195,   164,     4,   353,   107,   214,
          693,     4,   462,   441,  2659,  1682,  2172,  1054,  3183,  1937,
         1583,  1353,  2313,  3792,  3394,  1937,     6,  1244,     4,  2011,
            5,  2681, 82927,    17,  3311,     1,   128, 13743,    73,     3,
           16,   254,   984,     5,     6,    16,     1,  6721,     2,   150,
           37,     2,  2291,  2108,    15,     6,     3,    16,  2518,    15,
           86,    12,    29,   742,   193, 12302,    98,   652,   211,    36,
            1,   848,     5,  3698,     4,     1,  4585,     1, 61942,    32,
           14,   807,    17,    98,  2141,     5,  1493,     1, 26152,  3382,
          486,     1,  3382,   146,  3689,     5,   760,     1, 14681,  1497,
            4,     1, 27644,    17, 16464,     4,     7,  2009,    55,    38,
          324,     1,   648,     9,    14,     8,    59,     4,    30, 75943,
            2,     1,   423,     4, 15196,   724,     1, 35746,    10,    77,
        24888,     1,  4166,    32,     1,  4084,     4,   337,     5,   112,
           11,    65, 24888,     1,  1420,    15,     1,   404,     8,     1,
          865, 14052,    22,     7,  7563,  1771,    13, 16264,     1,  3710,
            4,     1, 15443,     8,  1095,     5,     2,     1,   149,    11,
            8,    52,    48,    17,  1306,  1095,   929,     1,   534,    92,
          222,   983,    86,     8,     1,  4079,  1785,     4,  1020,     5,
         5896,   174,     3,    16,     7,   859,    61,   259,   246,     4,
            0,    12,    53,   744,   184,    24,    65,   664,     9,   185,
          292,    24,   142,     2,    18,  3936,     2,    18,  1290,   184,
            8,    34,     1,    85,   320,    14,     8,  1633,    31,    14,
        25582,     5,    84,  6711,    67,   724,     5,    14,     8,    75,
            3,    16,   173,     0,   292,    19,     7,   601,  1098,     5,
            8,   509,   126,   335,    27,   300,   512,    12,     7, 25582,
           43,  2873,   724,    27,    14,     8,     7, 73899,   846,     3,
           64,     9,    29,   120,    16,   803,    58,   302,  5785,  1334,
            4,     0,  1671,   533,     3,    16,    53,  4527,    78,     1,
         3166,     4,     1,  1104,     5,  1814,    15, 21486,     2,    94,
          182,     1,   533,     8,   906,   335,   165,   580,    53,  2589,
            3,    70,    14,    45,    58,  4152,     6])
    trial_offset = torch.tensor([  0,  33, 276, 306, 316, 441, 519, 529])

    preds = model.forward(text= trial_text, offsets= trial_offset)

    print(preds.shape)
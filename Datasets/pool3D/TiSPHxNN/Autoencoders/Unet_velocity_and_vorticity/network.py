import torch.nn as nn

leakiness = 0.01

input_res = 256
feature_vector_size = 256

L0 = 3
L1 = 8
L2 = L1 * 2
L3 = L2 * 2
L4 = L3 * 2
L5 = L4 * 2
L6 = L5 * 2

LM=L6
feature_size = input_res // 2**6



class ConvAutoencoder(nn.Module):
    
    def __init__(self, type='train'):
        super(ConvAutoencoder, self).__init__()

        if type == 'train':
            dropout_probability = 0.2
        elif type == 'test':
            dropout_probability = 0.0

        self.feature_maps = {}

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(L0, L1, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(L1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.Conv2d(L1, L2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(L2),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.Conv2d(L2, L3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(L3),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.Conv2d(L3, L4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(L4),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.Conv2d(L4, L5, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(L5),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.Conv2d(L5, L6, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(L6),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),
        )

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(LM * feature_size * feature_size, feature_vector_size), 
            nn.BatchNorm1d(feature_vector_size),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),
            nn.Linear(feature_vector_size, LM * feature_size * feature_size),
            nn.BatchNorm1d(LM * feature_size * feature_size),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),
            nn.Unflatten(1, (LM, feature_size, feature_size)),
        )

        # Decoder
        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(L6, L5, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(L5),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.ConvTranspose2d(L5, L4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(L4),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.ConvTranspose2d(L4, L3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(L3),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.ConvTranspose2d(L3, L2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(L2),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.ConvTranspose2d(L2, L1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(L1),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

            nn.ConvTranspose2d(L1, L0, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, input, strategy='whole'):
        if strategy == 'whole':
            x = self.encoder(input)
            x = self.bottleneck(x)
            x = self.decoder(x)
            
        if strategy == 'skip_bottleneck':
            x = self.encoder(input)
            x = self.decoder(x)
        
        return x
        
    def get_activation(self, name):
        def hook(model, input, output):
            self.feature_maps[name] = output.detach()
        return hook
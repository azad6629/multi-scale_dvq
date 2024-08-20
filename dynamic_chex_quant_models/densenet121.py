import torch
import torch.nn as nn
from .quantise import VectorQuantizer
import torch.nn.functional as F
from torchvision.models import densenet121

def normalize_entropy(entropy, max_possible_entropy):
    return entropy / max_possible_entropy
def calculate_entropy(soft_codes):
    entropies = []
    for matrix in soft_codes:
        flat = matrix.flatten()
        probabilities = torch.histc(flat, bins=256, min=0, max=1) / flat.numel()
        entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10))
        entropies.append(entropy)
    return entropies


class dynamic_dense121(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Configuration parameters
        z_channels = 128
        ne = config['vqconfig']['codebook_size']
        ed = config['vqconfig']['codebook_dim']
        cc = config['vqconfig']['commitment_beta']
        
        # Load pretrained DenseNet model
        densenet = densenet121(pretrained=True)
        densenet_layers = list(densenet.features.children())
        
        # Extract layers for different granularity levels
        self.layer1 = nn.Sequential(*densenet_layers[:6])    
        self.layer2 = nn.Sequential(*densenet_layers[6:8])   
        self.layer3 = nn.Sequential(*densenet_layers[8:10])  
        self.layer4 = nn.Sequential(*densenet_layers[10:])   
                
        self.quantize_coarse = VectorQuantizer(ne, ed, cc)
        self.quantize_median = VectorQuantizer(ne, ed, cc)
        self.quantize_fine   = VectorQuantizer(ne, ed, cc)
        
        #upsampling
        self.conv_out_coarse = nn.Conv2d(512, z_channels, kernel_size=3, stride=1, padding=1)
        self.conv_out_median = nn.Conv2d(256,  z_channels, kernel_size=3, stride=1, padding=1)
                
        self.gate_median_pool = nn.AvgPool2d(2, 2)
        self.gate_fine_pool   = nn.AvgPool2d(4, 4)

        self.classifier = nn.Sequential(nn.Linear(1024, 15),nn.Sigmoid())
        
    def forward(self, x):
        bs = x.shape[0]
        # Extract features from different layers
        h_fine    = self.layer1(x)          
        h_median  = self.layer2(h_fine)    
        h_coarse  = self.layer3(h_median)   
        ftrs      = self.layer4(h_coarse)   
        
        
        h_coarse = self.conv_out_coarse(h_coarse)
        h_median = self.conv_out_median(h_median)
        
        quantized_coarse, loss_coarse, _, _, soft_codes_coarse = self.quantize_coarse(h_coarse)
        quantized_median, loss_median, _, _, soft_codes_median = self.quantize_median(h_median)
        quantized_fine,   loss_fine,   _, _, soft_codes_fine = self.quantize_fine(h_fine)
        
        soft_codes = [soft_codes_coarse, soft_codes_median, soft_codes_fine]
        entropies  = calculate_entropy(soft_codes)
        
        entropy_coarse = entropies[0]
        entropy_median = entropies[1]
        entropy_fine   = entropies[2]
        
        entropies = torch.tensor([entropy_coarse, entropy_median, entropy_fine])
        weights = F.softmax(entropies, dim=0)
        quant = (
                    weights[0] * quantized_coarse +
                    weights[1] * self.gate_median_pool(quantized_median) +
                    weights[2] * self.gate_fine_pool(quantized_fine)
                )        
        
        quant = F.adaptive_avg_pool2d(quant, (1, 1)).reshape(bs, -1)
        cont  = F.adaptive_avg_pool2d(ftrs, (1, 1)).reshape(bs, -1)
        padding_size = cont.size(1) - quant.size(1)        
        quant_pad = F.pad(quant, (0, padding_size), 'constant',0)
        
        combined_features = torch.add(cont, quant_pad)
        output = self.classifier(combined_features)
        loss = loss_coarse+loss_median+loss_fine
        return output, loss
    
'''
Pix2pix model from
https://github.com/eriklindernoren/PyTorch-GAN/blob/36d3c77e5ff20ebe0aeefd322326a134a279b93e/implementations/pix2pix/models.py
'''

import torch.nn as nn
import torch.nn.functional as F
import torch


##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        if (x.shape[-2] != skip_input.shape[-2]) or (x.shape[-1] != skip_input.shape[-1]):
            x = nn.functional.interpolate(x, size=skip_input.shape[-2:])
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNetSmall(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, size_factor=4):
        super(GeneratorUNetSmall, self).__init__()

        self.down1 = UNetDown(in_channels, size_factor*16, normalize=False)
        self.down2 = UNetDown(size_factor*16, size_factor*32)
        self.down3 = UNetDown(size_factor*32, size_factor*64)
        self.down4 = UNetDown(size_factor*64, size_factor*128, dropout=0.5)
        self.down5 = UNetDown(size_factor*128, size_factor*128, normalize=False, dropout=0.5)

        self.up1 = UNetUp(size_factor*128, size_factor*128, dropout=0.5)
        self.up2 = UNetUp(size_factor*256, size_factor*128)
        self.up3 = UNetUp(size_factor*192, size_factor*32)
        self.up4 = UNetUp(size_factor*64, size_factor*16)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(size_factor*32, out_channels, 4, padding=1)        
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5, d4)   
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        return self.final(u4)


# class GeneratorUNetSmall(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3):
#         super(GeneratorUNetSmall, self).__init__()

#         self.down1 = UNetDown(in_channels, 64, normalize=False)
#         self.down2 = UNetDown(64, 128)
#         self.down3 = UNetDown(128, 256)
#         self.down4 = UNetDown(256, 512, dropout=0.5)
#         self.down5 = UNetDown(512, 512, normalize=False, dropout=0.5)

#         self.up1 = UNetUp(512, 512, dropout=0.5)
#         self.up2 = UNetUp(1024, 512)
#         self.up3 = UNetUp(768, 128)
#         self.up4 = UNetUp(256, 64)

#         self.final = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.Conv2d(128, out_channels, 4, padding=1)        
#         )

#     def forward(self, x):
#         # U-Net generator with skip connections from encoder to decoder
#         d1 = self.down1(x)      # 64
#         d2 = self.down2(d1)     # 128
#         d3 = self.down3(d2)     # 256
#         d4 = self.down4(d3)     # 512
#         d5 = self.down5(d4)     # 512

#         u1 = self.up1(d5, d4)   # 512 + 512
#         u2 = self.up2(u1, d3)   # 512 + 256
#         u3 = self.up3(u2, d2)   # 128 + 128
#         u4 = self.up4(u3, d1)   # 64 + 64
#         return self.final(u4)

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, scale_factor=1):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, scale_factor*32, normalize=False)
        self.down2 = UNetDown(scale_factor*32, scale_factor*64)
        self.down3 = UNetDown(scale_factor*64, scale_factor*128)
        self.down4 = UNetDown(scale_factor*128, scale_factor*256, dropout=0.5)
        self.down5 = UNetDown(scale_factor*256, scale_factor*256, dropout=0.5)
        self.down6 = UNetDown(scale_factor*256, scale_factor*256, dropout=0.5)
        self.down7 = UNetDown(scale_factor*256, scale_factor*256, dropout=0.5)
        self.down8 = UNetDown(scale_factor*256, scale_factor*256, normalize=False, dropout=0.5)

        self.up1 = UNetUp(scale_factor*256, scale_factor*256, dropout=0.5)
        self.up2 = UNetUp(scale_factor*512, scale_factor*256, dropout=0.5)
        self.up3 = UNetUp(scale_factor*512, scale_factor*256, dropout=0.5)
        self.up4 = UNetUp(scale_factor*512, scale_factor*256, dropout=0.5)
        self.up5 = UNetUp(scale_factor*512, scale_factor*128)
        self.up6 = UNetUp(scale_factor*256, scale_factor*64)
        self.up7 = UNetUp(scale_factor*128, scale_factor*32)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(scale_factor*64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)      # in: in_channels, out: 32
        d2 = self.down2(d1)     # in: 32, out: 64
        d3 = self.down3(d2)     # in: 64, out: 128
        d4 = self.down4(d3)     # in: 128, out: 256
        d5 = self.down5(d4)     # in: 256, out: 256
        d6 = self.down6(d5)     # in: 256, out: 256
        d7 = self.down7(d6)     # in: 256, out: 256
        d8 = self.down8(d7)     # in: 256, out: 256
        
        u1 = self.up1(d8, d7)   # in: d8 (256), out: up1 (256) + d7 (256)   
        u2 = self.up2(u1, d6)   # in: u1 (512), out: up2 (256) + d6 (256)   
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)



# -----------------------------------------------------------------------------

def test():
    device = torch.device('cuda')
    args = {
        'in_channels': 3,
        'out_channels': 8    
    }

    model = GeneratorUNetSmall(**args)
    model = model.to(device)
    # model.initialize(init_method='kaiming', verbose=True)

    # forward
    B = 3
    H, W = 333, 456
    x = torch.randn(B, args['in_channels'], H, W).to(device)

    print(torch.cuda.memory_allocated(device)/(1024*1024*1024))

    import torchvision.models as models
    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        feat = model.forward(x)
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    print(torch.cuda.memory_allocated(device)/(1024*1024*1024))

    import ipdb; ipdb.set_trace()

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    test()
import torch
    
class BaseVessels(torch.nn.Module):
    def __init__(self, inChan, outChan):
        super(BaseVessels, self).__init__()
        self.inChan = inChan
        self.outChan = outChan

        self.enco3 = BaseVessels.doubleBlock(self.inChan, 64)
        self.enco2 = BaseVessels.doubleBlock(64, 128)
        self.enco1 = BaseVessels.doubleBlock(128, 256)
        self.enco0 = BaseVessels.doubleBlock(256, 512)

        self.bottleneck = BaseVessels.doubleBlock(512, 1024)

        self.upconv0 = BaseVessels.upconvBlock(1024, 512)
        self.upconv1 = BaseVessels.upconvBlock(512, 256)
        self.upconv2 = BaseVessels.upconvBlock(256, 128)
        self.upconv3 = BaseVessels.upconvBlock(128, 64)

        self.deco0 = BaseVessels.doubleBlock(1024, 512)
        self.deco1 = BaseVessels.doubleBlock(512, 256)
        self.deco2 = BaseVessels.doubleBlock(256, 128)
        self.deco3 = BaseVessels.doubleBlock(128, 64)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.threshold = torch.nn.Threshold(0.5, 0)
        self.output = torch.nn.Conv2d(64, self.outChan, kernel_size=1)

    def forward(self, patch):
        enco3 = self.enco3(patch)
        enco2 = self.enco2(self.pool(enco3))
        enco1 = self.enco1(self.pool(enco2))
        enco0 = self.enco0(self.pool(enco1))

        bottleneck = self.bottleneck(self.pool(enco0))

        tmp = self.upconv0(bottleneck)
        deco0 = self.deco0(torch.cat((tmp, enco0), dim=1))
        tmp = self.upconv1(deco0)
        deco1 = self.deco1(torch.cat((tmp, enco1), dim=1))
        tmp = self.upconv2(deco1)
        deco2 = self.deco2(torch.cat((tmp, enco2), dim=1))
        tmp = self.upconv3(deco2)
        deco3 = self.deco3(torch.cat((tmp, enco3), dim=1))

        return self.threshold(torch.sigmoid(self.output(deco3)))

    def doubleBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.Conv2d(inChan, outChan, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(outChan),
        torch.nn.ReLU(),
        torch.nn.Conv2d(outChan, outChan, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(outChan),
        torch.nn.ReLU())

    def upconvBlock(inChan, outChan):
        return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inChan, outChan, kernel_size=2, stride=2))

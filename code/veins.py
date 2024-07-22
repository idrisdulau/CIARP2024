import torch

class BaseVeins(torch.nn.Module):
    def __init__(self, inChan, outChan):
        super(BaseVeins, self).__init__()
        self.inChan = inChan
        self.outChan = outChan

        self.enco6 = BaseVeins.doubleBlock(self.inChan, 16)
        self.enco5 = BaseVeins.doubleBlock(16, 32)
        self.enco4 = BaseVeins.doubleBlock(32, 64)
        self.enco3 = BaseVeins.doubleBlock(64, 128)
        self.enco2 = BaseVeins.doubleBlock(128, 256)
        self.enco1 = BaseVeins.doubleBlock(256, 512)
        self.enco0 = BaseVeins.doubleBlock(512, 1024)

        self.bottleneck = BaseVeins.doubleBlock(1024, 2048)

        self.upconv0 = BaseVeins.upconvBlock(2048, 1024)
        self.upconv1 = BaseVeins.upconvBlock(1024, 512)
        self.upconv2 = BaseVeins.upconvBlock(512, 256)
        self.upconv3 = BaseVeins.upconvBlock(256, 128)
        self.upconv4 = BaseVeins.upconvBlock(128, 64)
        self.upconv5 = BaseVeins.upconvBlock(64, 32)
        self.upconv6 = BaseVeins.upconvBlock(32, 16)

        self.deco0 = BaseVeins.doubleBlock(2048, 1024)
        self.deco1 = BaseVeins.doubleBlock(1024, 512)
        self.deco2 = BaseVeins.doubleBlock(512, 256)
        self.deco3 = BaseVeins.doubleBlock(256, 128)
        self.deco4 = BaseVeins.doubleBlock(128, 64)
        self.deco5 = BaseVeins.doubleBlock(64, 32)
        self.deco6 = BaseVeins.doubleBlock(32, 16)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.threshold = torch.nn.Threshold(0.5, 0)
        self.output = torch.nn.Conv2d(16, self.outChan, kernel_size=1)

    def forward(self, patch):
        enco6 = self.enco6(patch)
        enco5 = self.enco5(self.pool(enco6))
        enco4 = self.enco4(self.pool(enco5))
        enco3 = self.enco3(self.pool(enco4))
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
        tmp = self.upconv4(deco3)
        deco4 = self.deco4(torch.cat((tmp, enco4), dim=1))
        tmp = self.upconv5(deco4)
        deco5 = self.deco5(torch.cat((tmp, enco5), dim=1))
        tmp = self.upconv6(deco5)
        deco6 = self.deco6(torch.cat((tmp, enco6), dim=1))

        return self.threshold(torch.sigmoid(self.output(deco6)))

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
        torch.nn.Conv2d(inChan, outChan, kernel_size=1, padding=0),
        torch.nn.Upsample(scale_factor=2, mode="bicubic"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogleNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)  # output = [batch, 64, 112, 112]
        # maxpool2d默认向下取整，如果不开启ceil_mode=True，此时的输出为[batch, 64, 55, 55]
        self.maxPool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # output = [batch, 64, 56, 56]

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)  # output = [batch, 192, 56, 56]
        self.maxPool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # output = [batch, 192, 28, 28]

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)  # output = [batch, 256, 28, 28]
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)  # output = [batch, 480, 28, 28]
        self.maxPool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # output = [batch, 480, 14, 14]

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)  # output = [batch, 512, 14, 14]
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)  # output = [batch, 512, 14, 14]
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)  # output = [batch, 512, 14, 14]
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)  # output = [batch, 528, 14, 14]
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)  # output = [batch, 832, 14, 14]
        self.maxPool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # output = [batch, 832, 7, 7]

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)  # output = [batch, 832, 7, 7]
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)  # output = [batch, 1024, 7, 7]

        # self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)  # output = [batch, 1024, 1, 1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)  # output = [batch, 1000, 1, 1]

        if self.aux_logits:
            self.aux_logits1 = InceptionAux(512, num_classes)
            self.aux_logits2 = InceptionAux(528, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # x = [N 3 224 224]
        x = self.conv1(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxPool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxPool3(x)

        x = self.inception4a(x)
        if self.training and self.aux_logits:  # eval model lose this layer
            aux1 = self.aux_logits1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:  # eval model lose this layer
            aux2 = self.aux_logits2(x)
        x = self.inception4e(x)
        x = self.maxPool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.AverPool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output = [batch, 128, 4, 4]
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.AverPool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)  # 1 代表flatten从channel处向后展平
        x = F.dropout(x, 0.5, training=self.training)  # 原论文中dropout参数为0.7
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3reduce, ch3x3,
                 ch5x5reduce, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch_1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3reduce, kernel_size=1),
            BasicConv2d(ch3x3reduce, ch3x3, kernel_size=3, padding=1)
        )
        self.branch_3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5reduce, kernel_size=1),
            BasicConv2d(ch5x5reduce, ch5x5, kernel_size=5, padding=2)
        )
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x4 = self.branch_4(x)
        outputs = [x1, x2, x3, x4]
        return torch.cat(outputs, 1)  # NCHW, 1为在channel通道进行合并


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = GoogleNet().to(device)
#     summary(model, input_size=(3, 224, 224))

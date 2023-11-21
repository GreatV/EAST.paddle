import paddle
from paddle.vision.models import vgg16
import math
from weight_init import weight_init_


class extractor(paddle.nn.Layer):

    def __init__(self, pretrained):
        super(extractor, self).__init__()
        vgg16_bn = vgg16(pretrained=False, batch_norm=True)
        self.features = vgg16_bn.features

    def forward(self, x):
        out = []
        for m in self.features:
            x = m(x)
            if isinstance(m, paddle.nn.MaxPool2D):
                out.append(x)
        return out[1:]


class merge(paddle.nn.Layer):

    def __init__(self):
        super(merge, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1024, out_channels=128,
            kernel_size=1)
        self.bn1 = paddle.nn.BatchNorm2D(num_features=128)
        self.relu1 = paddle.nn.ReLU()
        self.conv2 = paddle.nn.Conv2D(in_channels=128, out_channels=128,
            kernel_size=3, padding=1)
        self.bn2 = paddle.nn.BatchNorm2D(num_features=128)
        self.relu2 = paddle.nn.ReLU()
        self.conv3 = paddle.nn.Conv2D(in_channels=384, out_channels=64,
            kernel_size=1)
        self.bn3 = paddle.nn.BatchNorm2D(num_features=64)
        self.relu3 = paddle.nn.ReLU()
        self.conv4 = paddle.nn.Conv2D(in_channels=64, out_channels=64,
            kernel_size=3, padding=1)
        self.bn4 = paddle.nn.BatchNorm2D(num_features=64)
        self.relu4 = paddle.nn.ReLU()
        self.conv5 = paddle.nn.Conv2D(in_channels=192, out_channels=32,
            kernel_size=1)
        self.bn5 = paddle.nn.BatchNorm2D(num_features=32)
        self.relu5 = paddle.nn.ReLU()
        self.conv6 = paddle.nn.Conv2D(in_channels=32, out_channels=32,
            kernel_size=3, padding=1)
        self.bn6 = paddle.nn.BatchNorm2D(num_features=32)
        self.relu6 = paddle.nn.ReLU()
        self.conv7 = paddle.nn.Conv2D(in_channels=32, out_channels=32,
            kernel_size=3, padding=1)
        self.bn7 = paddle.nn.BatchNorm2D(num_features=32)
        self.relu7 = paddle.nn.ReLU()
        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                weight_init_(m.weight, "kaiming_normal_", mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init_Constant = paddle.nn.initializer.Constant(value=0)
                    init_Constant(m.bias)
            elif isinstance(m, paddle.nn.BatchNorm2D):
                init_Constant = paddle.nn.initializer.Constant(value=1)
                init_Constant(m.weight)
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)

    def forward(self, x):
        y = paddle.nn.functional.interpolate(x=x[3], scale_factor=2, mode=
            'bilinear', align_corners=True)
        y = paddle.concat(x=(y, x[2]), axis=1)
        y = self.relu1(self.bn1(self.conv1(y)))
        y = self.relu2(self.bn2(self.conv2(y)))
        y = paddle.nn.functional.interpolate(x=y, scale_factor=2, mode=
            'bilinear', align_corners=True)
        y = paddle.concat(x=(y, x[1]), axis=1)
        y = self.relu3(self.bn3(self.conv3(y)))
        y = self.relu4(self.bn4(self.conv4(y)))
        y = paddle.nn.functional.interpolate(x=y, scale_factor=2, mode=
            'bilinear', align_corners=True)
        y = paddle.concat(x=(y, x[0]), axis=1)
        y = self.relu5(self.bn5(self.conv5(y)))
        y = self.relu6(self.bn6(self.conv6(y)))
        y = self.relu7(self.bn7(self.conv7(y)))
        return y


class output(paddle.nn.Layer):

    def __init__(self, scope=512):
        super(output, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=32, out_channels=1,
            kernel_size=1)
        self.sigmoid1 = paddle.nn.Sigmoid()
        self.conv2 = paddle.nn.Conv2D(in_channels=32, out_channels=4,
            kernel_size=1)
        self.sigmoid2 = paddle.nn.Sigmoid()
        self.conv3 = paddle.nn.Conv2D(in_channels=32, out_channels=1,
            kernel_size=1)
        self.sigmoid3 = paddle.nn.Sigmoid()
        self.scope = 512
        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                weight_init_(m.weight, "kaiming_normal_", mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init_Constant = paddle.nn.initializer.Constant(value=0)
                    init_Constant(m.bias)

    def forward(self, x):
        score = self.sigmoid1(self.conv1(x))
        loc = self.sigmoid2(self.conv2(x)) * self.scope
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
        geo = paddle.concat(x=(loc, angle), axis=1)
        return score, geo


class EAST(paddle.nn.Layer):

    def __init__(self, pretrained=True):
        super(EAST, self).__init__()
        self.extractor = extractor(pretrained)
        self.merge = merge()
        self.output = output()

    def forward(self, x):
        return self.output(self.merge(self.extractor(x)))


if __name__ == '__main__':
    m = EAST()
    x = paddle.randn(shape=[1, 3, 256, 256])
    score, geo = m(x)
    print(score.shape)
    print(geo.shape)

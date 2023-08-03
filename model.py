import torch
import torch.nn as nn
import torch.nn.functional as F

class MINIVGG(nn.Module):
    def __init__(self, num_classes=100, init_weights=True):
        super(MINIVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = (3, 3), stride = (1, 1)),
            nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False),
            nn.Conv2d(64, 128, kernel_size = (3, 3), stride = (1, 1)),
            nn.BatchNorm2d(128, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False),
            nn.Conv2d(128, 128, kernel_size = (3, 3), stride = (1, 1)),
            nn.BatchNorm2d(128, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            nn.Conv2d(128, 128, kernel_size = (3, 3), stride = (1, 1)),
            nn.BatchNorm2d(128, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Sequential(
            nn.Linear(in_features = 3200, out_features = 512, bias = True),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.3, inplace = False),
            nn.Linear(in_features = 512, out_features = 256, bias = True),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.3, inplace = False),
            nn.Linear(in_features = 256, out_features = 100, bias = True)
        )
        if init_weights:
            self._initialize_weights()
    def forward(self, x):
        x = self.features(x)    
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
vgg = MINIVGG()

class CUSTOM_MINIVGG(nn.Module):
    def __init__(self, num_classes=100, init_weights=True):
        super(CUSTOM_MINIVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = (3, 3), stride = (1, 1)),
            nn.BatchNorm2d(32, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False),
            nn.Conv2d(32, 64, kernel_size = (3, 3), stride = (1, 1)),
            nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False),
            # nn.Conv2d(64, 128, kernel_size = (3, 3), stride = (1, 1)),
            # nn.BatchNorm2d(128, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            # nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size = (3, 3), stride = (1, 1)),
            # nn.BatchNorm2d(128, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.classifier = nn.Sequential(
            # nn.Linear(in_features = 3200, out_features = 512, bias = True),
            # nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.3, inplace = False),
            nn.Linear(in_features = 1600, out_features = 512, bias = True),
            nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.3, inplace = False),
            nn.Linear(in_features = 512, out_features = 256, bias = True),
            nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.3, inplace = False),
            nn.Linear(in_features = 256, out_features = 128, bias = True),
            nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.3, inplace = False),
            nn.Linear(in_features = 128, out_features = 64, bias = True)
        )
        if init_weights:
            self._initialize_weights()
    def forward(self, x):
        x = self.features(x)    
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
custom_vgg = CUSTOM_MINIVGG()





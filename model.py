import torch



class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.module_list = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=3, padding=3),
            torch.nn.SiLU(),
            torch.nn.BatchNorm2d(num_features=8),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, stride=3, padding=3),
            torch.nn.SiLU(),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=3, padding=3),
            torch.nn.SiLU(),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=6, stride=1, padding=0),
            torch.nn.SiLU(),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        ])

    def forward(self, x):
        for module in self.module_list: x = module(x)
        x = x * 2 - 1
        return x
import torch

def circles(h: float, w: float, center: torch.FloatTensor, radii: torch.FloatTensor):
    x = torch.linspace(-1, 1, w).view(1, -1).expand((radii.shape[0], 1, -1))
    y = torch.linspace(-1, 1, h).view(-1, 1).expand((radii.shape[0], -1, 1))

    x_centered_scaled = (x - (center[:, 0].view(-1, 1, 1) / w - 0.5) * 2) / radii.view(-1, 1, 1)
    y_centered_scaled = (y - (center[:, 1].view(-1, 1, 1) / h - 0.5) * 2) / radii.view(-1, 1, 1)
    mask = ((x_centered_scaled ** 2 + y_centered_scaled**2) < 1.0) * 1.0

    return mask

def damage_mask(batch_size: int, width: int, height: int) -> torch.FloatTensor:
    radii = torch.rand(batch_size) * 0.3 + 0.1
    center = torch.rand(batch_size, 2) * torch.tensor([height, width]).view(1, -1)
    mask = circles(h=height, w=width, center=center, radii=radii)
    return 1.0 - mask

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    test = damage_mask(height=100, width=100, batch_size=10) * 255
    for i in range(test.shape[0]):
        plt.imshow(test[i])
        plt.show()
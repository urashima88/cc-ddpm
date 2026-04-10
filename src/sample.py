import torch
import matplotlib.pyplot as plt

class ImgDataTransformer():
    def __init__(self, mean, std, device):
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(3, 1, 1)

    def tensor_to_img(self, tensor):
        return (tensor * self.std + self.mean).clamp(0, 1).mul(255).byte().permute(1, 2, 0).detach().cpu().numpy()
    
def sample_images(model, n_samples, img_size, img_data_transformer, n_classes=None, cls=None, cols=3):
    device = model.device
    if n_classes:
        if cls:
            y = torch.full((n_samples,), cls, dtype=torch.long, device=device)
        else:
            y = torch.randint(0, n_classes, (n_samples,), device=device)
    else:
        y = None

    xts = model.sample(n_samples, shape=img_size, y=y)

    samples = []

    for i in range(xts.shape[0]):
        samples.append(img_data_transformer.tensor_to_img(xts[i]))

    rows = (n_samples // cols) + (0 if n_samples % cols == 0 else 1)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    for i in range(len(samples)):
        row = i // cols
        axes[row, i % cols].imshow(samples[i])
        axes[row, i % cols].grid(False)

    plt.show()
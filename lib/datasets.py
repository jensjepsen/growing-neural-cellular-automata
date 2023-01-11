import torch, PIL.Image, numpy as np

class DataSet(torch.utils.data.Dataset):
    def __init__(self, image_num: int=0, length=100) -> None:
        super().__init__()
        with PIL.Image.open('images/emoji.png') as im:
            im = np.array(im)
        self.image = torch.tensor(im / 255.0, dtype=torch.float32)[:, 0 + 40*image_num:40 * (image_num + 1), :]
        self.length = length
    
    def __getitem__(self, idx):
        return self.image
    
    def __len__(self):
        return self.length


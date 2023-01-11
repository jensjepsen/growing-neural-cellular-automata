import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F, pytorch_lightning as pl, PIL.Image, numpy as np, torch.utils.data, torchvision.utils
import typing

from lib.damage import damage_mask

class Cell(nn.Module):
    """
        Cell state is 16 dim tensor:
            [r, g, b, a, ...]
            where 0:4 carry interpretable values
            and 4:16 are state variables.
            When a == 0         , cell is dead
                    0 < a < 0.1    , cell is growing
                    0.1 < a        , cell is mature
        
        Grid state is a img_height x img_width x state_dim tensor
    """
    def __init__(self,
            cell_state_dimension=16,
            update_rule_hidden_dim=128,
            alpha_alive_threshold=0.1,
            update_probability=0.5,
            input_shape=(40, 40)
        ) -> None:
        super().__init__()

        self.alpha_alive_threshold = alpha_alive_threshold
        self.update_probability = update_probability
        self.cell_state_dimension = cell_state_dimension
        self.input_shape = input_shape

        # A fixed initial state, from which all others will evolve :O
        # we set the alpha and hidden state of cell at the center of the grid to all ones
        #self.initial_state = torch.zeros([*input_shape, cell_state_dimension], dtype=torch.float32)
        
        self.register_buffer('initial_state', torch.zeros([*input_shape, cell_state_dimension], dtype=torch.float32))
        self.initial_state[input_shape[0] // 2, input_shape[1] // 2, 3:] = torch.tensor([1.0] * (cell_state_dimension - 3))
        
        # Define vertical and horizontal sobel filters,
        # that we'll use to calculate the gradient of the grid, later
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ] * 16 * 1, dtype=torch.float32).view(16, 1, 3, 3) / 8)

        self.register_buffer('sobel_y', self.sobel_x.permute(0, 1, 3, 2))

        # Define the learnable update rule for each cell,
        # in the form of a fully connected NN from
        # cell_perception_dim -> cell_state
        self.state_update_rule = nn.Sequential(
            nn.Linear(3 * cell_state_dimension, update_rule_hidden_dim),
            #nn.LayerNorm(update_rule_hidden_dim),
            nn.ReLU(),
            nn.Linear(update_rule_hidden_dim, cell_state_dimension, bias=False),
        )
        with torch.no_grad():
            self.state_update_rule[-1].weight.data.fill_(0.0)

    def get_initial_state(self, batch_size: int):
        return self.initial_state.expand(batch_size, self.input_shape[0], self.input_shape[1], self.cell_state_dimension)

    def perceive(self, state_grid: torch.Tensor) -> torch.Tensor:
        """
            A grid of each individual cells perception is calculated,
            before each cell observes and acts on perception
        """
        state_grid = state_grid.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        sobel_x = F.conv2d(input=state_grid, weight=self.sobel_x, stride=1, padding=1, groups=self.cell_state_dimension)
        sobel_y = F.conv2d(input=state_grid, weight=self.sobel_y, stride=1, padding=1, groups=self.cell_state_dimension)
        
        
        perception_grid = torch.concat([
            state_grid,
            sobel_x,
            sobel_y
        ], dim=1)
        perception_grid = perception_grid.permute(0, 2, 3, 1).to(memory_format=torch.contiguous_format)
        
        return perception_grid
    
    def get_alive_mask(self, state_grid: torch.Tensor):
        state_grid = state_grid.permute(0, 3, 1, 2)
        mask: torch.Tensor = (F.max_pool2d(input=state_grid[:, 3:4, :, :], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) >= self.alpha_alive_threshold) * 1.0
        mask = mask.permute(0, 2, 3, 1)
        return mask

    
    def update(self, state_grid: torch.Tensor):
        alive_mask = self.get_alive_mask(state_grid)
        masked_grid = state_grid * alive_mask
        perception = self.perceive(masked_grid)
                
        flat_perception = perception #.reshape(-1, self.cell_state_dimension * 3)
        update: torch.Tensor = self.state_update_rule(flat_perception)
        
        # Do probabilistic update,
        # to simulate that cells update at different times
        mask = ((torch.rand(update.shape[0], self.input_shape[0], self.input_shape[1]) > self.update_probability) * 1.0).unsqueeze(-1).to(state_grid.device)
        
        update = (
            update
            * mask
        ) #.view(*(*perception.shape[:-1], update.shape[-1]))
        
        new_state = state_grid + update
        alive_mask = self.get_alive_mask(new_state)
        return new_state * alive_mask

    def rollout(self, steps: int, batch_size: int, initial_state: typing.Optional[torch.Tensor]):
        if initial_state is None:
            state_grid = self.get_initial_state(batch_size=batch_size)
        else:
            state_grid = initial_state
        
        states = [state_grid]
        
        for _ in range(steps):
            state_grid = self.update(state_grid)
            states.append(state_grid)
        
        return states

    def forward(self, steps: int, batch_size: int, initial_state: typing.Optional[torch.Tensor]=None):
        states = self.rollout(steps, batch_size, initial_state=initial_state)
        stacked_states = torch.stack(states)
        alpha = stacked_states[..., 3:4].clamp(0, 1)
        rgb = ((1 - alpha + stacked_states[..., :3]).clamp(0, 1) * 255).to(dtype=torch.uint8)
        return rgb, stacked_states

class ONNXWrapper(nn.Module):
        def __init__(self, cell: nn.Module) -> None:
            super().__init__()
            self.cell = cell
        
        def forward(self, steps: torch.Tensor, batch_size: torch.Tensor, initial_state: torch.Tensor):
            if (initial_state == 0).all():
                state = None
            else:
                state = initial_state
            rgb, states = self.cell(steps[0], batch_size[0], state)
            last_state = states[-1]
            return rgb, last_state

class DataSet(torch.utils.data.Dataset):
    def __init__(self, image_num: int=0) -> None:
        super().__init__()
        with PIL.Image.open('images/emoji.png') as im:
            im = np.array(im)
        self.image = torch.tensor(im / 255.0, dtype=torch.float32)[:, 0 + 40*image_num:40 * (image_num + 1), :]
            
    
    def __getitem__(self, idx):
        return self.image
    
    def __len__(self):
        return 100


class Model(pl.LightningModule):
    def __init__(
                self,
                use_pool: bool=True,
                use_damage: bool=True,
                input_shape: typing.Tuple[int, int]=(40, 40),
                damage_num: int=3,
                image_num: int=0,
                batch_size: int=10
            ) -> None:
        super().__init__()
        self.cell: Cell = torch.jit.script(Cell())
        
        self.use_pool = use_pool
        self.use_damage = use_damage

        self.input_shape = input_shape
        self.damage_num = damage_num
        self.image_num = image_num
        self.batch_size = batch_size

        if use_pool:
            self.register_buffer('pool', self.cell.initial_state.expand(1024, self.cell.input_shape[0], self.cell.input_shape[1], self.cell.cell_state_dimension).clone())
    
    def sample_pool(self, samples):
        idxs = torch.randint(0, self.pool.shape[0] - 1, (samples,)).to(self.device)
        return idxs, self.pool[idxs]

    def export_onnx(self, path: str, **kwargs):
        wrapped = ONNXWrapper(self.cell)
        return torch.onnx.export(
            torch.jit.script(wrapped),
            (torch.tensor([100]), torch.tensor([1]), torch.zeros(1, 40, 40, 16, dtype=torch.float)),
            path,
            opset_version=16,
            input_names=['steps', 'batch_size', 'initial_state'],
            **kwargs
        )


    def item_loss(self, output, target) -> torch.Tensor:
        return ((output[..., :4] - target) ** 2).mean(dim=(1, 2, 3))

    def loss(self, output, target) -> torch.Tensor:
        return self.item_loss(output, target).mean(dim=0)

    def training_step(self, batch, batch_idx):
        target = batch[0].to(self.device)
        with torch.no_grad():
            if self.use_pool:
                pool_idxs, pool_samples = self.sample_pool(batch.shape[0])
                pool_loss = self.item_loss(pool_samples, target)
                _, sorted_loss_idxs = pool_loss.sort(descending=True)
                sorted_pool_samples, sorted_pool_idxs = pool_samples[sorted_loss_idxs], pool_idxs[sorted_loss_idxs]
                initial_state = sorted_pool_samples
                
                initial_state[0] = self.cell.initial_state
                if self.use_damage:
                    mask = damage_mask(self.damage_num, self.input_shape[0], self.input_shape[1]).unsqueeze(-1).to(self.device)
                    initial_state[-self.damage_num:] = initial_state[-self.damage_num:] * mask

            else:
                initial_state = None
        
        output, raw = self.cell.forward(np.random.randint(64, 92), batch_size=batch.shape[0], initial_state=initial_state)

        with torch.no_grad():
            if self.use_pool:
                self.pool[sorted_pool_idxs] = raw[-1].detach()

        #if batch_idx % 5 == 0:
        #    grid = torchvision.utils.make_grid(output[-1].permute(0, 3, 1, 2))
            
        #    self.logger.experiment.add_image('test', grid, self.current_epoch)
        return self.loss(raw[-1], target)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=2e-3, betas=(0.5, 0.5))
        return {
            'optimizer': optimizer,
            'lr_scheduler': optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9999)
        }

    def train_dataloader(self):
        return torch.utils.data.DataLoader(DataSet(image_num=self.image_num), batch_size=self.batch_size, num_workers=0)

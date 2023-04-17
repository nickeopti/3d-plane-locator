import pytorch_lightning as pl
import torch
import torch.nn
import torchvision.models.resnet


class FullRegressor(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()

        self.learning_rate = learning_rate

        self.resnet = torchvision.models.resnet.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V2)
        self.elu = torch.nn.ELU()
        self.hidden = torch.nn.Linear(1000, 128)
        self.regressor = torch.nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = self.elu(x)
        x = self.hidden(x)
        x = self.elu(x)
        x = self.regressor(x)

        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, z, d, m = batch
        x, z, d, m = x.float(), z.float(), d.float(), m.float()
        z_, d_, m_ = map(torch.stack, zip(*self(x)))

        z_mse = torch.nn.functional.mse_loss(z_, z)
        d_mse = torch.nn.functional.mse_loss(d_, d)
        m_mse = torch.nn.functional.mse_loss(m_, m)

        loss = z_mse + d_mse + m_mse

        z_mae = torch.abs(z_ - z).mean()
        d_mae = torch.abs(d_ - d).mean()
        m_mae = torch.abs(m_ - m).mean()
        
        self.log('train_loss', loss)

        self.log('z_mae', z_mae)
        self.log('d_mae', d_mae)
        self.log('m_mae', m_mae)

        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer



class BinRegressor(pl.LightningModule):
    def __init__(self, range: int, bins: int = 132, bin_overlap: float = 0.5, learning_rate: float = 1e-3):
        super().__init__()

        self.range = range
        self.n_bins = bins
        self.bin_overlap = bin_overlap
        self.bin_width = self.range / (self.n_bins * (1 - self.bin_overlap) + self.bin_overlap - 1)
        self.learning_rate = learning_rate

        self.resnet = torchvision.models.resnet.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V2)
        self.elu = torch.nn.ELU()
        self.hidden = torch.nn.Linear(1000, 512)
        self.confidence = torch.nn.Linear(512, self.n_bins)
        self.offset = torch.nn.Linear(512, self.n_bins)
        self.softmax = torch.nn.Softmax()

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = self.elu(x)
        x = self.hidden(x)
        x = self.elu(x)

        bin_confidences = self.confidence(x)
        bin_offsets = self.offset(x)

        indices = torch.arange(0, self.n_bins, device=self.device)
        regression_predictions = bin_offsets + (indices - 1) * self.bin_width * (1 - self.bin_overlap) + self.bin_width / 2

        return bin_confidences, regression_predictions
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        xs, zs, _, _ = batch

        loss_confidence = torch.zeros(1, device=self.device)
        loss_location = torch.zeros(1, device=self.device)

        for x, z in zip(xs, zs):
            with torch.no_grad():
                i_max = int(torch.floor(z / self.bin_width / (1 - self.bin_overlap) + 1).item())
                i_min = int(torch.ceil((z - self.bin_width) / self.bin_width / (1 - self.bin_overlap) + 1).item())
                bins_containing_z = torch.arange(i_min, i_max + 1)

                lower = (torch.arange(0, self.n_bins) - 1) * self.bin_width * (1 - self.bin_overlap)
                upper = lower + self.bin_width
                bins_ground_truth = torch.logical_and(lower <= z, upper >= z)
            
            bin_confidences, regressed = self(x.unsqueeze(0))

            regression_prediction = regressed[0][bins_containing_z]
            # regression_prediction = bin_offsets[0][bins_containing_z] + (bins_containing_z - 1) * self.bin_width * (1 - self.bin_overlap) + self.bin_width / 2

            self.log('mae_loc', torch.abs(regression_prediction - z.float().repeat(regression_prediction.shape)))

            loss_confidence += self.cross_entropy(bin_confidences[0], bins_ground_truth / bins_ground_truth.sum())
            loss_location += self.mse(regression_prediction, z.float().repeat(regression_prediction.shape))

        self.log('loss_conf', loss_confidence)
        self.log('loss_loc', loss_location)
        self.log('loss', loss_confidence + loss_location)

        return loss_confidence + loss_location

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

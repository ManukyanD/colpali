import torch


class TrainablePCA(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = torch.nn.Linear(
            self.config.pca_in_size, self.config.pca_out_size, bias=False
        )
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, hidden_states, attention_mask):
        out = self.linear(hidden_states)
        loss = self.mse_loss(
            hidden_states[attention_mask],
            torch.matmul(out, self.linear.weight)[attention_mask],
        )
        return loss, out

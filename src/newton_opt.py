import typing as tp

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, hessian
from torch.utils.data import DataLoader, Dataset


def make_hessian(
    model: nn.Module, x: tuple, y: torch.Tensor, loss_fn: tp.Callable
) -> tp.Any:
    """Computes the Hessian of `model` evaluated at `x`.

    Args:
        model (nn.Module): A neural network
        x (tuple): inputs to the network s.t. model(*x) is a valid statement
        y (torch.Tensor): Targets
        loss_fn (callable): Loss function of form f(outputs, target)

    Returns:
        tp.Any: Hessian
    """

    # Zero existing gradients
    for p in model.parameters():
        p.grad = None

    make_loss = lambda params, x, y: loss_fn(functional_call(model, params, x), y)

    params = dict(model.named_parameters())
    # This returns a nested dictionary of dictionaries
    hess_dict: dict[str, dict[str, torch.Tensor]] = hessian(make_loss)(params, x, y)

    # Effective shape: (num_params, num_params)
    # example: A linear model mapping R^4 -> R with bias has 2 parameters: weight and bias.
    # within these there are 5 elements (4 in weight, 1 in bias). So `hess_dict` is a dict of
    # dicts, with the keys mapping to different blocks of the hessian matrix: dict[i][j] gives
    # grad_i ( grad_j (loss) )  [order might be off here, but i think it should be symmetric anyway]
    # The shapes of these (after flattening) are H_ww: (4, 4), H_wb: (4, 1), H_bw: (1, 4), H_bb: (1, 1) in this example
    # These form the four blocks of the full Hessian matrix of shape (5, 5).
    #
    #   --------------------
    #   |           |      |
    #   |   H_ww    | H_wb |
    #   |           |      |
    #   |           |      |
    #   |------------------|
    #   | H_bw      | H_bb |
    #   --------------------

    named_params = {k: v for k, v in model.named_parameters()}
    # Important that order of parameters is consistent
    param_names = sorted(list(named_params.keys()))
    param_count = sum(p.numel() for p in named_params.values())
    # hess_mat = torch.empty((param_count, param_count))
    with torch.no_grad():
        hess_mat = torch.full(
            (param_count, param_count),
            float("nan"),
            device=next(model.parameters()).device,  # assume all on same device
        )
        cur_row = 0
        for i_key in param_names:
            cur_col = 0
            block_height = 0
            for j_key in param_names:
                # shape: (*param_i.shape, *param_j.shape)
                hess_block_ij = hess_dict[i_key][j_key]

                # reshape to 2D matrix
                hess_block_ij = hess_block_ij.reshape(
                    named_params[i_key].numel(), named_params[j_key].numel()
                )
                block_height = hess_block_ij.shape[0]
                hess_mat[
                    cur_row : cur_row + hess_block_ij.shape[0],
                    cur_col : cur_col + hess_block_ij.shape[1],
                ] = hess_block_ij
                cur_col += hess_block_ij.shape[1]
            cur_row += block_height

    return hess_mat


def make_gradient(
    model: nn.Module, x: tuple, y: torch.Tensor, loss_fn: tp.Callable
) -> torch.Tensor:
    """Computes the gradient of `model` evaluated at `x`. I'm sure there should be a
    way to rewrite the Hessian computation to produce this as a byproduct, but
    I don't feel like figuring it out right now.

    Args:
        model (nn.Module): A neural network
        x (tuple): inputs to the network s.t. model(*x) is a valid statement
        y (torch.Tensor): Targets
        loss_fn (callable): Loss function of form f(outputs, target)

    Returns:
        torch.Tensor: Gradient
    """
    # Zero existing gradients
    for p in model.parameters():
        p.grad = None

    outputs = model(*x)
    loss = loss_fn(outputs, y)
    loss.backward()

    named_params = list(model.named_parameters())
    named_params.sort(key=lambda tup: tup[0])  # sort by name for consistency
    return torch.cat([p.grad.flatten() for _, p in named_params])


def make_newton_step(
    model: nn.Module,
    x: tuple,
    y: torch.Tensor,
    loss_fn: tp.Callable,
    update_params: bool = True,
    clip_step_size: float | None = 1.0,
) -> dict[str, torch.Tensor]:
    """Runs one step of Newton's algorithm on a model with parameters `params`.
    The parameters will be optionally modified in-place and the computed
    optimization step is returned for each parameter.

    Args:
        model (nn.Module): The neural network model
        x (tuple): Inputs to the network
        y (torch.Tensor): Targets
        loss_fn (callable): Loss function of form f(outputs, target)
        update_params (bool, optional): Whether to update the parameters in-place. Defaults to True.

    Returns:
        list[torch.Tensor]: Optimal step for each parameter.
    """
    hessian = make_hessian(model, x, y, loss_fn)
    gradient = make_gradient(model, x, y, loss_fn)
    # Ultimately, we want the inverse hessian-gradient product:  -H^{-1} g
    with torch.no_grad():
        hgp = -torch.linalg.lstsq(hessian, gradient).solution
    if clip_step_size is not None:
        step_norm = torch.norm(hgp)
        if step_norm > clip_step_size:
            hgp = hgp * clip_step_size / step_norm

    # Split the flat hgp into parameter-shaped chunks
    named_params = list(model.named_parameters())
    named_params.sort(key=lambda tup: tup[0])  # sort by name for consistency
    numels = [p.numel() for _, p in named_params]
    updates = torch.split(hgp, numels)

    param_steps = {
        name: step.reshape(param.shape)
        for (name, param), step in zip(named_params, updates)
    }

    if update_params:
        # Update parameters in-place
        with torch.no_grad():
            for name, param in model.named_parameters():
                param += param_steps[name]

    return param_steps


class Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x**2


class SyntheticDataset(Dataset):
    def __init__(self, dset_size: int, d_data: int) -> None:
        super().__init__()
        self.data = torch.randn(dset_size, d_data)
        self.true_w = torch.linspace(1, 2, d_data)
        self.targets = (self.data * self.true_w[None, :]).sum(
            dim=1, keepdim=True
        ) + 0.1 * torch.randn(dset_size, 1)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx, :], self.targets[idx, :]


if __name__ == "__main__":
    d_data = 32
    num_epochs = 500
    dataset_size = 2000
    batch_size = 64

    # model = nn.Linear(d_data, 1)
    model = nn.Sequential(nn.Linear(d_data, d_data), nn.ReLU(), nn.Linear(d_data, 1))

    loss = nn.MSELoss()

    dataset = SyntheticDataset(dataset_size, d_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []

    print(f"Initial loss:")
    for data, targets in dataloader:
        outputs = model(data)
        train_loss = loss(outputs, targets)
        print(f"Train loss {train_loss.item():.4f}")
        losses.append(train_loss.item())
        break
    for step in range(num_epochs):
        for data, targets in dataloader:
            make_newton_step(model, (data,), targets, loss, clip_step_size=100)
        with torch.no_grad():
            outputs = model(data)
            train_loss = loss(outputs, targets)
            losses.append(train_loss.item())
            print(f"Epoch {step}: Train loss {train_loss.item():.4f}")

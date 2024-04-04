import torch
import numpy as np
from tqdm import tqdm
from model import SimpleCNN
from dataset import MyDataset, split_dataset
from torch.utils.data import DataLoader





BATCH_SIZE = 32
TOTAL_EPOCH = 10
LEARNING_RATE = 0.1
LEARNING_DECAY_RATE = 0.995
TRAIN_STEP_ITER = 1000





def get_lr(optimizer: torch.optim.Optimizer) -> float:
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_epoch(
        model: torch.nn.Module,
        data_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> float:
    model.train()
    total_loss: float = 0.0
    inputs_amount: int = 0
    pbar = tqdm(data_loader, total=TRAIN_STEP_ITER)
    for idx, (inputs, truths) in enumerate(pbar):
        if idx == TRAIN_STEP_ITER: break
        optimizer.zero_grad()
        outputs = model(inputs.cuda())
        loss: torch.Tensor = criterion(outputs, truths.cuda())
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item() * inputs.size(0)
        inputs_amount += inputs.size(0)
        avg_loss = total_loss / inputs_amount
        pbar.set_description(f"[TRAIN] Loss: {avg_loss:0.10} | LR: {get_lr(optimizer):0.10}")
    return avg_loss


def valid_epoch(
        model: torch.nn.Module,
        data_loader: DataLoader,
        criterion: torch.nn.Module,
    ) -> float:
    model.eval()
    total_loss: float = 0.0
    inputs_amount: int = 0
    pbar = tqdm(data_loader)
    for inputs, truths in pbar:
        outputs = model(inputs.cuda())
        loss: torch.Tensor = criterion(outputs, truths.cuda())
        total_loss += loss.item() * inputs.size(0)
        inputs_amount += inputs.size(0)
        avg_loss = total_loss / inputs_amount
        pbar.set_description(f"[VALID] Loss: {avg_loss:0.10}")
    return avg_loss


def find_outlier(
        model: torch.nn.Module,
        data_loader: DataLoader,
        criterion: torch.nn.Module,
    ) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_outputs, losses = [], []
    for inputs, truths in tqdm(data_loader):
        outputs: torch.Tensor = model(inputs.cuda())
        outputs_np: np.ndarray = outputs.detach().cpu().numpy()
        all_outputs.extend(outputs_np.flatten().tolist())
        loss: torch.Tensor = criterion(outputs, truths.cuda())
        loss_np: np.ndarray = loss.detach().cpu().numpy()
        losses.extend(loss_np.flatten().tolist())
    return np.array(losses), np.array(all_outputs)


def train():

    dataset = MyDataset()
    loader  = DataLoader(dataset, BATCH_SIZE, num_workers=6, persistent_workers=True, pin_memory=True)
    train_dataset, valid_dataset = split_dataset(dataset)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, num_workers=5, persistent_workers=True, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, BATCH_SIZE, num_workers=5, persistent_workers=True, pin_memory=True)

    model = SimpleCNN().cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=LEARNING_DECAY_RATE)

    best_val_loss = 1.0
    for epoch in range(TOTAL_EPOCH):
        print(f"Epoch: {epoch+1:2} / {TOTAL_EPOCH:2}")
        train_epoch(model, train_loader, criterion, optimizer, lr_scheduler)
        val_loss = valid_epoch(model, valid_loader, criterion)
        if epoch > 2 and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

    print(f"\nBest valid loss: {best_val_loss}.\n")

    losses, corr_preds = find_outlier(best_model, loader, torch.nn.MSELoss(reduction="none"))
    worst_idxs         = np.argsort(losses)[::-1]
    worst_filepaths    = np.array(dataset.inputs_filepaths)[worst_idxs[:10]]
    worst_corr_truths  = dataset.truths.flatten()[worst_idxs[:10]]
    worst_corr_preds   = corr_preds[worst_idxs[:10]]
    worst_losses       = losses[worst_idxs[:10]]
    with open("Files with Largest Losses.csv", 'w') as csv_file:
        csv_file.write(f"File,Correlation Truth,Correlation Prediction,Loss\n")
        for filepath, corr_t, corr_p, loss in \
            zip(worst_filepaths, worst_corr_truths, worst_corr_preds, worst_losses):
            csv_file.write(f"{filepath.split('/')[-1]},{corr_t},{corr_p},{loss}\n")
    return
    




if __name__ == "__main__":
    train()
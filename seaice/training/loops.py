import torch


def train_one_epoch(model, optimiser, criterion, data_loader, device):
    model.train()
    train_loss = 0.0

    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        optimiser.zero_grad()
        output = model(input.permute(0, 3, 1, 2))  # change to (B, C, H, W)
        loss = criterion(output, target.permute(0, 3, 1, 2))  # change to (B, C, H, W)
        loss.backward()
        optimiser.step()
        train_loss += loss.item() * input.size(0)
    return train_loss / len(data_loader.dataset)


def test_one_epoch(model, criterion, data_loader, device):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            output = model(input.permute(0, 3, 1, 2))  # change to (B, C, H, W)
            loss = criterion(
                output, target.permute(0, 3, 1, 2)
            )  # change to (B, C, H, W)
            test_loss += loss.item() * input.size(0)
    return test_loss / len(data_loader.dataset)

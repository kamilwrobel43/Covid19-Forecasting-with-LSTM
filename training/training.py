from tqdm import tqdm
import torch
import copy

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device, patience=3):
    best_avg_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
            sequences = batch["sequence"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                sequences = batch["sequence"].to(device)
                labels = batch["label"].to(device).unsqueeze(1)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)

        if avg_val_loss < best_avg_loss:
            best_avg_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "../best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_model_wts)
    return model

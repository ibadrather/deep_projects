import torch
from tqdm import tqdm, trange


def trainer(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    learning_rate=1e-5,
    epochs=100,
    device="cuda",
    validate_every=10,
    patience=100,
):

    # Initialize
    train_loss = torch.zeros(0)
    val_loss = torch.zeros(0)

    best_val_loss = 10e10
    patience = 0

    model.train()

    with trange(epochs) as progress_bar:
        for epoch in progress_bar:
            for batch_idx, (feature, target) in enumerate(
                tqdm(train_loader, leave=False)
            ):
                # Hardware acceleration
                feature, target = feature.to(device), target.to(device)

                # Forward pass
                logits = model(feature)

                # Loss
                train_loss = criterion(logits, target)

                # Backward pass
                train_loss.backward()

                # Update weights
                optimizer.step()
                optimizer.zero_grad()

                # Progess bar
                progress_bar.set_description(
                    f"Epoch: {epoch+1}/{epochs}, Training Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}"
                )

            # Validation
            if val_loader is not None and epoch % validate_every == 0:
                model.eval()
                with torch.no_grad():
                    for batch_idx, (feature, target) in enumerate(
                        tqdm(val_loader, leave=False)
                    ):
                        # Hardware acceleration
                        feature, target = feature.to(device), target.to(device)

                        # Forward pass
                        logits = model(feature)

                        # Loss
                        val_loss = criterion(logits, target)

                        # Progess bar
                        progress_bar.set_description(
                            f"Epoch: {epoch+1}/{epochs}, Training Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}"
                        )

                        # save model if validation loss has decreased
                        if val_loss.item() < best_val_loss:
                            torch.save(model.state_dict(), f"model_{epoch}.pt")
                            best_val_loss = val_loss.item()
                            patience = 0

                        else:
                            patience += 1   

                            # early stopping 
                            if patience >= patience:
                                return model

                # Back to training
                model.train()

    return model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from datetime import datetime
from torchsummary import summary


class TTCCNN(nn.Module):
    def __init__(self):
        super(TTCCNN, self).__init__()
        
    def build_model(self):
        print("building model")
        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        summary(model, (1, 100, 100))
        return model
        

    def fit_model(self, model, input_data, output_data, epochs, batch_size, validation_split, save_model=True):
        # Convert data to PyTorch tensors
        input_data = torch.tensor(input_data, dtype=torch.float32)
        output_data = torch.tensor(output_data, dtype=torch.float32)
        
        # Create DataLoader
        dataset = TensorDataset(input_data, output_data)
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Define loss and optimizer
        criterion = nn.MSELoss()  # Replace with your custom loss if needed
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Training loop
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # Validation loop
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}')

        # Save model
        if save_model:
            now = datetime.now()
            model_version = f"{now.day}{now.month}{now.hour}{now.minute}"
            model_name = f'CNN_TTC_ks5325_sigmoid_ncc_loss_{model_version}_training_set_{len(input_data)}.pth'
            torch.save(model.state_dict(), './Models/TTC_models/' + model_name)

        return model_name


import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets  

def test(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = test_loss / total
    accuracy = correct / total
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

def train(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    best_loss = float('inf')
    loss_counter = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            if step % 10 == 0:
                print(f"[Epoch {epoch+1} | Step {step}] Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}")
        val_loss, val_accuracy = test(model, valid_loader, criterion, device)
        print(f"val_accuracy={val_accuracy:.4f}, train_loss={train_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            loss_counter = 0
        else:
            loss_counter += 1
            if loss_counter >= 2:
                print("Early stopping triggered.")
                break

    return model

def net(num_classes=133):
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def create_data_loaders(data_dir, batch_size):
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, valid_loader, test_loader

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)
    train_loader, valid_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    model = train(model, train_loader, valid_loader, criterion, optimizer, device, args.epochs)
    test_loss, test_accuracy = test(model, test_loader, criterion, device)

    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), path)

    print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', 'data'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '.'))

    args = parser.parse_args()
    main(args)
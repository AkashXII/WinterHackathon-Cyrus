from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.models as models
from sklearn.model_selection import train_test_split

# 1. Load dataset (HF → Python list)
hf_dataset = load_dataset("aneeshd27/Corals-Classification")["train"]
data = list(hf_dataset)

# 2. Train / test split
train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42
)

# 3. Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 4. Simple PyTorch Dataset
class CoralDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]["image"].convert("RGB")
        label = self.data[idx]["label"]
        image = self.transform(image)
        return image, label

train_dataset = CoralDataset(train_data, transform)
test_dataset = CoralDataset(test_data, transform)

# 5. DataLoaders (NO workers, Windows-safe)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# 6. Model
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 7. Loss & optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 8. Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# 9. Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 10. Save model
torch.save(model.state_dict(), "resnet50_coral.pth")
print("Model saved")

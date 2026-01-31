import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# -------------------------------------------------
# 1. Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------------------------------
# 2. Dataset Path (ONLY TRAIN)
# -------------------------------------------------
train_dir = "Corn-Diseases"

# -------------------------------------------------
# 3. Transforms
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------------------------
# 4. Dataset & DataLoader
# -------------------------------------------------
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)

# -------------------------------------------------
# 5. Model (ResNet50)
# -------------------------------------------------
model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# -------------------------------------------------
# 6. Loss & Optimizer
# -------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# -------------------------------------------------
# 7. Training
# -------------------------------------------------
epochs = 10
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")

# -------------------------------------------------
# 8. Evaluation on TRAIN DATA
# -------------------------------------------------
model.eval()

y_true = []
y_pred = []
y_prob = []

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_prob.extend(probabilities.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# -------------------------------------------------
# 9. Metrics
# -------------------------------------------------
accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro")
recall    = recall_score(y_true, y_pred, average="macro")
f1        = f1_score(y_true, y_pred, average="macro")

print("\nMETRICS (TRAIN DATA ONLY)")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-Score :", f1)

# -------------------------------------------------
# 10. Confusion Matrix
# -------------------------------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Train Dataset)")
plt.show()

# -------------------------------------------------
# 11. AUC-ROC (Multiclass)
# -------------------------------------------------
y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

plt.figure(figsize=(8, 6))

for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC-ROC Curve (Train Dataset)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# -------------------------------------------------
# 12. Metric Bar Plot
# -------------------------------------------------
metrics = [accuracy, precision, recall, f1]
labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]

plt.figure(figsize=(7, 5))
plt.bar(labels, metrics)
plt.ylim(0, 1)
plt.title("Classification Metrics (Train Dataset)")
plt.ylabel("Score")
plt.grid(axis="y")
plt.show()

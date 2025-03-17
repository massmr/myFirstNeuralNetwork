import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Génération de données fictives
def generate_data(n_samples=100):
    X = np.random.rand(n_samples, 2).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 1).astype(np.float32).reshape(-1, 1)  # Classification binaire
    return torch.tensor(X), torch.tensor(y)

X_train, y_train = generate_data(200)
X_test, y_test = generate_data(50)

# Définition du modèle
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 6)
        self.fc2 = nn.Linear(6, 4)
        self.fc3 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = NeuralNetwork()

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entraînement
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Époque {epoch+1}, Perte: {loss.item():.4f}')

# Évaluation
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = (y_pred > 0.5).float()
    accuracy = (y_pred == y_test).float().mean()
    print(f'Précision: {accuracy:.2f}')

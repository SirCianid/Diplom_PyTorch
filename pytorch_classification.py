"""1. Загрузка необходимых библиотек."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

"""2. Генерируем данные для обучения и разделяем выборки. В данной модели будем иммитировать классификацию спам/не спам.
Основываться будем на длине условных сообщений, и колличестве восклицательных знаков.
num_samples - количество записей (тех самых сообщений).
message_lengths - генерируем случайные значения длинны условных сообщений.
exclamation_mark - генерируем случайное колличество восклицательных знаков.
spam_probability - вычисляем вероятность того, что сообщение будет спамом.
spam_labels - на основе spam_probability присваиваем сообщениям метку "спам" (0) или "не спам" (1) соответсвенно.
x - матрица признаков.
y - вектор целевых переменных (меток класса)."""
np.random.seed(42)
num_samples = 200
message_lengths = np.random.uniform(10, 100, num_samples)
exclamation_marks = np.random.randint(0, 10, num_samples)
spam_probability = 0.2 + 0.005 * message_lengths + 0.05 * exclamation_marks
spam_labels = (np.random.rand(num_samples) < spam_probability).astype(int)

X = np.column_stack((message_lengths, exclamation_marks))
y = spam_labels

"""2.1. Разделяем данные на обучающую и тестовую выборки. Также воспользуемся trin_test_split от scikit-learn."""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""3. Масштабируем входные признаки."""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""3.1. Преобразуем данные в тензоры PyTorch."""
X_train = torch.from_numpy(X_train_scaled).float()
X_test = torch.from_numpy(X_test_scaled).float()
y_train = torch.from_numpy(y_train).float().reshape(-1, 1)
y_test = torch.from_numpy(y_test).float().reshape(-1, 1)

"""4. Определение модели.
В функции __init__ - оределяем линейный слой который и будет обучаться.
Функции forward - метод, который определяет как данные будут обрабатываться. Сначала происходит линейное преобразование, 
а затем к результату применяется сигмоидальная функция"""


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegression(2)

"""5. Задаем функцию потерь и оптимизатора."""
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

"""6. Запускаем обучение модели, также, с выводом информации о прогрессе обучения через каждые 200 эпох."""
num_epochs = 2000
for epoch in range(num_epochs):
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")

"""7. Оценка модели и визуализация результатов."""
with torch.no_grad():
    y_predicted_test = model(X_test)
    y_predicted_test_cls = (y_predicted_test >= 0.5).float()
    accuracy = accuracy_score(y_test.numpy(), y_predicted_test_cls.numpy())
    print(f"Accuracy: {accuracy:.4f}")

"""7.1. Вычисляем матрицу ошибок и визуализируем ее."""
cm = confusion_matrix(y_test.numpy(), y_predicted_test_cls.numpy())
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.xlabel('Прогнозируемые значения')
plt.ylabel('Реальные значения')
plt.title('Матрица ошибок')
plt.show()

"""7.2. Получаем отчет о классификации."""
report = classification_report(y_test.numpy(), y_predicted_test_cls.numpy())
print("Classification Report:")
print(report)

"""7.3. Визуализация результатов."""
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.squeeze().numpy(), label="Data", cmap='RdBu')
with torch.no_grad():
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 100),
                            torch.linspace(y_min, y_max, 100))
    grid = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1)
    Z = model(grid).reshape(xx.shape)
    plt.contourf(xx.numpy(), yy.numpy(), Z.detach().numpy(), cmap='RdBu', alpha=0.4)

plt.xlabel("Длинна сообщения")
plt.ylabel("Восклицательные знаки")
plt.title("Классификация")
plt.show()

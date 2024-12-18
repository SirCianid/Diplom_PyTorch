"""1. Загрузка необходимых библиотек."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

"""2. Генерируем данные для обучения. Для более удобного понимания, модель будет предсказывать стоимость 
квартиры основываясь на ее площади. 
np.random.seed - фиксирует случайные значения для воспроизводимости результатов.
house_sizes - генерирует случайные значения площади квартиры в кв. м.
house_prices - вычисляет стоимость квартиры (в тыс. руб.) зависящую от ее площади, 
и случайного шума, который иммитирует различные факторы оказывающие влияние на стоимость жилья."""
np.random.seed(42)
house_sizes = np.random.uniform(75, 250, 200).reshape(-1, 1)
usd_to_rub = 104
house_prices = (640 * house_sizes + 7640 + np.random.normal(0, 10000, 200).reshape(-1, 1))*usd_to_rub

"""3. Разделяем данные на обучающую и тестовую выборки. В данном случае уже не будем прибегать к ручному разделению,
 а воспользуемся инструментом из библиотеки scikit-learn."""
X_train, X_test, y_train, y_test = train_test_split(house_sizes, house_prices, test_size=0.2, random_state=42)

"""4. Производим масштабирование входных и выходных данных."""
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

"""4.1. Преобразуем массивы NumPy в тензоры PyTorch."""
X_train = torch.from_numpy(X_train_scaled).float()
X_test = torch.from_numpy(X_test_scaled).float()
y_train = torch.from_numpy(y_train_scaled).float()
y_test = torch.from_numpy(y_test_scaled).float()

"""5. Определение модели. В PyTorch модели определяются как, так называемые, классовые модули, 
наследуемые от базового класса всех нейросетевых слоев и моделей - nn.Module.
В функции __init__ - оределяем линейный слой который и будет обучаться.
Функции forward - метод, который определяет как данные будут обрабатываться. Задаем просто линейный слой."""


class HousePriceRegression(nn.Module):
    def __init__(self):
        super(HousePriceRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = HousePriceRegression()

"""6. Задаем функцию потерь и оптимизатора."""
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

"""7. Запускаем обучение модели, с выводом информации о прогрессе обучения через каждые 200 эпох."""
num_epochs = 2000
for epoch in range(num_epochs):
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")

"""8. Оценка модели и визуализация результатов."""
with torch.no_grad():
    y_predicted_test = model(X_test)
    y_predicted_train = model(X_train)

"""8.1. Переводим предсказанные данные в оригинальный масштаб."""
y_predicted_test_orig = scaler_y.inverse_transform(y_predicted_test.numpy())
y_predicted_train_orig = scaler_y.inverse_transform(y_predicted_train.numpy())
y_test_orig = scaler_y.inverse_transform(y_test.numpy())
y_train_orig = scaler_y.inverse_transform(y_train.numpy())
X_test_orig = scaler_X.inverse_transform(X_test.numpy())
X_train_orig = scaler_X.inverse_transform(X_train.numpy())

"""8.4. Рассчитываем метрики и выводим результаты."""
mse = mean_squared_error(y_test_orig, y_predicted_test_orig)
mae = mean_absolute_error(y_test_orig, y_predicted_test_orig)
r2 = r2_score(y_test_orig, y_predicted_test_orig)

print(f"MSE (Mean Squared Error) на тестовых данных: {mse:.2f}")
print(f"MAE (Mean Absolute Error) на тестовых данных: {mae:.2f}")
print(f"R^2 (Coefficient of Determination) на тестовых данных: {r2:.2f}")

"""8.3. Визуализация результатов работы модели."""
plt.scatter(X_train_orig, y_train_orig, label="Train data")
plt.scatter(X_test_orig, y_test_orig, label="Test data")
plt.plot(X_test_orig, y_predicted_test_orig, color="red", label="Regression Line")
plt.xlabel("Площадь жилья (кв. м)")
plt.ylabel("Стоимость жилья (тыс. руб.)")
plt.legend()
plt.show()

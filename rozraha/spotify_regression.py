import pandas as pd

# Завантаження даних
df = pd.read_csv("spotify_top_1000_tracks.csv")

# Перевіримо перші рядки
print(df.head())

# Інформація про набір
print(df.info())

from sklearn.model_selection import train_test_split

# Створення нових ознак
df['track_name_len'] = df['track_name'].apply(len)
df['artist_name_len'] = df['artist'].apply(len)
df['release_year'] = pd.to_datetime(df['release_date'], format='mixed', errors='coerce').dt.year


# Оновлений список ознак
features = ['duration_min', 'track_name_len', 'artist_name_len', 'release_year']
target = 'popularity'

# Матриця X і вектор y
X = df[features]
y = df[target]

# Поділ на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Створення та навчання моделі
lr = LinearRegression()
lr.fit(X_train, y_train)

# Прогнозування
y_pred_lr = lr.predict(X_test)

# Оцінка точності
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# Вивід результатів
print("Лінійна регресія (OLS) — результати:")
print(f"MAE: {mae_lr:.2f}")
print(f"RMSE: {rmse_lr:.2f}")
print(f"R²: {r2_lr:.4f}")
print("Коефіцієнти:")
for feat, coef in zip(features, lr.coef_):
    print(f"{feat}: {coef:.4f}")

from sklearn.linear_model import Ridge

# Створюємо та тренуємо Ridge-модель
ridge = Ridge(alpha=10.0)  # Можеш змінити alpha (λ), наприклад: 1.0, 0.1, 100
ridge.fit(X_train, y_train)

# Прогноз
y_pred_ridge = ridge.predict(X_test)

# Оцінка
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

# Вивід результатів
print("\nRidge-регресія (L2) — результати:")
print(f"MAE: {mae_ridge:.2f}")
print(f"RMSE: {rmse_ridge:.2f}")
print(f"R²: {r2_ridge:.4f}")
print("Коефіцієнти:")
for feat, coef in zip(features, ridge.coef_):
    print(f"{feat}: {coef:.4f}")

from sklearn.linear_model import Lasso

# Створюємо та тренуємо Lasso-модель
lasso = Lasso(alpha=1.0)  # Можна експериментувати з alpha: 0.1, 1.0, 10.0
lasso.fit(X_train, y_train)

# Прогноз
y_pred_lasso = lasso.predict(X_test)

# Оцінка
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso = r2_score(y_test, y_pred_lasso)

# Вивід результатів
print("\nLasso-регресія (L1) — результати:")
print(f"MAE: {mae_lasso:.2f}")
print(f"RMSE: {rmse_lasso:.2f}")
print(f"R²: {r2_lasso:.4f}")
print("Коефіцієнти:")
for feat, coef in zip(features, lasso.coef_):
    print(f"{feat}: {coef:.4f}")


print("Train set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# Для моделі OLS
results_df = pd.DataFrame({
    'Реальне значення': y_test,
    'Прогноз OLS': y_pred_lr,
    'Прогноз Ridge': y_pred_ridge,
    'Прогноз Lasso': y_pred_lasso
})

print(results_df.head(10))  # або збережи все:
results_df.to_csv("results_comparison.csv", index=False)


import matplotlib.pyplot as plt
import seaborn as sns

# OLS — прогноз проти реальних значень
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.6)
plt.xlabel("Реальні значення (popularity)")
plt.ylabel("Прогнозовані значення")
plt.title("Лінійна регресія (OLS): прогноз vs реальність")
plt.grid(True)
plt.show()

# Ridge
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_ridge, alpha=0.6, color='orange')
plt.xlabel("Реальні значення (popularity)")
plt.ylabel("Прогнозовані значення")
plt.title("Ridge-регресія: прогноз vs реальність")
plt.grid(True)
plt.show()

# Lasso
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_lasso, alpha=0.6, color='green')
plt.xlabel("Реальні значення (popularity)")
plt.ylabel("Прогнозовані значення")
plt.title("Lasso-регресія: прогноз vs реальність")
plt.grid(True)
plt.show()


import pandas as pd

# Побудова таблиці з коефіцієнтами
coef_df = pd.DataFrame({
    "Ознака": features,
    "OLS": lr.coef_,
    "Ridge": ridge.coef_,
    "Lasso": lasso.coef_
})

# Перетворюємо ознаку в індекс і будуємо графік
coef_df.set_index("Ознака").plot(kind="bar", figsize=(10, 6))
plt.title("Порівняння коефіцієнтів моделей")
plt.ylabel("Значення коефіцієнта")
plt.grid(True)
plt.tight_layout()
plt.show()


# Створення таблиці з прогнозами та похибками
results_df = pd.DataFrame({
    'Реальне значення (popularity)': y_test.values,
    'Прогноз OLS': y_pred_lr,
    'Помилка OLS': abs(y_test.values - y_pred_lr),
    'Прогноз Ridge': y_pred_ridge,
    'Помилка Ridge': abs(y_test.values - y_pred_ridge),
    'Прогноз Lasso': y_pred_lasso,
    'Помилка Lasso': abs(y_test.values - y_pred_lasso)
})

# Округлення для зручності
results_df = results_df.round(2)

# Збереження у CSV-файл
results_df.to_csv("results_comparison.csv", index=False, encoding='utf-8-sig')

print("\n Результати збережено у файл 'results_comparison.csv'")

# Побудова гістограм похибок для кожної моделі
plt.figure(figsize=(10, 6))
sns.histplot(abs(y_test - y_pred_lr), bins=30, color='blue', label='OLS', kde=True)
sns.histplot(abs(y_test - y_pred_ridge), bins=30, color='orange', label='Ridge', kde=True)
sns.histplot(abs(y_test - y_pred_lasso), bins=30, color='green', label='Lasso', kde=True)

plt.title("Розподіл абсолютних похибок (MAE) для моделей")
plt.xlabel("Абсолютна похибка (|реальне − прогноз|)")
plt.ylabel("Кількість спостережень")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Підготовка даних у форматі "довгий формат" (long-form DataFrame)
boxplot_df = pd.DataFrame({
    'OLS': abs(y_test - y_pred_lr),
    'Ridge': abs(y_test - y_pred_ridge),
    'Lasso': abs(y_test - y_pred_lasso)
})

# Перетворюємо для Seaborn
boxplot_df_melted = boxplot_df.melt(var_name='Модель', value_name='Абсолютна похибка')

# Побудова boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Модель', y='Абсолютна похибка', hue='Модель', data=boxplot_df_melted, palette='Set2', legend=False)
plt.title("Boxplot абсолютних похибок для моделей")
plt.grid(True)
plt.tight_layout()
plt.show()

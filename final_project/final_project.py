import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

df = pd.read_excel("ipp_2022_2025.xlsx")
df.to_csv("ipp_2022_2025.csv", index=False, encoding="utf-8-sig")

df = pd.read_csv("ipp_2022_2025.csv")
print(df.head())
print(df.info())

# проверка пропусков
print(df.isna().sum())

# сезонность
seasonality = df.groupby("month")["positive"].mean()
plt.figure()
seasonality.plot(kind="line", marker="o")
plt.xlabel("Месяц")

plt.ylabel("Среднее количество выявленных")
plt.title("Сезонность выявляемости ИППП по месяцам")
plt.grid(True)
plt.savefig('figures/seasonality.png', dpi=300, bbox_inches="tight")
plt.close()

# распределение по полу
sex_distribution = df.groupby("sex")["positive"].mean()
plt.figure()
sex_distribution.plot(kind="bar")
plt.xlabel("Пол")
plt.ylabel("Среднее количество выявленных")
plt.title("Выявляемость ИППП по полу")
plt.grid(axis="y")
plt.savefig('figures/sex_distribution.png', dpi=300, bbox_inches="tight")
plt.close()

# распределение по типу инфекции
disease_distribution = (
    df.groupby("disease")["positive"]
    .mean()
    .sort_values()
)
plt.figure()
disease_distribution.plot(kind="barh")
plt.xlabel("Среднее количество выявленных")
plt.ylabel("Инфекция")
plt.title("Выявляемость ИППП по типу инфекции")
plt.grid(axis="x")
plt.savefig('figures/disease_distribution.png', dpi=300, bbox_inches="tight")
plt.close()

# распределение выявленных случаев
plt.figure()
df["positive"].hist(bins=20)
plt.xlabel("Количество выявленных")
plt.ylabel("Частота")
plt.title("Распределение выявленных случаев")
plt.savefig('figures/positive_hist.png', dpi=300, bbox_inches="tight")
plt.close()
# распределение обследованных
plt.figure()
df["tested"].hist(bins=20)
plt.xlabel("Количество обследованных")
plt.ylabel("Частота")
plt.title("Распределение обследованных")
plt.savefig('figures/tested_hist.png', dpi=300, bbox_inches="tight")
plt.close()

# yдаляем строки без ключевых значений
df = df.dropna(subset=["tested", "positive"])

# кодирование категориальных признаков
df_ml = pd.get_dummies(
    df,
    columns=["sex", "disease"],
    drop_first=True
)

# формирование признаков и цели
X = df_ml.drop(columns=["positive"])
y = df_ml["positive"]

# разбиваем на тестовую и обучающую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# проверка
print(y.value_counts(normalize=True))
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

# линейная регрессия
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# оценка качества


def evaluate(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return mae, rmse


mae_lr, rmse_lr = evaluate(y_test, y_pred_lr)
mae_rf, rmse_rf = evaluate(y_test, y_pred_rf)

print("\nРЕЗУЛЬТАТЫ МОДЕЛЕЙ")
print(f"Linear Regression -> MAE: {mae_lr:.2f},RMSE:{rmse_lr:.2f}")
print(f"Random Forest    -> MAE: {mae_rf:.2f},RMSE: {rmse_rf:.2f}")

# интерпретация random forest

feature_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure()
feature_importance.head(10).plot(kind="barh")
plt.xlabel("Важность признака")
plt.title("Топ-5 наиболее важных признаков")
plt.gca().invert_yaxis()
plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches="tight")
plt.close()

print("\nТоп-5 важных признаков:")
print(feature_importance.head(5))


print("\nАнализ завершён. Графики сохранены в папке figures")

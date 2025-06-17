import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR


def run_traditional_ml_analysis(file_path):
    print("--- 1. 加载和探索数据 ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 数据文件未在路径 '{file_path}' 找到。请检查路径。")
        return

    print("数据前5行:\n", df.head())
    print("\n数据维度:", df.shape)
    print("\n数据信息:")
    df.info()
    print("\n描述性统计:\n", df.describe().T)

    print("\n--- 2. 探索性数据分析 (EDA) ---")
    print("正在生成EDA图表 ")

    # 平均用电量 vs. 周状态
    plt.figure(figsize=(10, 7))
    sns.barplot(data=df, x="WeekStatus", y="Usage_kWh", errorbar=None)
    plt.title("Average Usage by WeekStatus", fontsize=25)
    plt.show()

    # 相关性热力图
    plt.figure(figsize=(16, 12))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f")
    plt.title("Correlation Heatmap", fontsize=15)
    plt.show()

    print("\n--- 3. 数据预处理 ---")
    if 'date' in df.columns:
        df = df.drop(['date'], axis=1)
    df_processed = pd.get_dummies(df, drop_first=True)

    # 分离特征和目标
    X = df_processed.drop('Usage_kWh', axis=1)
    y = df_processed['Usage_kWh']

    # 特征选择
    select_reg = SelectKBest(k=10, score_func=f_regression).fit(X, y)
    X_selected = select_reg.transform(X)
    selected_cols = X.columns[select_reg.get_support()]
    print("选择的10个特征:\n", selected_cols)

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    print("\n--- 4. 模型训练与评估 ---")

    models_dict = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=0.5),
        'Lasso Regression': Lasso(alpha=0.5),
        'ElasticNet Regression': ElasticNet(alpha=0.5),
        'Support Vector Regression': SVR(kernel='rbf')
    }

    results = {}

    for name, model in models_dict.items():
        print(f"\n训练模型: {name}")
        if name == 'Support Vector Regression':
            model.fit(X_scaled, y)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2_test = metrics.r2_score(y_test, y_pred)

        if name == 'Support Vector Regression':
            r2_train = metrics.r2_score(y, model.predict(X_scaled))
        else:
            r2_train = model.score(X_train, y_train)

        results[name] = {'RMSE': rmse, 'R2_Test': r2_test, 'R2_Train': r2_train}

        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"R² (训练集): {r2_train:.4f}")
        print(f"R² (测试集): {r2_test:.4f}")

    print("\n--- 5. 模型性能比较 ---")

    model_names = list(results.keys())
    r2_scores = [res['R2_Test'] for res in results.values()]
    rmse_values = [res['RMSE'] for res in results.values()]

    # R2分数比较图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x=r2_scores, y=model_names)
    plt.xlabel('R² Score (Test Set)')
    plt.title('Comparison of R² Scores')

    # RMSE比较图
    plt.subplot(1, 2, 2)
    sns.barplot(x=rmse_values, y=model_names)
    plt.xlabel('RMSE')
    plt.title('Comparison of RMSE')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data_file_path = 'Steel_industry_data.csv'
    run_traditional_ml_analysis(data_file_path)
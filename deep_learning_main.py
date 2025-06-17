import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

from data_utils import load_and_preprocess_data, create_supervised_dataset, create_multiscale_multitask_dataset
from models_pytorch import LSTMModel, GRUModel, CNNLSTMModel, TransformerModel, MultiScaleMultiTaskLSTM
from training_utils import train_model, evaluate_model, train_multitask_model


def run_deep_learning_analysis(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.001

    # data prepare
    df_full = load_and_preprocess_data(file_path)

    features = [col for col in df_full.columns if col != 'Usage_kWh']
    target = ['Usage_kWh']
    TIME_STEPS = 24

    X, y, scaler_X_simple, scaler_y_simple = create_supervised_dataset(
        df_full, target_cols=target, feature_cols=features, time_steps=TIME_STEPS
    )

    # 划分数据集
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # DataLoader
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
                              batch_size=BATCH_SIZE)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()),
                            batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()),
                             batch_size=BATCH_SIZE)

    # 训练和评估简单模型
    simple_models = {
        "LSTM": LSTMModel(input_dim=X_train.shape[2], hidden_dim=64, output_dim=1, num_layers=2),
        "GRU": GRUModel(input_dim=X_train.shape[2], hidden_dim=64, output_dim=1, num_layers=2),
    }

    results = {}
    for name, model in simple_models.items():
        print(f"\n--- 训练 {name} 模型 ---")
        model = train_model(model, train_loader, val_loader, EPOCHS, LR, device)
        print(f"\n--- 评估 {name} 模型 ---")
        predictions, actuals = evaluate_model(model, test_loader, scaler_y_simple, device)
        results[name] = {'predictions': predictions, 'actuals': actuals}

    #   为多任务模型准备数据
    print("\n--- 准备多任务学习数据 ---")
    mt_targets = ['Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh']
    mt_features = [col for col in df_full.columns if col not in mt_targets]

    [X_mt_short, X_mt_long], Y_mt, _, scaler_y_mt = create_multiscale_multitask_dataset(
        df_full, targets=mt_targets, features=mt_features, short_steps=24, long_steps=168  # 1周
    )


    # 训练和评估多任务模型
    print("\n--- 训练多任务多尺度LSTM模型 ---")
    mt_model = MultiScaleMultiTaskLSTM(
        input_dim=len(mt_features),
        hidden_dim_short=64, hidden_dim_long=64,
        fusion_dim=128, task_hidden_dim=64, num_tasks=len(mt_targets)
    )

    print("多任务多尺度LSTM模型 ")

    # result
    plt.figure(figsize=(15, 8))
    plt.plot(results['LSTM']['actuals'], label='Actual', color='blue')
    plt.plot(results['LSTM']['predictions'], label='LSTM', color='red', linestyle='--')
    plt.plot(results['GRU']['predictions'], label='GRU', color='green', linestyle=':')
    plt.legend()
    plt.title("深度学习模型预测结果比较")
    plt.show()


if __name__ == '__main__':
    data_file_path = 'Steel_industry_data.csv'
    run_deep_learning_analysis(data_file_path)
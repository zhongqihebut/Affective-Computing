import jittor as jt
from jittor.dataset import Dataset, DataLoader
import numpy as np

# 假设你的数据集
data_m = np.random.rand(1959, 8, 48, 16, 1).astype(np.float32)  # 生成模拟数据

# 自定义 Dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]  # 取出单个样本

    def __len__(self):
        return len(self.data)

# 创建数据集实例
dataset = CustomDataset(data_m)

# 使用 Jittor 的 DataLoader 进行批量加载
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 遍历 DataLoader 获取每个 batch
for batch_idx, batch_data in enumerate(dataloader):
    print(f"Batch {batch_idx}: shape {batch_data.shape}")
    # 如果想访问具体的 batch 数据
    # batch_data.shape = (32, 8, 48, 16, 1)

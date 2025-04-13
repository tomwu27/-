# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ==================== 1. 数据加载与预处理 ====================
# 加载数据集（假设文件名为Superstore.csv）
file_path = r'C:\Users\28493\Desktop\Sample - Superstore.csv'
df = pd.read_csv(file_path, encoding='latin1')


# 转换为时间序列数据
df['Order Date'] = pd.to_datetime(df['Order Date'])
df = df.sort_values('Order Date').set_index('Order Date')

# 按周聚合销售额（示例目标列）
ts_sales = df['Sales'].resample('W').sum()

# 处理缺失值（用前向填充）
ts_sales = ts_sales.fillna(method='ffill')


# ==================== 2. 小波变换压缩 ====================
def wavelet_compress(data, wavelet='db4', level=5, mode='soft'):
    """小波压缩主函数"""
    # 小波分解
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # 计算通用阈值
    sigma = np.std(coeffs[-level])
    threshold = sigma * np.sqrt(2 * np.log(len(data)))

    # 阈值处理细节系数
    coeffs_thresh = [coeffs[0]]  # 保留近似系数
    for i in range(1, len(coeffs)):
        coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode=mode))

    # 重构信号
    compressed_data = pywt.waverec(coeffs_thresh, wavelet)

    # 计算压缩率
    original_size = len(data)
    compressed_size = len(coeffs_thresh[0]) + sum([len(c) // 2 for c in coeffs_thresh[1:]])
    compression_ratio = compressed_size / original_size

    return compressed_data[:len(data)], compression_ratio


# 执行压缩
ts_compressed, compression_ratio = wavelet_compress(ts_sales.values)

# ==================== 3. 结果可视化 ====================
plt.figure(figsize=(12, 6))
plt.plot(ts_sales.index, ts_sales.values, label='原始销售额', linewidth=1)
plt.plot(ts_sales.index, ts_compressed, 'r--', label='小波压缩后', linewidth=1.5)
plt.title(f'超市销售额小波压缩对比 (压缩率: {compression_ratio:.1%})')
plt.xlabel('日期')
plt.ylabel('销售额')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==================== 4. 效果评估 ====================
# 计算重建误差
mse = mean_squared_error(ts_sales, ts_compressed)
print(f'=== 评估结果 ===\n'
      f'原始数据长度: {len(ts_sales)}\n'
      f'压缩后有效数据量: {int(len(ts_sales) * compression_ratio)}\n'
      f'压缩率: {compression_ratio:.1%}\n'
      f'重建MSE: {mse:.2f}\n'
      f'信噪比(SNR): {10 * np.log10(np.var(ts_sales) / mse):.2f} dB')

# ==================== 5. 保存压缩结果 ====================
compressed_df = pd.DataFrame({
    'Date': ts_sales.index,
    'Original_Sales': ts_sales.values,
    'Compressed_Sales': ts_compressed
})
compressed_df.to_csv('compressed_sales.csv', index=False)
# Visualization.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def smooth_data(data, window_size=10):
    """使用移动平均平滑数据（保持相同长度）"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 加载训练数据
data = np.load('training_history.npz')

# 创建科学论文风格的图表
plt.figure(figsize=(12, 10))
gs = GridSpec(3, 2, figure=plt.gcf())
plt.rcParams.update({'font.size': 12, 'font.family': 'Times New Roman'})

# 计算有效数据范围
window = 20
valid_len = len(data['total_reward']) - window + 1
episodes_valid = data['episode'][:valid_len]

# 子图1：奖励曲线
ax1 = plt.subplot(gs[0, :])
smoothed_reward = smooth_data(data['total_reward'], window)
ax1.plot(episodes_valid, smoothed_reward, color='#2c7bb6', linewidth=2)
ax1.set_xlabel('Training Episode')
ax1.set_ylabel('Smoothed Reward')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_title('(a) Training Reward Trend', y=-0.3)

# 子图2：资源利用率
ax2 = plt.subplot(gs[1, 0])
smoothed_cpu = smooth_data(data['avg_cpu'], window)
smoothed_ram = smooth_data(data['avg_ram'], window)
ax2.plot(episodes_valid, smoothed_cpu, 
        label='CPU', color='#d7191c', linewidth=2)
ax2.plot(episodes_valid, smoothed_ram,
        label='RAM', color='#fdae61', linewidth=2, linestyle='--')
ax2.set_xlabel('Training Episode')
ax2.set_ylabel('Utilization Rate')
ax2.legend(framealpha=0.9)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_title('(b) Resource Utilization', y=-0.3)

# 子图3：系统资源
ax3 = plt.subplot(gs[1, 1])
smoothed_vms = smooth_data(data['avg_vms'], window)
ax3.plot(episodes_valid, smoothed_vms,
        color='#018571', linewidth=2)
ax3.set_xlabel('Training Episode')
ax3.set_ylabel('VM Count')
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.set_title('(c) VM Allocation', y=-0.3)

# 子图4：任务队列
ax4 = plt.subplot(gs[2, 0])
smoothed_queue = smooth_data(data['avg_queue'], window)
ax4.plot(episodes_valid, smoothed_queue,
        color='#7570b3', linewidth=2)
ax4.set_xlabel('Training Episode')
ax4.set_ylabel('Queue Length')
ax4.grid(True, linestyle='--', alpha=0.7)
ax4.set_title('(d) Task Queue', y=-0.3)

# 子图5：TD误差
ax5 = plt.subplot(gs[2, 1])
smoothed_td = smooth_data(data['td_errors'], window)
ax5.plot(episodes_valid, smoothed_td,
        color='#1a9850', linewidth=2)
ax5.set_xlabel('Training Episode')
ax5.set_ylabel('TD Error')
ax5.grid(True, linestyle='--', alpha=0.7)
ax5.set_title('(e) Learning Stability', y=-0.3)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# 保存为高分辨率图片
plt.savefig('Training_Dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
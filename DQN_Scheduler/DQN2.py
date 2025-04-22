# DQN2.py
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, Model
from collections import deque
import heapq

# ---------------------- 环境模型增强 ----------------------
class CloudTask:
    """任务对象定义"""
    def __init__(self, task_id, cpu_demand, ram_demand, duration):
        self.id = task_id
        self.cpu = cpu_demand    # 任务需要的CPU资源(0~1)
        self.ram = ram_demand    # 任务需要的RAM资源(0~1)
        self.duration = duration # 任务持续时间(时间步)
        self.remaining = duration

class CloudEdgeEnv:
    """强化学习环境-云边协同系统"""
    def __init__(self, max_vms=10, max_tasks=100):
        self.max_vms = max_vms
        self.num_vms = 1
        self.task_queue = deque()       # 等待队列
        self.running_tasks = []         # 执行中任务(最小堆)
        self.time_step = 0
        self.task_id = 0

        # 系统参数
        self.task_arrival_prob = 0.6    # 任务到达概率
        self.max_task_duration = 5      # 任务最大持续时间
    
    def _generate_task(self):
        """生成新任务"""
        if random.random() < self.task_arrival_prob:
            cpu = np.clip(np.random.normal(0.3, 0.15), 0.1, 0.9)
            ram = np.clip(np.random.normal(0.3, 0.15), 0.1, 0.9)
            duration = random.randint(1, self.max_task_duration)
            self.task_queue.append(CloudTask(self.task_id, cpu, ram, duration))
            self.task_id += 1
    #advanced       
    def _process_tasks(self):
        """处理当前任务"""
        #available_cpu = 1.0 / self.num_vms
        #available_ram = 1.0 / self.num_vms
        
        # ad改为每个VM独立资源池
        available_per_vm = {
            "cpu": 1.0,
            "ram": 1.0
        }
        available_resources = [available_per_vm.copy() for _ in range(self.num_vms)]

        
        # 处理执行中的任务
        completed = []
        for task in self.running_tasks:
            task.remaining -= 1
            if task.remaining <= 0:
                completed.append(task)
        for task in completed:
            self.running_tasks.remove(task)
            
        # 分配新任务
        '''while self.task_queue and len(self.running_tasks) < self.max_vms:
            task = self.task_queue.popleft()
            if task.cpu <= available_cpu and task.ram <= available_ram:
                self.running_tasks.append(task)
                available_cpu -= task.cpu
                available_ram -= task.ram
        '''
        #ad

        while self.task_queue and len(self.running_tasks) < self.max_vms:
            task = self.task_queue.popleft()
            allocated = False
            for vm_res in available_resources:
                if (vm_res["cpu"] >= task.cpu and 
                    vm_res["ram"] >= task.ram):
                    self.running_tasks.append(task)
                    vm_res["cpu"] -= task.cpu
                    vm_res["ram"] -= task.ram
                    allocated = True
                    break
            if not allocated:
                self.task_queue.appendleft(task)  # 放回队列前端
                break

    def reset(self):
        """重置环境状态"""
        self.num_vms = 1
        self.task_queue.clear()
        self.running_tasks = []
        self.time_step = 0
        self.task_id = 0
        return self._get_state()
    
    def step(self, action):
        """执行动作并返回新状态"""
        self.time_step += 1
        
        # 执行VM扩展动作
        if action == 1 and self.num_vms < self.max_vms:
            self.num_vms += 1
        elif action == 2 and self.num_vms > 1:
            self.num_vms -= 1
        
        # 任务生命周期管理
        self._generate_task()
        self._process_tasks()
        
        # 计算资源利用率
        cpu_usage = sum(t.cpu for t in self.running_tasks) / self.num_vms
        ram_usage = sum(t.ram for t in self.running_tasks) / self.num_vms
        
        # 奖励函数设计
        reward = self._calculate_reward(cpu_usage, ram_usage)
        
        done = (self.time_step >= 200)
        return self._get_state(), reward, done, {}

    def _get_state(self):
        """获取状态向量"""
        cpu = sum(t.cpu for t in self.running_tasks) / self.num_vms
        ram = sum(t.ram for t in self.running_tasks) / self.num_vms
        return np.array([cpu, ram, self.num_vms/self.max_vms, len(self.task_queue)/100])

    def _calculate_reward(self, cpu, ram):
        """多目标奖励函数"""
        # 资源利用率惩罚
        util_penalty = 0.5*(cpu + ram) 
        # 负载不均衡惩罚
        imbalance_penalty = 0.3*abs(cpu - ram)
        # SLA违规惩罚（假设CPU>90%算违规）
        sla_penalty = 2.0 if cpu > 0.9 else 0.0
        return -(util_penalty + imbalance_penalty + sla_penalty)

# ---------------------- 改进的DQN算法 ----------------------
class DuelingDQN(Model):
    """Dueling DQN网络结构"""
    def __init__(self, action_size):
        super(DuelingDQN, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.value = layers.Dense(1, activation='linear')
        self.advantage = layers.Dense(action_size, activation='linear')

    #add 显式声明输入形状
    def build(self, input_shape):
        # 显式定义输入层
        super(DuelingDQN, self).build(input_shape)
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        v = self.value(x)
        a = self.advantage(x)
        return v + (a - tf.reduce_mean(a, axis=1, keepdims=True))

class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def add(self, experience):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return []
            
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)
        
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

class AdvancedDQNAgent:
    """改进的DQN Agent"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update_freq = 100
        self.time_step = 0
        
        # 双重网络设计
        self.q_network = DuelingDQN(action_size)
        self.target_network = DuelingDQN(action_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)
        
        # 优先经验回放
        self.memory = PrioritizedReplayBuffer(capacity=100000)
        
        # 训练数据记录
        self.td_errors = []

    def act(self, state):
        """ε-greedy策略"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.numpy()[0])
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def train(self):
        """训练步骤"""
        if len(self.memory.buffer) < self.batch_size:
            return
            
        # 从优先回放中采样
        samples, indices, weights = self.memory.sample(self.batch_size)
        
        # 转换数据格式
        states = np.array([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        next_states = np.array([s[3] for s in samples])
        dones = np.array([s[4] for s in samples])
        
        # 计算目标Q值（Double DQN）
        future_q = self.target_network(next_states)
        target_q = rewards + (1 - dones) * self.gamma * tf.reduce_max(future_q, axis=1)
        
        # 计算当前Q值
        with tf.GradientTape() as tape:
            current_q = self.q_network(states)
            action_masks = tf.one_hot(actions, self.action_size)
            current_q = tf.reduce_sum(current_q * action_masks, axis=1)
            
            # 计算优先权（TD误差）
            td_errors = tf.abs(target_q - current_q)
            loss = tf.reduce_mean(weights * tf.square(target_q - current_q))
            
        # 更新网络参数
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
        
        # 更新优先权
        self.memory.update_priorities(indices, td_errors.numpy() + 1e-5)
        
        # 记录TD误差
        self.td_errors.extend(td_errors.numpy().tolist())
        
        # 定期更新目标网络
        self.time_step += 1
        if self.time_step % self.target_update_freq == 0:
            self.target_network.set_weights(self.q_network.get_weights())

# ---------------------- 训练流程 ----------------------
if __name__ == "__main__":
    env = CloudEdgeEnv(max_vms=10)
    agent = AdvancedDQNAgent(state_size=4, action_size=3)

    # 训练参数
    episodes = 1000
    render_interval = 50

    # 初始化数据记录
    history = {
        'episode': [],
        'total_reward': [],
        'avg_vms': [],
        'avg_cpu': [],
        'avg_ram': [],
        'avg_queue': [],
        'td_errors': []
    }

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_vms = []
        episode_cpu = []
        episode_ram = []
        episode_queue = []

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            # 记录当前状态的数据
            episode_vms.append(env.num_vms)
            episode_cpu.append(state[0])
            episode_ram.append(state[1])
            episode_queue.append(state[3] * 100)  # 恢复实际队列长度
            
            agent.memory.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            agent.train()

        # 计算平均指标
        avg_vms = np.mean(episode_vms)
        avg_cpu = np.mean(episode_cpu)
        avg_ram = np.mean(episode_ram)
        avg_queue = np.mean(episode_queue)

        # 记录到history
        history['episode'].append(ep)
        history['total_reward'].append(total_reward)
        history['avg_vms'].append(avg_vms)
        history['avg_cpu'].append(avg_cpu)
        history['avg_ram'].append(avg_ram)
        history['avg_queue'].append(avg_queue)
        history['td_errors'].append(np.mean(agent.td_errors[-100:]))  # 记录最近100步的平均TD误差
        agent.td_errors = []  # 清空当前记录

        # 更新探索率
        agent.update_epsilon()

        # 定期输出训练信息
        if ep % render_interval == 0:
            print(f"Episode: {ep}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
            print(f"Current VM numbers: {env.num_vms}, Number of queued tasks: {len(env.task_queue)}")

    # 训练完成后保存历史数据
    np.savez('training_history2.npz', **history)
    agent.q_network.save_weights('dqn_model_weights2.h5')
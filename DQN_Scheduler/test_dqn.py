import unittest
import numpy as np
from DQN2 import CloudEdgeEnv, CloudTask, AdvancedDQNAgent

class TestCloudEdgeEnv(unittest.TestCase):
    def setUp(self):
        self.env = CloudEdgeEnv(max_vms=5, max_tasks=20)
        self.env.reset()
        
    def test_initial_state(self):
        """Test environment initialization status"""
        state = self.env.reset()
        self.assertEqual(self.env.num_vms, 1)
        self.assertEqual(len(self.env.task_queue), 0)
        self.assertEqual(len(self.env.running_tasks), 0)
        self.assertTrue(np.allclose(state, [0, 0, 0.2, 0]))  # [cpu, ram, vms/max_vms, queue/max_tasks]

    def test_vm_scaling_actions(self):
        """Test VM extension action boundaries"""
        # 测试增加到最大VM数量
        for _ in range(6):
            self.env.step(1)  # 尝试增加VM
        self.assertEqual(self.env.num_vms, 5)  # max_vms=5
        
        # 测试减少到最小VM数量
        for _ in range(6):
            self.env.step(2)  # 尝试减少VM
        self.assertEqual(self.env.num_vms, 1)

    def test_task_processing(self):
        """Test task processing logic"""
        # 手动添加3个任务
        task1 = CloudTask(1, 0.2, 0.3, 2)
        task2 = CloudTask(2, 0.3, 0.2, 2)
        task3 = CloudTask(3, 0.5, 0.5, 2)
        self.env.task_queue.extend([task1, task2, task3])
        
        # 处理任务（当前1个VM，可用资源1.0）
        self.env._process_tasks()
        
        # 验证任务分配
        self.assertEqual(len(self.env.running_tasks), 2)  # 前两个任务资源总和0.5 < 1.0
        self.assertEqual(len(self.env.task_queue), 1)    # 第三个任务留在队列

    def test_resource_utilization(self):
        """Test resource utilization calculation"""
        # 添加两个任务到运行队列
        task1 = CloudTask(1, 0.4, 0.3, 2)
        task2 = CloudTask(2, 0.3, 0.4, 2)
        self.env.running_tasks = [task1, task2]
        self.env.num_vms = 2  # 每个VM可用0.5资源
        
        # 计算利用率
        cpu_usage = sum(t.cpu for t in self.env.running_tasks) / self.env.num_vms
        ram_usage = sum(t.ram for t in self.env.running_tasks) / self.env.num_vms
        
        self.assertAlmostEqual(cpu_usage, (0.4+0.3)/2)  # 0.35
        self.assertAlmostEqual(ram_usage, (0.3+0.4)/2)  # 0.35

    def test_reward_calculation(self):
        """Test reward calculation logic"""
        # 正常情况
        reward1 = self.env._calculate_reward(0.6, 0.6)
        #self.assertAlmostEqual(reward1, -(0.6*0.5 + 0 + 0))  # -0.3
        self.assertAlmostEqual(reward1, -(0.5*(0.6+0.6)), delta=1e-6)
        
        # SLA违规
        reward2 = self.env._calculate_reward(0.95, 0.8)
        expected = -(0.5*(0.95+0.8) + 0.3*0.15 + 2.0)
        self.assertAlmostEqual(reward2, expected)
        
        # 负载不均衡
        reward3 = self.env._calculate_reward(0.8, 0.3)
        imbalance = 0.3*(0.8-0.3)
        self.assertAlmostEqual(reward3, -(0.5*1.1 + imbalance + 0))

class TestTrainedAgent(unittest.TestCase):
    def test_agent_decision(self):
        """Test the decision logic of the trained Agent"""
        env = CloudEdgeEnv(max_vms=5)
        agent = AdvancedDQNAgent(state_size=4, action_size=3)
        #agent.q_network.load_weights('dqn_model_weights.h5')  # 加载训练好的模型
         # 初始化网络权重（构建计算图）
        dummy_state = np.zeros((1, 4))  # 创建虚拟输入
        _ = agent.q_network(dummy_state)  # 触发网络构建
        
        agent.q_network.load_weights('dqn_model_weights.h5')  # 现在可以安全加载权重

        #ad 强制进入预测模式
        agent.epsilon = 0.0  # 完全禁用探索
        agent.noise_scale = 0.0
        
        # 模拟高负载场景
        high_load_state = np.array([0.95, 0.9, 0.2, 0.8])  # 高利用率，低VM数量
        action = agent.act(high_load_state)
        self.assertEqual(action, 1)  # 期望增加VM
        
        # 模拟低负载场景
        low_load_state = np.array([0.2, 0.3, 0.8, 0.1])    # 低利用率，高VM数量
        action = agent.act(low_load_state)
        self.assertEqual(action, 2)  # 期望减少VM

    def test_utilization_optimization(self):
        """Test the optimization effect of resource utilization rate"""
        env = CloudEdgeEnv(max_vms=5)
        agent = AdvancedDQNAgent(state_size=4, action_size=3)
        #agent.q_network.load_weights('dqn_model_weights.h5')
         # 初始化网络权重（构建计算图）
        dummy_state = np.zeros((1, 4))  # 创建虚拟输入
        _ = agent.q_network(dummy_state)  # 触发网络构建
        
        agent.q_network.load_weights('dqn_model_weights.h5')  # 现在可以安全加载权重
        
        state = env.reset()
        done = False
        cpu_usages = []
        ram_usages = []
        
        while not done:
            action = agent.act(state)
            next_state, _, done, _ = env.step(action)
            cpu_usages.append(next_state[0])
            ram_usages.append(next_state[1])
            state = next_state
        
        avg_cpu = np.mean(cpu_usages)
        avg_ram = np.mean(ram_usages)
        
        # 验证平均利用率在10%-90%的合理范围
        self.assertTrue(0.1 <= avg_cpu <= 0.9,f"Average CPU utilization{avg_cpu:.2f} exceeds the target range")
        self.assertTrue(0.1 <= avg_ram <= 0.9,f"Average RAM utilization{avg_ram:.2f} exceeds the target range")
        
        # 验证负载均衡（CPU和RAM差异小于20%）
        imbalance = np.mean(np.abs(np.array(cpu_usages) - np.array(ram_usages)))
        self.assertLess(imbalance, 0.2,f"Resource imbalance{imbalance:.2f} is excessive(Should be<0.2)")
                       
        # 打印详细数据
        print(f"\nResource optimization test results:")
        print(f"CPU average: {avg_cpu:.2f}  RAM average: {avg_ram:.2f}")
        print(f"Imbalance degree: {imbalance:.2f}")

if __name__ == '__main__':
    unittest.main()
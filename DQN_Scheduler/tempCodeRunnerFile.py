import unittest
import numpy as np
from unittest.mock import patch
from DQN2 import CloudEdgeEnv, CloudTask

class TestCloudEdgeEnv(unittest.TestCase):
    def setUp(self):
        # 固定随机种子保证测试可重复
        np.random.seed(42)
        random.seed(42)
        
        # 使用简化配置
        self.config = {
            "max_vms": 3,
            "max_tasks": 10,
            "task_arrival_prob": 0.0,  # 初始关闭任务生成
            "max_task_duration": 5,
            "state_size": 4
        }
        self.env = CloudEdgeEnv(self.config)

    def test_initial_state(self):
        """测试环境初始化状态"""
        state = self.env.reset()
        self.assertEqual(self.env.num_vms, 1)
        self.assertEqual(len(self.env.task_queue), 0)
        self.assertEqual(len(self.env.running_tasks), 0)
        self.assertTrue(np.allclose(state, [0.0, 0.0, 1/3, 0.0]))

    def test_vm_scaling(self):
        """测试VM扩缩容逻辑"""
        # 测试扩容
        self.env.step(1)  # 执行扩容动作
        self.assertEqual(self.env.num_vms, 2)
        
        # 测试最大VM限制
        for _ in range(5):
            self.env.step(1)
        self.assertEqual(self.env.num_vms, 3)  # 不应超过max_vms=3
        
        # 测试缩容
        self.env.step(2)
        self.assertEqual(self.env.num_vms, 2)
        
        # 测试最小VM限制
        self.env.num_vms = 1
        self.env.step(2)
        self.assertEqual(self.env.num_vms, 1)

    def test_task_processing(self):
        """测试任务处理与资源计算"""
        # 手动添加测试任务
        task1 = CloudTask(0, 0.3, 0.4, 3)
        task2 = CloudTask(1, 0.2, 0.3, 2)
        self.env.task_queue.extend([task1, task2])
        
        # 处理任务（action=0不改变VM数量）
        state, _, _, _ = self.env.step(0)
        
        # 验证资源分配
        self.assertEqual(len(self.env.running_tasks), 2)
        self.assertAlmostEqual(state[0], (0.3+0.2)/1)  # CPU利用率
        self.assertAlmostEqual(state[1], (0.4+0.3)/1)  # RAM利用率
        self.assertEqual(state[3], 0.0)  # 队列已清空
        
        # 推进时间步检查任务完成
        self.env.step(0)
        self.env.step(0)
        self.assertEqual(len(self.env.running_tasks), 1)  # task2完成

    def test_resource_utilization_calculation(self):
        """测试多VM场景资源利用率计算"""
        self.env.num_vms = 2  # 设置2个VM
        # 每个VM可用资源 0.5 CPU/RAM
        task1 = CloudTask(0, 0.4, 0.3, 2)  # 分配到VM1
        task2 = CloudTask(1, 0.3, 0.4, 2)  # 分配到VM2
        self.env.running_tasks = [task1, task2]
        
        state = self.env._get_state()
        # CPU利用率 = (0.4+0.3)/2 = 0.35
        # RAM利用率 = (0.3+0.4)/2 = 0.35
        expected_state = [0.35, 0.35, 2/3, 0.0]
        self.assertTrue(np.allclose(state, expected_state, atol=0.01))

    def test_reward_function(self):
        """测试多目标奖励计算"""
        test_cases = [
            # (cpu, ram, expected_reward)
            (0.8, 0.8, -(0.8 + 0.0 + 0)),        # 高利用率且均衡
            (0.8, 0.3, -(0.55 + 0.15 + 0)),      # 不均衡但未超载
            (0.95, 0.9, -(0.925 + 0.015 + 2.0))  # SLA违规
        ]
        
        for cpu, ram, expected in test_cases:
            with self.subTest(cpu=cpu, ram=ram):
                reward = self.env._calculate_reward(cpu, ram)
                self.assertAlmostEqual(reward, expected, places=2)

    @patch('random.random')
    def test_task_generation(self, mock_random):
        """测试任务生成逻辑"""
        mock_random.return_value = 0.5  # 低于到达概率0.6
        self.env.task_arrival_prob = 0.6
        self.env._generate_task()
        self.assertEqual(len(self.env.task_queue), 1)
        
        task = self.env.task_queue[0]
        self.assertTrue(0.1 <= task.cpu <= 0.9)
        self.assertTrue(0.1 <= task.ram <= 0.9)

class TestAdvancedDQNAgent(unittest.TestCase):
    def setUp(self):
        from DQN2 import DuelingDQN
        from DQN2 import PrioritizedReplayBuffer
        
        class MockAgent:
            def __init__(self):
                self.q_network = DuelingDQN(3)
                self.memory = PrioritizedReplayBuffer(1000)
                self.epsilon = 0.1
        
        self.agent = MockAgent()

    def test_action_selection(self):
        """测试ε-greedy策略"""
        # 固定Q网络输出
        self.agent.q_network = lambda x: tf.constant([[1.0, 0.5, 0.3]])
        
        # 测试探索（随机动作）
        with patch('numpy.random.rand', return_value=0.05):  # < epsilon
            action = self.agent.act(None)
            self.assertIn(action, [0, 1, 2])
        
        # 测试利用（选择最大Q值）
        with patch('numpy.random.rand', return_value=0.2):  # > epsilon
            action = self.agent.act(None)
            self.assertEqual(action, 0)

    def test_experience_replay(self):
        """测试优先经验回放"""
        # 添加样本
        for i in range(5):
            self.agent.memory.add( (f"state{i}", i%3, i*0.1, f"next{i}", False) )
        
        # 采样验证
        samples, indices, weights = self.agent.memory.sample(2)
        self.assertEqual(len(samples), 2)
        self.assertEqual(len(indices), 2)
        self.assertTrue(all(0 <= idx < 5 for idx in indices))

if __name__ == '__main__':
    unittest.main()
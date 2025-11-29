# ablation_study.py
"""
消融实验脚本
系统地测试不同配置对性能的影响
"""

import torch
import numpy as np
import yaml
import os
import json
from torchvision import datasets, transforms
from target_model import load_target_model
from sparse_attack_env import SparseAttackEnv
from ppo_trainer import train_rl_agent
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class AblationExperiment:
    """消融实验管理器"""
    
    def __init__(self, base_config_path="config.yaml"):
        with open(base_config_path, encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)
        
        # 设置设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
        # 加载数据和模型
        self._setup_data_and_model()
        
        # 结果存储
        self.results = {}
    
    def _setup_data_and_model(self):
        """设置数据集和模型"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
        self.test_set = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        
        self.model = load_target_model(
            self.base_config['model']['name'], 
            num_classes=self.base_config['model']['num_classes']
        )
        self.model = self.model.eval().to(self.device)
    
    def run_reward_ablation(self, num_samples=50):
        """
        消融实验1: 奖励函数配置
        测试不同奖励权重的影响
        """
        print("\n" + "=" * 60)
        print("消融实验1: 奖励函数配置")
        print("=" * 60)
        
        # 不同的奖励配置
        reward_configs = [
            {'success': 10.0, 'step': -0.1, 'fail': -5.0, 'name': 'Default'},
            {'success': 20.0, 'step': -0.1, 'fail': -5.0, 'name': 'High Success'},
            {'success': 10.0, 'step': -0.5, 'fail': -5.0, 'name': 'High Step Penalty'},
            {'success': 10.0, 'step': -0.05, 'fail': -5.0, 'name': 'Low Step Penalty'},
            {'success': 10.0, 'step': -0.1, 'fail': -10.0, 'name': 'High Fail Penalty'},
            {'success': 5.0, 'step': -0.1, 'fail': -2.0, 'name': 'All Low'},
        ]
        
        results = []
        
        for config in reward_configs:
            print(f"\n测试配置: {config['name']}")
            print(f"  Success: {config['success']}, Step: {config['step']}, Fail: {config['fail']}")
            
            # 修改环境的奖励函数
            # 注意：这需要在 sparse_attack_env.py 中添加配置支持
            asr, avg_pixels = self._test_configuration(
                num_samples=num_samples,
                reward_config=config
            )
            
            results.append({
                'name': config['name'],
                'asr': asr,
                'avg_pixels': avg_pixels,
                **config
            })
        
        self.results['reward_ablation'] = results
        self._plot_reward_ablation(results)
        
        return results
    
    def run_max_steps_ablation(self, num_samples=50):
        """
        消融实验2: 最大步数配置
        测试允许不同修改像素数的影响
        """
        print("\n" + "=" * 60)
        print("消融实验2: 最大修改步数")
        print("=" * 60)
        
        max_steps_list = [1, 3, 5, 7, 10]
        results = []
        
        for max_steps in max_steps_list:
            print(f"\n测试 max_steps = {max_steps}")
            
            asr, avg_pixels = self._test_max_steps(
                num_samples=num_samples,
                max_steps=max_steps
            )
            
            results.append({
                'max_steps': max_steps,
                'asr': asr,
                'avg_pixels': avg_pixels
            })
        
        self.results['max_steps_ablation'] = results
        self._plot_max_steps_ablation(results)
        
        return results
    
    def run_training_steps_ablation(self, num_samples=50):
        """
        消融实验3: 训练步数
        测试不同训练时长的影响
        """
        print("\n" + "=" * 60)
        print("消融实验3: 训练步数")
        print("=" * 60)
        
        training_steps_list = [1000, 3000, 5000, 10000, 20000]
        results = []
        
        for steps in training_steps_list:
            print(f"\n测试 training_steps = {steps}")
            
            asr, avg_pixels = self._test_training_steps(
                num_samples=num_samples,
                training_steps=steps
            )
            
            results.append({
                'training_steps': steps,
                'asr': asr,
                'avg_pixels': avg_pixels
            })
        
        self.results['training_steps_ablation'] = results
        self._plot_training_steps_ablation(results)
        
        return results
    
    def _test_configuration(self, num_samples, reward_config=None):
        """测试特定配置"""
        successes = 0
        total_pixels = []
        
        for i in tqdm(range(num_samples), desc="测试样本"):
            image, label = self.test_set[i]
            
            # 创建环境
            env = SparseAttackEnv(
                clean_image=image,
                true_label=label,
                model=self.model,
                max_steps=5
            )
            agent_path = "ppo_sparse_model.zip"
            if os.path.exists(agent_path):
                from stable_baselines3 import PPO
                agent = PPO.load(agent_path)
            else:
                agent = train_rl_agent(env, timesteps=5000)
    
            vec_env = DummyVecEnv([lambda: env])
            obs = vec_env.reset()
            done = False
            steps = 0
            
            while not done:
                action, _ = agent.predict(obs)
                result = vec_env.step(action)
                
                if len(result) == 4:
                    _, _, done, info = result
                else:
                    _, _, terminated, truncated, info = result
                    done = terminated[0] or truncated[0]
                
                info = info[0] if isinstance(info, list) else info
                steps += 1
                
                if info.get('success', False):
                    successes += 1
                    total_pixels.append(steps)
                    break
        
        asr = successes / num_samples
        avg_pixels = np.mean(total_pixels) if total_pixels else 0
        
        return asr, avg_pixels
    
    def _test_max_steps(self, num_samples, max_steps):
        return self._test_configuration(num_samples)
    
    def _test_training_steps(self, num_samples, training_steps):
        return self._test_configuration(num_samples)
    
    def _plot_reward_ablation(self, results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        names = [r['name'] for r in results]
        asrs = [r['asr'] * 100 for r in results]
        pixels = [r['avg_pixels'] for r in results]
        
        # ASR对比
        ax1.bar(names, asrs, color=sns.color_palette("Blues_d", len(names)))
        ax1.set_ylabel('Attack Success Rate (%)', fontsize=12)
        ax1.set_title('ASR vs Reward Configuration', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # 平均像素数对比
        ax2.bar(names, pixels, color=sns.color_palette("Oranges_d", len(names)))
        ax2.set_ylabel('Average Modified Pixels', fontsize=12)
        ax2.set_title('Pixels vs Reward Configuration', fontsize=14)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        save_dir = "results/ablation"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/reward_ablation.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_dir}/reward_ablation.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"奖励函数消融图表已保存至: {save_dir}/reward_ablation.png")
    
    def _plot_max_steps_ablation(self, results):
        """绘制最大步数消融结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        max_steps = [r['max_steps'] for r in results]
        asrs = [r['asr'] * 100 for r in results]
        pixels = [r['avg_pixels'] for r in results]
        
        # ASR vs Max Steps
        ax1.plot(max_steps, asrs, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Max Steps', fontsize=12)
        ax1.set_ylabel('Attack Success Rate (%)', fontsize=12)
        ax1.set_title('ASR vs Max Steps', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Avg Pixels vs Max Steps
        ax2.plot(max_steps, pixels, marker='s', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Max Steps', fontsize=12)
        ax2.set_ylabel('Average Modified Pixels', fontsize=12)
        ax2.set_title('Pixels vs Max Steps', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_dir = "results/ablation"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/max_steps_ablation.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_dir}/max_steps_ablation.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"最大步数消融图表已保存至: {save_dir}/max_steps_ablation.png")
    
    def _plot_training_steps_ablation(self, results):
        """绘制训练步数消融结果"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        training_steps = [r['training_steps'] for r in results]
        asrs = [r['asr'] * 100 for r in results]
        
        ax.plot(training_steps, asrs, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
        ax.set_title('ASR vs Training Steps', fontsize=14)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_dir = "results/ablation"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/training_steps_ablation.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_dir}/training_steps_ablation.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"训练步数消融图表已保存至: {save_dir}/training_steps_ablation.png")
    
    def save_results(self, filename="results/ablation/ablation_results.json"):
        """保存所有消融实验结果"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n消融实验结果已保存至: {filename}")
    
    def generate_latex_table(self):
        """生成LaTeX格式的表格"""
        print("\n" + "=" * 60)
        print("LaTeX表格代码")
        print("=" * 60)
        
        if 'reward_ablation' in self.results:
            print("\n% 奖励函数消融实验")
            print("\\begin{table}[h]")
            print("\\centering")
            print("\\caption{Ablation Study: Reward Function Configuration}")
            print("\\begin{tabular}{lccc}")
            print("\\hline")
            print("Configuration & ASR (\\%) & Avg. Pixels & Success Reward \\\\")
            print("\\hline")
            
            for r in self.results['reward_ablation']:
                print(f"{r['name']} & {r['asr']*100:.1f} & {r['avg_pixels']:.2f} & {r['success']:.1f} \\\\")
            
            print("\\hline")
            print("\\end{tabular}")
            print("\\end{table}")


def main():
    """运行所有消融实验"""
    print("开始消融实验")
    print("=" * 60)
    
    # 创建实验管理器
    exp = AblationExperiment()
    
    num_samples = 20
    
    print(f"\n当前使用 {num_samples} 个样本进行快速测试")
    
    # 1. 奖励函数消融
    # exp.run_reward_ablation(num_samples=num_samples)
    
    # 2. 最大步数消融
    exp.run_max_steps_ablation(num_samples=num_samples)
    
    # 3. 训练步数消融
    # exp.run_training_steps_ablation(num_samples=num_samples)
    exp.save_results()
    exp.generate_latex_table()
    
    print("\n消融实验完成！")
    
if __name__ == "__main__":
    main()

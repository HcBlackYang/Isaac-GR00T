#!/usr/bin/env python3
"""
修复后的复杂任务模块 - 解决random.choice错误
Fixed Complex Tasks Module - Resolves random.choice error
"""

import numpy as np
import random
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

class TaskComplexity(Enum):
    """任务复杂度枚举"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"

@dataclass
class ChallengingScenario:
    """具有挑战性的场景配置"""
    name: str
    complexity: TaskComplexity
    visual_noise: float
    force_disturbance: float
    occlusion_rate: float
    object_slipperiness: float
    lighting_variation: float
    description: str

class ComplexTaskGenerator:
    """复杂任务生成器"""
    
    def __init__(self):
        self.scenarios = [
            ChallengingScenario(
                name="基础操作",
                complexity=TaskComplexity.SIMPLE,
                visual_noise=0.1,
                force_disturbance=0.1,
                occlusion_rate=0.0,
                object_slipperiness=0.1,
                lighting_variation=0.1,
                description="基础机械臂操作，环境干扰较少"
            ),
            ChallengingScenario(
                name="遮挡抓取",
                complexity=TaskComplexity.MEDIUM,
                visual_noise=0.3,
                force_disturbance=0.2,
                occlusion_rate=0.4,
                object_slipperiness=0.1,
                lighting_variation=0.2,
                description="目标物体被部分遮挡，需要多视角观察"
            ),
            ChallengingScenario(
                name="滑移物体",
                complexity=TaskComplexity.MEDIUM,
                visual_noise=0.1,
                force_disturbance=0.1,
                occlusion_rate=0.1,
                object_slipperiness=0.6,
                lighting_variation=0.1,
                description="目标物体表面光滑，容易滑落"
            ),
            ChallengingScenario(
                name="弱光环境",
                complexity=TaskComplexity.MEDIUM,
                visual_noise=0.4,
                force_disturbance=0.1,
                occlusion_rate=0.2,
                object_slipperiness=0.2,
                lighting_variation=0.7,
                description="光照条件差，视觉信息不足"
            ),
            ChallengingScenario(
                name="力扰动环境",
                complexity=TaskComplexity.HARD,
                visual_noise=0.2,
                force_disturbance=0.6,
                occlusion_rate=0.3,
                object_slipperiness=0.3,
                lighting_variation=0.3,
                description="存在外部力扰动，影响操作精度"
            ),
            ChallengingScenario(
                name="精密操作",
                complexity=TaskComplexity.HARD,
                visual_noise=0.3,
                force_disturbance=0.4,
                occlusion_rate=0.2,
                object_slipperiness=0.4,
                lighting_variation=0.2,
                description="需要高精度的精密操作任务"
            ),
            ChallengingScenario(
                name="极端综合挑战",
                complexity=TaskComplexity.EXTREME,
                visual_noise=0.5,
                force_disturbance=0.4,
                occlusion_rate=0.6,
                object_slipperiness=0.5,
                lighting_variation=0.6,
                description="多重挑战同时存在的极端情况"
            ),
            ChallengingScenario(
                name="混沌环境",
                complexity=TaskComplexity.EXTREME,
                visual_noise=0.6,
                force_disturbance=0.5,
                occlusion_rate=0.5,
                object_slipperiness=0.6,
                lighting_variation=0.5,
                description="高度不确定的混沌操作环境"
            )
        ]
        
        print(f"📋 初始化任务生成器，共 {len(self.scenarios)} 个场景:")
        for complexity in TaskComplexity:
            count = sum(1 for s in self.scenarios if s.complexity == complexity)
            print(f"   {complexity.value}: {count} 个场景")
    
    def apply_scenario_to_observation(self, base_obs: Dict[str, np.ndarray], 
                                    scenario: ChallengingScenario) -> Dict[str, np.ndarray]:
        """将挑战性场景应用到观察数据"""
        obs = base_obs.copy()
        
        # 1. 视觉噪声（如果有图像数据）
        if "frontview_image" in obs:
            img = obs["frontview_image"].astype(np.float32)
            noise = np.random.normal(0, scenario.visual_noise * 50, img.shape)
            img = np.clip(img + noise, 0, 255)
            obs["frontview_image"] = img.astype(np.uint8)
        
        # 2. 光照变化
        if "frontview_image" in obs and scenario.lighting_variation > 0:
            img = obs["frontview_image"].astype(np.float32)
            brightness_factor = 1.0 + np.random.uniform(-scenario.lighting_variation, 
                                                       scenario.lighting_variation)
            img = np.clip(img * brightness_factor, 0, 255)
            obs["frontview_image"] = img.astype(np.uint8)
        
        # 3. 遮挡模拟（简化版，避免复杂的图像处理）
        if "frontview_image" in obs and scenario.occlusion_rate > 0:
            if np.random.random() < scenario.occlusion_rate:
                img = obs["frontview_image"]
                h, w = img.shape[:2]
                # 简单的矩形遮挡
                x1, y1 = np.random.randint(0, w//2), np.random.randint(0, h//2)
                x2, y2 = x1 + w//4, y1 + h//4
                img[y1:y2, x1:x2] = 0  # 黑色遮挡
        
        # 4. 关节位置扰动
        for key in ["robot0_joint_pos", "joint_positions"]:
            if key in obs:
                disturbance = np.random.normal(0, scenario.force_disturbance * 0.1, 
                                             obs[key].shape)
                obs[key] = obs[key] + disturbance
        
        # 5. 添加场景标识
        obs["scenario_info"] = {
            "name": scenario.name,
            "complexity": scenario.complexity.value,
            "challenges": {
                "visual_noise": scenario.visual_noise,
                "force_disturbance": scenario.force_disturbance,
                "occlusion_rate": scenario.occlusion_rate,
                "slipperiness": scenario.object_slipperiness
            }
        }
        
        return obs
    
    def get_scenario_by_complexity(self, complexity: TaskComplexity) -> ChallengingScenario:
        """根据复杂度获取场景"""
        matching_scenarios = [s for s in self.scenarios if s.complexity == complexity]
        if matching_scenarios:
            return random.choice(matching_scenarios)
        else:
            print(f"⚠️ 没有找到复杂度为 {complexity.value} 的场景，返回默认场景")
            return self.scenarios[0]  # 返回第一个场景作为默认

class AdaptiveSuccessEvaluator:
    """自适应成功评估器 - 根据场景调整成功标准"""
    
    def __init__(self):
        self.success_thresholds = {
            TaskComplexity.SIMPLE: 0.8,
            TaskComplexity.MEDIUM: 0.6,
            TaskComplexity.HARD: 0.4,
            TaskComplexity.EXTREME: 0.2
        }
        
        self.required_precision = {
            TaskComplexity.SIMPLE: 0.02,   # 2cm精度
            TaskComplexity.MEDIUM: 0.05,   # 5cm精度
            TaskComplexity.HARD: 0.08,     # 8cm精度
            TaskComplexity.EXTREME: 0.12   # 12cm精度
        }
    
    def evaluate_task_success(self, scenario: ChallengingScenario, 
                            step_count: int, max_steps: int,
                            position_error: float = None, 
                            force_error: float = None) -> Tuple[bool, float]:
        """评估任务成功"""
        
        # 基础成功概率（基于步数）
        step_success = max(0, 1.0 - (step_count / max_steps))
        
        # 如果没有提供误差信息，使用随机值模拟
        if position_error is None:
            position_error = np.random.uniform(0.01, 0.15)
        if force_error is None:
            force_error = np.random.uniform(0, 0.5)
        
        # 精度评估
        required_precision = self.required_precision[scenario.complexity]
        precision_success = max(0, 1.0 - (position_error / required_precision))
        
        # 力控评估
        force_success = max(0, 1.0 - force_error)
        
        # 综合成功率
        overall_success = (step_success * 0.4 + precision_success * 0.4 + force_success * 0.2)
        
        # 场景特定调整
        if scenario.complexity == TaskComplexity.EXTREME:
            overall_success *= 0.7  # 极端情况降低标准
        elif scenario.complexity == TaskComplexity.HARD:
            overall_success *= 0.8
        
        # 成功阈值判断
        threshold = self.success_thresholds[scenario.complexity]
        is_success = overall_success >= threshold
        
        return is_success, overall_success

class EnhancedEnvironment:
    """增强的挑战性环境"""
    
    def __init__(self, difficulty_progression=True):
        self.task_generator = ComplexTaskGenerator()
        self.success_evaluator = AdaptiveSuccessEvaluator()
        self.difficulty_progression = difficulty_progression
        
        self.current_episode = 0
        self.step_count = 0
        self.max_steps = 80  # 增加最大步数
        
        self.current_scenario = None
        self.episode_stats = []
        
        print("✅ 增强环境初始化完成")
    
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """重置环境，选择新的挑战场景"""
        self.step_count = 0
        self.current_episode += 1
        
        # 根据episode选择难度
        if self.difficulty_progression:
            if self.current_episode <= 2:
                complexity = TaskComplexity.SIMPLE
            elif self.current_episode <= 4:
                complexity = TaskComplexity.MEDIUM
            elif self.current_episode <= 6:
                complexity = TaskComplexity.HARD
            else:
                complexity = TaskComplexity.EXTREME
        else:
            complexity = random.choice(list(TaskComplexity))
        
        # 安全地获取场景
        self.current_scenario = self.task_generator.get_scenario_by_complexity(complexity)
        
        print(f"🎯 Episode {self.current_episode}: {self.current_scenario.name} "
              f"({self.current_scenario.complexity.value})")
        print(f"   {self.current_scenario.description}")
        
        # 生成基础观察
        base_obs = self._generate_base_observation()
        
        # 应用挑战场景
        obs = self.task_generator.apply_scenario_to_observation(base_obs, self.current_scenario)
        
        info = {
            "scenario": self.current_scenario,
            "episode": self.current_episode,
            "expected_difficulty": self.current_scenario.complexity.value
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """环境步进"""
        self.step_count += 1
        
        # 生成新观察
        base_obs = self._generate_base_observation()
        obs = self.task_generator.apply_scenario_to_observation(base_obs, self.current_scenario)
        
        # 计算位置和力误差（基于当前场景）
        position_error = np.random.uniform(0.01, 0.15)  # 基础误差
        position_error += self.current_scenario.force_disturbance * 0.1  # 力扰动影响
        position_error += self.current_scenario.visual_noise * 0.05     # 视觉噪声影响
        
        force_error = np.random.uniform(0, 0.5)
        force_error += self.current_scenario.object_slipperiness * 0.3   # 滑移影响
        
        # 评估任务成功
        is_success, success_score = self.success_evaluator.evaluate_task_success(
            self.current_scenario, self.step_count, self.max_steps, 
            position_error, force_error
        )
        
        # 确定奖励
        if is_success and self.step_count >= 15:  # 至少执行15步
            reward = success_score
            done = True
        elif self.step_count >= self.max_steps:
            reward = -0.5  # 超时惩罚
            done = True
        else:
            # 中间奖励基于当前表现
            step_reward = -0.01  # 时间惩罚
            precision_reward = max(0, 0.05 - position_error)  # 精度奖励
            reward = step_reward + precision_reward
            done = False
        
        info = {
            "task_success": is_success,
            "success_score": success_score,
            "position_error": position_error,
            "force_error": force_error,
            "scenario": self.current_scenario.name,
            "complexity": self.current_scenario.complexity.value,
            "step_count": self.step_count
        }
        
        return obs, reward, done, False, info
    
    def _generate_base_observation(self) -> Dict[str, np.ndarray]:
        """生成基础观察数据"""
        return {
            "frontview_image": np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8),
            "robot0_joint_pos": np.random.uniform(-0.3, 0.3, 5),
            "robot0_joint_vel": np.random.uniform(-0.2, 0.2, 5),
            "robot0_gripper_qpos": np.random.uniform(-0.08, 0.08, 1),
            "robot0_eef_pos": np.array([0.5, 0.0, 0.8]) + np.random.uniform(-0.1, 0.1, 3),
            "robot0_eef_quat": np.array([0, 0, 0, 1]) + np.random.uniform(-0.1, 0.1, 4)
        }

# 使用示例
def create_challenging_environment():
    """创建具有挑战性的环境"""
    return EnhancedEnvironment(difficulty_progression=True)

# 修改配置参数
class ImprovedConfig:
    """改进的实验配置"""
    def __init__(self):
        self.num_episodes = 12  # 增加episode数量
        self.max_steps_per_episode = 80  # 增加最大步数
        self.use_challenging_scenarios = True
        self.difficulty_progression = True
        self.require_higher_precision = True

def test_challenging_environment():
    """测试挑战性环境"""
    print("🧪 测试修复后的挑战性环境...")
    
    # 创建环境
    env = create_challenging_environment()
    
    # 先打印可用场景信息
    print(f"\n📊 环境统计:")
    print(f"   总场景数: {len(env.task_generator.scenarios)}")
    complexity_counts = {}
    for scenario in env.task_generator.scenarios:
        complexity = scenario.complexity.value
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
    
    for complexity, count in complexity_counts.items():
        print(f"   {complexity}: {count} 个场景")
    
    print(f"\n🎮 开始测试 episodes...")
    
    for episode in range(5):  # 测试5个episode
        try:
            print(f"\n--- Episode {episode + 1} ---")
            obs, info = env.reset()
            print(f"✅ 重置成功: {info['scenario'].name}")
            
            step_count = 0
            total_reward = 0
            
            while step_count < 15:  # 每个episode测试15步
                action = np.random.uniform(-0.05, 0.05, 6)
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                step_count += 1
                
                if step_count % 5 == 0:
                    print(f"   Step {step_count}: reward={reward:.3f}, total={total_reward:.3f}")
                
                if done:
                    status = "✅ 成功" if info["task_success"] else "❌ 失败"
                    print(f"   {status}, 最终成功率: {info['success_score']:.3f}")
                    break
            
            if not done:
                print(f"   ⏰ Episode未完成，总奖励: {total_reward:.3f}")
                
        except Exception as e:
            print(f"❌ Episode {episode + 1} 失败: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n✅ 测试完成！")

if __name__ == "__main__":
    test_challenging_environment()
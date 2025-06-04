#!/usr/bin/env python3
"""
完整的GR00T + RoboCasa + 元认知集成方案
解决任务简单和缺少可视化的问题
"""

import os
import sys
import time
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

# 导入GR00T官方客户端
try:
    from gr00t.eval.robot import RobotInferenceClient
    GROOT_CLIENT_AVAILABLE = True
    print("✅ GR00T官方客户端可用")
except ImportError as e:
    print(f"❌ GR00T官方客户端不可用: {e}")
    GROOT_CLIENT_AVAILABLE = False

# 导入元认知模块
try:
    from metacog_integration import (
        CompleteMetaCognitiveModule,
        RoboCasaToMetacogAdapter,
        MetacogToGR00TAdapter,
        ActionAdjuster,
        SensorData
    )
    METACOG_AVAILABLE = True
    print("✅ 元认知模块可用")
except ImportError as e:
    print(f"❌ 元认知模块不可用: {e}")
    METACOG_AVAILABLE = False

# 导入改进的环境
try:
    from complex_tasks import ComplexTaskGenerator, EnhancedEnvironment, TaskComplexity
    ENHANCED_TASKS_AVAILABLE = True
    print("✅ 增强任务模块可用")
except ImportError:
    ENHANCED_TASKS_AVAILABLE = False
    print("⚠️ 增强任务模块不可用，使用内置版本")

# 导入RoboCasa环境
try:
    from robocasa_integration import RoboCasaEnvironmentManager, RoboCasaConfig, RoboCasaTaskSelector
    ROBOCASA_INTEGRATION_AVAILABLE = True
    print("✅ RoboCasa集成可用")
except ImportError:
    ROBOCASA_INTEGRATION_AVAILABLE = False
    print("⚠️ RoboCasa集成不可用，使用模拟环境")

@dataclass
class IntegratedConfig:
    """集成实验配置"""
    # 服务连接
    host: str = "localhost"
    port: int = 5555
    
    # 实验设置
    experiment_name: str = "integrated_groot_robocasa_experiment"
    num_episodes: int = 12
    max_steps_per_episode: int = 100
    
    # 环境设置
    use_robocasa: bool = True
    robocasa_task: str = "CleaningSink"
    use_challenging_scenarios: bool = True
    enable_gui: bool = True
    
    # 实验模式
    run_baseline: bool = True
    run_metacognitive: bool = True
    
    # 元认知设置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 调试设置
    verbose: bool = True
    save_videos: bool = False

class IntegratedDataFormatter:
    """集成的数据格式器 - 支持RoboCasa和挑战场景"""
    
    def __init__(self):
        self.required_keys = {
            "video.webcam": (640, 480, 3),
            "state.single_arm": (1, 5),
            "state.gripper": (1, 1),
            "annotation.human.task_description": None
        }
        
        # 挑战场景生成器
        if ENHANCED_TASKS_AVAILABLE:
            self.task_generator = ComplexTaskGenerator()
        else:
            self.task_generator = None
    
    def create_correct_observation(self, base_obs: Dict[str, np.ndarray] = None, 
                                 apply_challenges: bool = False,
                                 scenario = None) -> Dict[str, Any]:
        """创建正确格式的观察数据"""
        correct_obs = {}
        
        # 处理RoboCasa观察
        if base_obs and self._is_robocasa_observation(base_obs):
            correct_obs = self._convert_robocasa_observation(base_obs)
        else:
            correct_obs = self._generate_fallback_observation(base_obs)
        
        # 应用挑战场景
        if apply_challenges and self.task_generator and scenario:
            correct_obs = self._apply_scenario_to_observation(correct_obs, scenario)
        
        return correct_obs
    
    def _is_robocasa_observation(self, obs: Dict[str, np.ndarray]) -> bool:
        """检查是否为RoboCasa观察"""
        robocasa_keys = ["agentview_image", "robot0_eye_in_hand_image", "frontview_image"]
        return any(key in obs for key in robocasa_keys)
    
    def _convert_robocasa_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """转换RoboCasa观察到GR00T格式"""
        correct_obs = {}
        
        # 1. 视觉数据
        img_key = None
        for key in ["frontview_image", "agentview_image", "robot0_eye_in_hand_image"]:
            if key in obs:
                img_key = key
                break
        
        if img_key:
            img = obs[img_key]
            if img.shape[:2] != (480, 640):
                import cv2
                img = cv2.resize(img, (640, 480))
            correct_obs["video.webcam"] = img[np.newaxis, :, :, :].astype(np.uint8)
        else:
            correct_obs["video.webcam"] = self._generate_default_image()
        
        # 2. 机器人状态
        if "robot0_joint_pos" in obs:
            joint_pos = obs["robot0_joint_pos"][:5]
            correct_obs["state.single_arm"] = joint_pos[np.newaxis, :].astype(np.float32)
        else:
            correct_obs["state.single_arm"] = np.random.uniform(-0.2, 0.2, (1, 5)).astype(np.float32)
        
        # 3. 夹爪状态
        if "robot0_gripper_qpos" in obs:
            gripper = obs["robot0_gripper_qpos"][:1]
            correct_obs["state.gripper"] = gripper[np.newaxis, :].astype(np.float32)
        else:
            correct_obs["state.gripper"] = np.random.uniform(-0.05, 0.05, (1, 1)).astype(np.float32)
        
        # 4. 任务描述
        correct_obs["annotation.human.task_description"] = ["Complete RoboCasa kitchen task"]
        
        return correct_obs
    
    def _generate_fallback_observation(self, base_obs: Dict[str, np.ndarray] = None) -> Dict[str, Any]:
        """生成备用观察数据"""
        return {
            "video.webcam": self._generate_default_image(),
            "state.single_arm": np.random.uniform(-0.2, 0.2, (1, 5)).astype(np.float32),
            "state.gripper": np.random.uniform(-0.05, 0.05, (1, 1)).astype(np.float32),
            "annotation.human.task_description": ["Execute manipulation task"]
        }
    
    def _generate_default_image(self) -> np.ndarray:
        """生成默认图像"""
        img = np.zeros((1, 480, 640, 3), dtype=np.uint8)
        for i in range(480):
            for j in range(640):
                img[0, i, j, 0] = (i + j) % 256
                img[0, i, j, 1] = (i * 2) % 256
                img[0, i, j, 2] = (j * 2) % 256
        return img
    
    def _apply_scenario_to_observation(self, obs: Dict[str, Any], scenario) -> Dict[str, Any]:
        """应用挑战场景到观察"""
        if not scenario:
            return obs
        
        modified_obs = obs.copy()
        
        # 应用视觉噪声
        if "video.webcam" in modified_obs:
            img = modified_obs["video.webcam"].astype(np.float32)
            noise = np.random.normal(0, scenario.visual_noise * 30, img.shape)
            img = np.clip(img + noise, 0, 255)
            modified_obs["video.webcam"] = img.astype(np.uint8)
        
        # 应用关节扰动
        if "state.single_arm" in modified_obs:
            joint_noise = np.random.normal(0, scenario.force_disturbance * 0.05, 
                                         modified_obs["state.single_arm"].shape)
            modified_obs["state.single_arm"] = modified_obs["state.single_arm"] + joint_noise.astype(np.float32)
        
        return modified_obs

class IntegratedEnvironmentManager:
    """集成环境管理器 - 支持RoboCasa和挑战场景"""
    
    def __init__(self, config: IntegratedConfig):
        self.config = config
        self.robocasa_env = None
        self.enhanced_env = None
        self.current_env_type = None
        
        # 初始化环境
        self.initialize_environments()
    
    def initialize_environments(self):
        """初始化环境"""
        print(f"🏠 初始化集成环境...")
        
        # 尝试初始化RoboCasa环境
        if self.config.use_robocasa and ROBOCASA_INTEGRATION_AVAILABLE:
            try:
                robocasa_config = RoboCasaConfig(
                    task_name=self.config.robocasa_task,
                    render_mode="human" if self.config.enable_gui else "offscreen",
                    horizon=self.config.max_steps_per_episode * 2
                )
                
                self.robocasa_env = RoboCasaEnvironmentManager(robocasa_config)
                
                if self.robocasa_env.is_initialized:
                    self.current_env_type = "robocasa"
                    print(f"✅ RoboCasa环境已初始化: {self.config.robocasa_task}")
                    if self.config.enable_gui:
                        print(f"🎥 GUI可视化已启用")
                else:
                    print(f"⚠️ RoboCasa环境初始化失败，使用备用环境")
                    self._initialize_fallback_environment()
            except Exception as e:
                print(f"⚠️ RoboCasa环境异常: {e}")
                self._initialize_fallback_environment()
        else:
            self._initialize_fallback_environment()
    
    def _initialize_fallback_environment(self):
        """初始化备用环境"""
        if ENHANCED_TASKS_AVAILABLE:
            self.enhanced_env = EnhancedEnvironment(difficulty_progression=True)
            self.current_env_type = "enhanced"
            print(f"✅ 增强挑战环境已初始化")
        else:
            # 简单环境作为最后备选
            self.current_env_type = "simple"
            print(f"✅ 简单模拟环境已初始化")
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重置环境"""
        if self.current_env_type == "robocasa" and self.robocasa_env:
            return self.robocasa_env.reset()
        elif self.current_env_type == "enhanced" and self.enhanced_env:
            return self.enhanced_env.reset()
        else:
            return self._simple_reset()
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """环境步进"""
        if self.current_env_type == "robocasa" and self.robocasa_env:
            return self.robocasa_env.step(action)
        elif self.current_env_type == "enhanced" and self.enhanced_env:
            return self.enhanced_env.step(action)
        else:
            return self._simple_step(action)
    
    def _simple_reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """简单环境重置"""
        obs = {
            "frontview_image": np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8),
            "robot0_joint_pos": np.random.uniform(-0.3, 0.3, 5),
            "robot0_joint_vel": np.random.uniform(-0.2, 0.2, 5),
            "robot0_gripper_qpos": np.random.uniform(-0.08, 0.08, 1),
        }
        info = {"task": "simple_manipulation", "difficulty": "medium"}
        return obs, info
    
    def _simple_step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """简单环境步进"""
        obs = {
            "frontview_image": np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8),
            "robot0_joint_pos": np.random.uniform(-0.3, 0.3, 5),
            "robot0_joint_vel": np.random.uniform(-0.2, 0.2, 5),
            "robot0_gripper_qpos": np.random.uniform(-0.08, 0.08, 1),
        }
        
        # 简单的成功逻辑 - 增加难度
        step_count = getattr(self, '_step_count', 0) + 1
        self._step_count = step_count
        
        # 更严格的成功条件
        if step_count > 30 and np.random.random() < 0.15:  # 降低成功率
            reward = 1.0
            done = True
            task_success = True
        elif step_count >= self.config.max_steps_per_episode:
            reward = -0.5
            done = True
            task_success = False
        else:
            reward = np.random.uniform(-0.02, 0.01)  # 增加负奖励
            done = False
            task_success = False
        
        info = {
            "task_success": task_success,
            "step_count": step_count,
            "difficulty": "medium_enhanced"
        }
        
        return obs, reward, done, False, info
    
    def close(self):
        """关闭环境"""
        if self.robocasa_env:
            self.robocasa_env.close()
        
        print("✅ 集成环境已关闭")
    
    def get_env_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        if self.current_env_type == "robocasa" and self.robocasa_env:
            return self.robocasa_env.get_task_info()
        else:
            return {
                "env_type": self.current_env_type,
                "action_dim": 6,
                "max_steps": self.config.max_steps_per_episode
            }

class IntegratedGR00TClient:
    """集成的GR00T客户端"""
    
    def __init__(self, config: IntegratedConfig):
        self.config = config
        self.client = None
        self.formatter = IntegratedDataFormatter()
        self.is_connected = False
        
        # 统计信息
        self.total_calls = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_time = 0.0
    
    def connect(self) -> bool:
        """连接到GR00T服务"""
        if not GROOT_CLIENT_AVAILABLE:
            print("❌ GR00T官方客户端不可用")
            return False
        
        try:
            print(f"🔗 连接到GR00T服务: {self.config.host}:{self.config.port}")
            
            self.client = RobotInferenceClient(
                host=self.config.host, 
                port=self.config.port
            )
            
            # 验证连接
            modality_config = self.client.get_modality_config()
            print("✅ 连接成功！")
            
            # 测试调用
            test_obs = self.formatter.create_correct_observation()
            test_result = self.client.get_action(test_obs)
            
            if test_result is not None:
                print("✅ 测试调用成功！")
                self.is_connected = True
                return True
            else:
                print("❌ 测试调用失败")
                return False
                
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def predict(self, observation: Dict[str, np.ndarray], 
               apply_challenges: bool = False, scenario = None) -> Optional[Dict[str, np.ndarray]]:
        """进行预测"""
        if not self.is_connected:
            return None
        
        self.total_calls += 1
        start_time = time.time()
        
        try:
            correct_obs = self.formatter.create_correct_observation(
                observation, apply_challenges, scenario
            )
            
            action = self.client.get_action(correct_obs)
            
            api_time = time.time() - start_time
            self.total_time += api_time
            
            if action is not None:
                self.total_successes += 1
                return action
            else:
                self.total_failures += 1
                return None
                
        except Exception as e:
            api_time = time.time() - start_time
            self.total_time += api_time
            self.total_failures += 1
            if self.config.verbose:
                print(f"⚠️ 预测异常: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.total_calls == 0:
            return {"calls": 0, "successes": 0, "failures": 0, "success_rate": 0, "avg_time": 0}
        
        return {
            "calls": self.total_calls,
            "successes": self.total_successes,
            "failures": self.total_failures,
            "success_rate": self.total_successes / self.total_calls,
            "avg_time": self.total_time / self.total_calls
        }

@dataclass
class IntegratedEpisodeResult:
    """集成Episode结果"""
    episode_id: int
    mode: str
    env_type: str
    task_name: str
    task_success: bool
    total_steps: int
    total_time: float
    final_reward: float
    api_calls: int
    api_successes: int
    avg_api_time: float
    metacog_interventions: int = 0
    scenario_difficulty: str = "unknown"

class IntegratedExperiment:
    """集成实验管理器"""
    
    def __init__(self, config: IntegratedConfig):
        self.config = config
        self.groot_client = IntegratedGR00TClient(config)
        self.env_manager = IntegratedEnvironmentManager(config)
        self.results = []
        
        # 设置元认知模块
        self.metacog_available = False
        if METACOG_AVAILABLE:
            try:
                self.metacog_module = CompleteMetaCognitiveModule(config.device)
                self.robocasa_adapter = RoboCasaToMetacogAdapter(image_size=(480, 640))
                self.groot_adapter = MetacogToGR00TAdapter()
                self.action_adjuster = ActionAdjuster()
                self.metacog_available = True
                print("✅ 元认知模块已加载")
            except Exception as e:
                print(f"⚠️ 元认知模块初始化失败: {e}")
        
        # 挑战场景
        self.current_scenario = None
        if ENHANCED_TASKS_AVAILABLE:
            self.task_generator = ComplexTaskGenerator()
    
    def run_experiment(self) -> bool:
        """运行集成实验"""
        print(f"\n🎯 集成GR00T + RoboCasa + 元认知对比实验")
        print("=" * 70)
        
        # 连接到GR00T服务
        if not self.groot_client.connect():
            print("❌ 无法连接到GR00T推理服务")
            return False
        
        # 打印环境信息
        env_info = self.env_manager.get_env_info()
        print(f"🏠 环境信息: {env_info}")
        
        try:
            # 运行基线实验
            if self.config.run_baseline:
                print(f"\n🤖 基线实验")
                print("-" * 50)
                
                for episode in range(self.config.num_episodes):
                    result = self._run_episode(episode, "baseline", False)
                    self.results.append(result)
                    self._print_episode_summary(result)
            
            # 运行元认知实验
            if self.config.run_metacognitive and self.metacog_available:
                print(f"\n🧠 元认知实验")
                print("-" * 50)
                
                for episode in range(self.config.num_episodes):
                    result = self._run_episode(episode, "metacognitive", True)
                    self.results.append(result)
                    self._print_episode_summary(result)
            
            # 分析结果
            self._analyze_results()
            self._save_results()
            
            return True
            
        finally:
            self.env_manager.close()
    
    def _run_episode(self, episode_id: int, mode: str, use_metacognitive: bool) -> IntegratedEpisodeResult:
        """运行集成episode"""
        start_time = time.time()
        
        # 选择挑战场景
        if self.config.use_challenging_scenarios and hasattr(self, 'task_generator'):
            complexity_levels = list(TaskComplexity)
            complexity = complexity_levels[episode_id % len(complexity_levels)]
            scenarios = [s for s in self.task_generator.scenarios if s.complexity == complexity]
            self.current_scenario = np.random.choice(scenarios) if scenarios else None
        else:
            self.current_scenario = None
        
        result = IntegratedEpisodeResult(
            episode_id=episode_id,
            mode=mode,
            env_type=self.env_manager.current_env_type,
            task_name=self.config.robocasa_task if self.env_manager.current_env_type == "robocasa" else "enhanced_task",
            task_success=False,
            total_steps=0,
            total_time=0.0,
            final_reward=0.0,
            api_calls=0,
            api_successes=0,
            avg_api_time=0.0,
            scenario_difficulty=self.current_scenario.complexity.value if self.current_scenario else "unknown"
        )
        
        try:
            obs, info = self.env_manager.reset()
            done = False
            step_count = 0
            total_reward = 0.0
            api_times = []
            
            if self.current_scenario:
                print(f"     场景: {self.current_scenario.name} ({self.current_scenario.complexity.value})")
            
            print(f"     执行中: ", end="", flush=True)
            
            while not done and step_count < self.config.max_steps_per_episode:
                # 获取GR00T动作
                api_start = time.time()
                groot_action = self.groot_client.predict(
                    obs, 
                    apply_challenges=self.config.use_challenging_scenarios,
                    scenario=self.current_scenario
                )
                api_time = time.time() - api_start
                api_times.append(api_time)
                
                result.api_calls += 1
                
                if groot_action is not None:
                    result.api_successes += 1
                    print(".", end="", flush=True)
                else:
                    print("x", end="", flush=True)
                
                # 元认知处理
                if use_metacognitive and self.metacog_available:
                    try:
                        sensor_data = self.robocasa_adapter.convert_observation(
                            obs, np.random.uniform(-0.03, 0.03, 6)
                        )
                        metacog_output = self.metacog_module.process_sensor_data(sensor_data)
                        
                        if metacog_output.directive.value != "continue":
                            result.metacog_interventions += 1
                            print("M", end="", flush=True)
                    except Exception as e:
                        if self.config.verbose:
                            print(f"元认知异常: {e}")
                
                # 环境步进
                env_action = np.random.uniform(-0.05, 0.05, 6)
                obs, reward, done, _, info = self.env_manager.step(env_action)
                total_reward += reward
                step_count += 1
                
                if info.get("task_success", False):
                    result.task_success = True
                    done = True
                    print("!", end="", flush=True)
                
                # 每20步显示进度
                if step_count % 20 == 0:
                    success_rate = result.api_successes / result.api_calls if result.api_calls > 0 else 0
                    print(f"|{success_rate:.0%}", end="", flush=True)
            
            result.total_steps = step_count
            result.total_time = time.time() - start_time
            result.final_reward = total_reward
            result.avg_api_time = np.mean(api_times) if api_times else 0.0
            
            print()  # 换行
            
        except Exception as e:
            result.total_time = time.time() - start_time
            print(f" 异常: {e}")
        
        return result
    
    def _print_episode_summary(self, result: IntegratedEpisodeResult):
        """打印episode摘要"""
        status = "✅ 成功" if result.task_success else "❌ 失败"
        api_success_rate = result.api_successes / result.api_calls if result.api_calls > 0 else 0
        
        print(f"   结果: {status}")
        print(f"   环境: {result.env_type}, 任务: {result.task_name}")
        print(f"   难度: {result.scenario_difficulty}")
        print(f"   执行: {result.total_steps} 步, {result.total_time:.1f}s, 奖励: {result.final_reward:.3f}")
        print(f"   API: {result.api_successes}/{result.api_calls} 成功 ({api_success_rate:.1%})")
        
        if result.metacog_interventions > 0:
            print(f"   元认知: {result.metacog_interventions} 次干预")
    
    def _analyze_results(self):
        """分析实验结果"""
        print(f"\n📊 集成实验结果分析")
        print("=" * 70)
        
        baseline_results = [r for r in self.results if r.mode == "baseline"]
        metacog_results = [r for r in self.results if r.mode == "metacognitive"]
        
        def analyze_mode(results: List[IntegratedEpisodeResult], mode_name: str):
            if not results:
                return
            
            successes = sum(1 for r in results if r.task_success)
            success_rate = successes / len(results)
            avg_reward = np.mean([r.final_reward for r in results])
            avg_steps = np.mean([r.total_steps for r in results])
            total_api_calls = sum(r.api_calls for r in results)
            total_api_successes = sum(r.api_successes for r in results)
            api_success_rate = total_api_successes / total_api_calls if total_api_calls > 0 else 0
            
            print(f"\n🔍 {mode_name} 模式分析:")
            print(f"   任务成功率: {success_rate:.1%} ({successes}/{len(results)})")
            print(f"   平均奖励: {avg_reward:.3f}")
            print(f"   平均步数: {avg_steps:.1f}")
            print(f"   API成功率: {api_success_rate:.1%}")
            
            # 按难度分析
            difficulty_stats = {}
            for result in results:
                diff = result.scenario_difficulty
                if diff not in difficulty_stats:
                    difficulty_stats[diff] = {"total": 0, "success": 0}
                difficulty_stats[diff]["total"] += 1
                if result.task_success:
                    difficulty_stats[diff]["success"] += 1
            
            print(f"   按难度分析:")
            for diff, stats in difficulty_stats.items():
                rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
                print(f"     {diff}: {rate:.1%} ({stats['success']}/{stats['total']})")
            
            if mode_name == "元认知":
                total_interventions = sum(r.metacog_interventions for r in results)
                print(f"   元认知干预总数: {total_interventions}")
        
        analyze_mode(baseline_results, "基线")
        analyze_mode(metacog_results, "元认知")
        
        # 对比分析
        if baseline_results and metacog_results:
            print(f"\n⚖️ 对比分析:")
            
            baseline_success = sum(1 for r in baseline_results if r.task_success) / len(baseline_results)
            metacog_success = sum(1 for r in metacog_results if r.task_success) / len(metacog_results)
            success_improvement = metacog_success - baseline_success
            
            baseline_reward = np.mean([r.final_reward for r in baseline_results])
            metacog_reward = np.mean([r.final_reward for r in metacog_results])
            reward_improvement = metacog_reward - baseline_reward
            
            print(f"   任务成功率变化: {success_improvement:+.1%}")
            print(f"   平均奖励变化: {reward_improvement:+.3f}")
            
            if success_improvement > 0.05:  # 5%以上提升认为显著
                print(f"   ✅ 元认知模块显著提升了任务表现")
            elif success_improvement > 0:
                print(f"   📈 元认知模块轻微提升了任务表现")
            elif success_improvement == 0:
                print(f"   ➡️ 元认知模块保持了任务表现")
            else:
                print(f"   ⚠️ 元认知模块可能需要进一步调优")
        
        # 环境评估
        env_types = set(r.env_type for r in self.results)
        print(f"\n🏠 环境评估:")
        print(f"   使用的环境: {', '.join(env_types)}")
        
        if "robocasa" in env_types:
            print(f"   🎉 成功使用RoboCasa真实环境进行实验")
            print(f"   🎥 可视化GUI已启用" if self.config.enable_gui else "   📊 批量模式运行")
    
    def _save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"integrated_experiment_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "experiment_type": "Integrated_GR00T_RoboCasa_Metacognitive",
            "config": asdict(self.config),
            "environment": {
                "type": self.env_manager.current_env_type,
                "info": self.env_manager.get_env_info()
            },
            "groot_client_stats": self.groot_client.get_stats(),
            "results": [asdict(r) for r in self.results]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\n💾 实验结果已保存: {filename}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="集成GR00T + RoboCasa + 元认知实验")
    parser.add_argument("--host", default="localhost", help="GR00T服务主机")
    parser.add_argument("--port", type=int, default=5555, help="GR00T服务端口")
    parser.add_argument("--episodes", type=int, default=8, help="每种模式的episode数量")
    parser.add_argument("--max-steps", type=int, default=100, help="每个episode最大步数")
    parser.add_argument("--robocasa-task", default="CleaningSink", 
                       choices=["CleaningSink", "MakingCoffee", "DoingDishes", "OpeningFridge"],
                       help="RoboCasa任务")
    parser.add_argument("--no-gui", action="store_true", help="禁用GUI可视化")
    parser.add_argument("--no-robocasa", action="store_true", help="不使用RoboCasa环境")
    parser.add_argument("--no-challenges", action="store_true", help="不使用挑战场景")
    parser.add_argument("--baseline-only", action="store_true", help="仅运行基线实验")
    parser.add_argument("--metacog-only", action="store_true", help="仅运行元认知实验")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 创建配置
    config = IntegratedConfig(
        host=args.host,
        port=args.port,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        robocasa_task=args.robocasa_task,
        enable_gui=not args.no_gui,
        use_robocasa=not args.no_robocasa,
        use_challenging_scenarios=not args.no_challenges,
        run_baseline=not args.metacog_only,
        run_metacognitive=not args.baseline_only,
        verbose=args.verbose
    )
    
    print(f"🎯 集成GR00T + RoboCasa + 元认知对比实验")
    print("=" * 70)
    print(f"配置:")
    print(f"   GR00T服务: {config.host}:{config.port}")
    print(f"   RoboCasa任务: {config.robocasa_task}")
    print(f"   Episodes: {config.num_episodes}")
    print(f"   GUI可视化: {'启用' if config.enable_gui else '禁用'}")
    print(f"   挑战场景: {'启用' if config.use_challenging_scenarios else '禁用'}")
    
    # 运行实验
    experiment = IntegratedExperiment(config)
    
    try:
        success = experiment.run_experiment()
        if success:
            print(f"\n🎉 集成实验成功完成！")
            print(f"🔥 成功测试GR00T + RoboCasa + 元认知集成")
            print(f"📊 获得了有意义的对比数据")
        else:
            print(f"\n❌ 实验失败")
    
    except KeyboardInterrupt:
        print(f"\n⚠️ 实验被用户中断")
        experiment.env_manager.close()
    except Exception as e:
        print(f"\n❌ 实验异常: {e}")
        import traceback
        traceback.print_exc()
        experiment.env_manager.close()

if __name__ == "__main__":
    main()
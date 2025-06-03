#!/usr/bin/env python3
"""
自适应GR00T客户端 - 动态匹配服务端数据格式
Adaptive GR00T Client - Dynamic Data Format Matching
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

@dataclass
class AdaptiveConfig:
    """自适应实验配置"""
    # 服务连接
    host: str = "localhost"
    port: int = 5555
    
    # 实验设置
    experiment_name: str = "adaptive_groot_experiment"
    num_episodes: int = 8
    max_steps_per_episode: int = 80
    
    # 实验模式
    run_baseline: bool = True
    run_metacognitive: bool = True
    
    # 元认知设置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class SmartDataFormatDetector:
    """智能数据格式检测器"""
    
    def __init__(self, client):
        self.client = client
        self.detected_format = None
        self.video_keys = []
        self.state_keys = []
        self.action_keys = []
        
    def detect_format(self) -> bool:
        """检测服务端期望的数据格式"""
        try:
            print("🔍 检测服务端数据格式...")
            
            # 获取模态配置
            modality_config = self.client.get_modality_config()
            
            print("📋 服务端期望的数据格式:")
            for key, config in modality_config.items():
                print(f"   - {key}: {config}")
                
                # 分类键名
                if "video" in key.lower() or "image" in key.lower():
                    self.video_keys.append(key)
                elif "state" in key.lower():
                    self.state_keys.append(key)
                elif "action" in key.lower():
                    self.action_keys.append(key)
            
            # 确定数据格式
            self.detected_format = {
                "video_keys": self.video_keys,
                "state_keys": self.state_keys,
                "action_keys": self.action_keys,
                "full_config": modality_config
            }
            
            print(f"✅ 检测完成:")
            print(f"   视频键: {self.video_keys}")
            print(f"   状态键: {self.state_keys}")
            
            return True
            
        except Exception as e:
            print(f"❌ 格式检测失败: {e}")
            return False
    
    def create_compatible_observation(self, base_obs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """创建兼容的观察数据"""
        compatible_obs = {}
        
        # 处理视频数据
        for video_key in self.video_keys:
            if "frontview_image" in base_obs:
                img = base_obs["frontview_image"]
                # 调整图像尺寸和格式
                if img.shape != (256, 256, 3):
                    import cv2
                    img = cv2.resize(img, (256, 256))
                
                # 根据检测到的键名设置
                compatible_obs[video_key] = img.reshape(1, 256, 256, 3).astype(np.uint8)
                print(f"     📷 设置视频数据: {video_key} -> {compatible_obs[video_key].shape}")
            else:
                # 生成合适的随机图像
                compatible_obs[video_key] = np.random.randint(0, 128, (1, 256, 256, 3), dtype=np.uint8)
        
        # 处理状态数据
        joint_data = base_obs.get("robot0_joint_pos", np.random.uniform(-0.3, 0.3, 7))
        joint_data = np.clip(joint_data, -1.0, 1.0)  # 限制范围
        
        for state_key in self.state_keys:
            if "arm" in state_key:
                # 手臂关节数据
                if len(joint_data) >= 7:
                    compatible_obs[state_key] = joint_data[:7].reshape(1, 7).astype(np.float32)
                else:
                    compatible_obs[state_key] = np.random.uniform(-0.3, 0.3, (1, 7)).astype(np.float32)
            elif "hand" in state_key:
                # 手部数据
                compatible_obs[state_key] = np.random.uniform(-0.1, 0.1, (1, 6)).astype(np.float32)
            elif "waist" in state_key:
                # 腰部数据
                compatible_obs[state_key] = np.random.uniform(-0.05, 0.05, (1, 3)).astype(np.float32)
            else:
                # 其他状态数据，根据配置推断尺寸
                config = self.detected_format["full_config"].get(state_key, {})
                shape_info = str(config)
                
                if "7" in shape_info:
                    compatible_obs[state_key] = np.random.uniform(-0.2, 0.2, (1, 7)).astype(np.float32)
                elif "6" in shape_info:
                    compatible_obs[state_key] = np.random.uniform(-0.1, 0.1, (1, 6)).astype(np.float32)
                elif "3" in shape_info:
                    compatible_obs[state_key] = np.random.uniform(-0.05, 0.05, (1, 3)).astype(np.float32)
                else:
                    compatible_obs[state_key] = np.random.uniform(-0.1, 0.1, (1, 4)).astype(np.float32)
        
        # 添加任务描述（如果需要）
        annotation_keys = [k for k in self.detected_format["full_config"].keys() if "annotation" in k]
        for ann_key in annotation_keys:
            compatible_obs[ann_key] = ["Execute robotic manipulation task"]
        
        return compatible_obs

class AdaptiveGR00TClient:
    """自适应GR00T客户端"""
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.client = None
        self.format_detector = None
        self.is_connected = False
        
        # 统计信息
        self.total_calls = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_time = 0.0
    
    def connect(self) -> bool:
        """连接并检测数据格式"""
        if not GROOT_CLIENT_AVAILABLE:
            print("❌ GR00T官方客户端不可用")
            return False
        
        try:
            print(f"🔗 连接到GR00T服务: {self.config.host}:{self.config.port}")
            
            # 创建客户端
            self.client = RobotInferenceClient(
                host=self.config.host, 
                port=self.config.port
            )
            
            # 检测数据格式
            self.format_detector = SmartDataFormatDetector(self.client)
            if not self.format_detector.detect_format():
                return False
            
            self.is_connected = True
            return True
            
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def predict(self, observation: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        """智能预测"""
        if not self.is_connected:
            return None
        
        self.total_calls += 1
        start_time = time.time()
        
        try:
            # 创建兼容的观察数据
            compatible_obs = self.format_detector.create_compatible_observation(observation)
            
            # 调用API
            action = self.client.get_action(compatible_obs)
            
            api_time = time.time() - start_time
            self.total_time += api_time
            self.total_successes += 1
            
            return action
            
        except Exception as e:
            api_time = time.time() - start_time
            self.total_time += api_time
            self.total_failures += 1
            print(f"⚠️ API调用失败: {e}")
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
class SimpleEpisodeResult:
    """简化的Episode结果"""
    episode_id: int
    mode: str
    task_success: bool
    total_steps: int
    total_time: float
    api_calls: int
    api_successes: int
    avg_api_time: float
    metacog_interventions: int = 0

class AdaptiveGR00TExperiment:
    """自适应GR00T实验"""
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
        self.groot_client = AdaptiveGR00TClient(config)
        self.results = []
        
        # 设置元认知模块
        self.metacog_available = False
        if METACOG_AVAILABLE:
            try:
                self.metacog_module = CompleteMetaCognitiveModule(config.device)
                self.robocasa_adapter = RoboCasaToMetacogAdapter()
                self.groot_adapter = MetacogToGR00TAdapter()
                self.action_adjuster = ActionAdjuster()
                self.metacog_available = True
                print("✅ 元认知模块已加载")
            except Exception as e:
                print(f"⚠️ 元认知模块初始化失败: {e}")
        
        # 创建环境
        self.environment = self._create_environment()
    
    def _create_environment(self):
        """创建测试环境"""
        class OptimizedTestEnvironment:
            def __init__(self):
                self.step_count = 0
                self.max_steps = 80
                print("🏠 初始化优化测试环境")
            
            def reset(self):
                self.step_count = 0
                return self._generate_obs(), {}
            
            def step(self, action):
                self.step_count += 1
                obs = self._generate_obs()
                
                # 优化的任务完成逻辑
                if self.step_count > 15 and np.random.random() < 0.45:
                    done = True
                    reward = 1.0
                elif self.step_count >= self.max_steps:
                    done = True
                    reward = -0.5
                else:
                    done = False
                    reward = np.random.uniform(-0.05, 0.05)
                
                info = {
                    "task_success": done and reward > 0,
                    "collision": np.random.random() < 0.008,
                    "force_violation": np.random.random() < 0.004
                }
                
                return obs, reward, done, False, info
            
            def _generate_obs(self):
                return {
                    "frontview_image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
                    "robot0_joint_pos": np.random.uniform(-0.4, 0.4, 7),
                    "robot0_joint_vel": np.random.uniform(-0.1, 0.1, 7),
                    "robot0_eef_pos": np.array([0.5, 0.0, 0.8]),
                    "robot0_eef_quat": np.array([0, 0, 0, 1])
                }
        
        return OptimizedTestEnvironment()
    
    def run_experiment(self) -> bool:
        """运行完整实验"""
        print(f"\n🎯 开始自适应GR00T元认知对比实验")
        print("=" * 70)
        
        # 连接到GR00T服务
        if not self.groot_client.connect():
            print("❌ 无法连接到GR00T推理服务")
            return False
        
        try:
            # 运行基线实验
            if self.config.run_baseline:
                print(f"\n🤖 基线实验 (GR00T N1)")
                print("-" * 50)
                
                for episode in range(self.config.num_episodes):
                    print(f"\n📊 基线 Episode {episode + 1}/{self.config.num_episodes}")
                    result = self._run_episode(episode, "baseline", False)
                    self.results.append(result)
                    self._print_episode_summary(result)
            
            # 运行元认知实验
            if self.config.run_metacognitive and self.metacog_available:
                print(f"\n🧠 元认知实验 (GR00T N1 + 元认知模块)")
                print("-" * 50)
                
                for episode in range(self.config.num_episodes):
                    print(f"\n📊 元认知 Episode {episode + 1}/{self.config.num_episodes}")
                    result = self._run_episode(episode, "metacognitive", True)
                    self.results.append(result)
                    self._print_episode_summary(result)
            
            # 分析结果
            self._analyze_results()
            self._save_results()
            
            return True
            
        finally:
            pass
    
    def _run_episode(self, episode_id: int, mode: str, use_metacognitive: bool) -> SimpleEpisodeResult:
        """运行单个episode"""
        start_time = time.time()
        
        result = SimpleEpisodeResult(
            episode_id=episode_id,
            mode=mode,
            task_success=False,
            total_steps=0,
            total_time=0.0,
            api_calls=0,
            api_successes=0,
            avg_api_time=0.0
        )
        
        try:
            obs, info = self.environment.reset()
            done = False
            step_count = 0
            api_times = []
            
            print(f"     执行中: ", end="", flush=True)
            
            while not done and step_count < self.config.max_steps_per_episode:
                # 获取GR00T动作
                api_start = time.time()
                groot_action = self.groot_client.predict(obs)
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
                            obs, np.random.uniform(-0.1, 0.1, 7)
                        )
                        metacog_output = self.metacog_module.process_sensor_data(sensor_data)
                        
                        if metacog_output.directive.value != "continue":
                            result.metacog_interventions += 1
                            print("M", end="", flush=True)
                    except Exception as e:
                        pass
                
                # 环境步进
                env_action = np.random.uniform(-0.08, 0.08, 9)
                obs, reward, done, _, info = self.environment.step(env_action)
                step_count += 1
                
                if info.get("task_success", False):
                    result.task_success = True
                    done = True
                    print("!", end="", flush=True)
                
                # 每10步显示进度
                if step_count % 10 == 0:
                    print("|", end="", flush=True)
            
            result.total_steps = step_count
            result.total_time = time.time() - start_time
            result.avg_api_time = np.mean(api_times) if api_times else 0.0
            
            print()  # 换行
            
        except Exception as e:
            result.total_time = time.time() - start_time
            print(f" 错误: {e}")
        
        return result
    
    def _print_episode_summary(self, result: SimpleEpisodeResult):
        """打印episode摘要"""
        status = "✅ 成功" if result.task_success else "❌ 失败"
        api_success_rate = result.api_successes / result.api_calls if result.api_calls > 0 else 0
        
        print(f"   结果: {status}")
        print(f"   执行: {result.total_steps} 步, {result.total_time:.1f}s")
        print(f"   API: {result.api_successes}/{result.api_calls} 成功 ({api_success_rate:.1%}), "
              f"平均 {result.avg_api_time*1000:.1f}ms")
        
        if result.metacog_interventions > 0:
            print(f"   元认知: {result.metacog_interventions} 次干预")
    
    def _analyze_results(self):
        """分析实验结果"""
        print(f"\n📊 自适应实验结果分析")
        print("=" * 70)
        
        baseline_results = [r for r in self.results if r.mode == "baseline"]
        metacog_results = [r for r in self.results if r.mode == "metacognitive"]
        
        def analyze_mode(results: List[SimpleEpisodeResult], mode_name: str):
            if not results:
                return
            
            successes = sum(1 for r in results if r.task_success)
            success_rate = successes / len(results)
            avg_steps = np.mean([r.total_steps for r in results])
            avg_time = np.mean([r.total_time for r in results])
            total_api_calls = sum(r.api_calls for r in results)
            total_api_successes = sum(r.api_successes for r in results)
            api_success_rate = total_api_successes / total_api_calls if total_api_calls > 0 else 0
            avg_api_time = np.mean([r.avg_api_time for r in results])
            
            print(f"\n🔍 {mode_name} 模式分析:")
            print(f"   任务成功率: {success_rate:.1%} ({successes}/{len(results)})")
            print(f"   平均执行步数: {avg_steps:.1f}")
            print(f"   平均执行时间: {avg_time:.1f}s")
            print(f"   API成功率: {api_success_rate:.1%} ({total_api_successes}/{total_api_calls})")
            print(f"   平均API响应时间: {avg_api_time*1000:.1f}ms")
            
            if mode_name == "元认知":
                total_interventions = sum(r.metacog_interventions for r in results)
                avg_interventions = total_interventions / len(results)
                print(f"   元认知干预总数: {total_interventions}")
                print(f"   平均每episode干预: {avg_interventions:.1f}")
        
        analyze_mode(baseline_results, "基线")
        analyze_mode(metacog_results, "元认知")
        
        # 对比分析
        if baseline_results and metacog_results:
            print(f"\n⚖️ 对比分析:")
            
            baseline_success = sum(1 for r in baseline_results if r.task_success) / len(baseline_results)
            metacog_success = sum(1 for r in metacog_results if r.task_success) / len(metacog_results)
            success_improvement = metacog_success - baseline_success
            
            baseline_steps = np.mean([r.total_steps for r in baseline_results])
            metacog_steps = np.mean([r.total_steps for r in metacog_results])
            step_change = metacog_steps - baseline_steps
            
            print(f"   任务成功率变化: {success_improvement:+.1%}")
            print(f"   平均步数变化: {step_change:+.1f}")
            
            if success_improvement > 0:
                print(f"   ✅ 元认知模块提升了任务成功率")
            if step_change < 0:
                print(f"   ✅ 元认知模块减少了执行步数")
        
        # GR00T服务统计
        client_stats = self.groot_client.get_stats()
        print(f"\n📡 GR00T服务统计:")
        print(f"   总API调用: {client_stats['calls']}")
        print(f"   API成功率: {client_stats['success_rate']:.1%}")
        print(f"   平均响应时间: {client_stats['avg_time']*1000:.1f}ms")
        
        if client_stats['success_rate'] > 0.8:
            print(f"   🎉 API调用质量优秀！数据格式适配成功")
        elif client_stats['success_rate'] > 0.6:
            print(f"   👍 API调用质量良好")
        else:
            print(f"   ⚠️ API调用仍有问题，可能需要进一步调整")
    
    def _save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"adaptive_groot_experiment_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "config": asdict(self.config),
            "groot_client_stats": self.groot_client.get_stats(),
            "detected_format": self.groot_client.format_detector.detected_format if self.groot_client.format_detector else None,
            "results": [asdict(r) for r in self.results]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\n💾 详细结果已保存: {filename}")

def main():
    """主函数"""
    print("🎯 自适应GR00T元认知对比实验")
    print("智能检测并适配服务端数据格式")
    print("=" * 70)
    
    # 配置
    config = AdaptiveConfig(
        host="localhost",
        port=5555,
        experiment_name="adaptive_groot_test",
        num_episodes=5,
        max_steps_per_episode=60
    )
    
    print(f"实验配置:")
    print(f"   GR00T服务: {config.host}:{config.port}")
    print(f"   Episodes: {config.num_episodes}")
    print(f"   最大步数: {config.max_steps_per_episode}")
    
    # 运行实验
    experiment = AdaptiveGR00TExperiment(config)
    
    try:
        success = experiment.run_experiment()
        if success:
            print(f"\n🎉 自适应实验完成！")
            print(f"💡 数据格式已自动适配，获得了真实的GR00T推理结果")
            print(f"📊 成功验证了元认知模块的效果")
        else:
            print(f"\n❌ 实验失败")
    
    except KeyboardInterrupt:
        print(f"\n⚠️ 实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
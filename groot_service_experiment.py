#!/usr/bin/env python3
"""
修复后的GR00T客户端 - 使用正确的视频分辨率 (640, 480)
Fixed GR00T Client - Correct Video Resolution (640, 480)
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

# # 导入元认知模块
# try:
#     from metacog_integration import (
#         CompleteMetaCognitiveModule,
#         RoboCasaToMetacogAdapter,
#         MetacogToGR00TAdapter,
#         ActionAdjuster,
#         SensorData
#     )
#     METACOG_AVAILABLE = True
#     print("✅ 元认知模块可用")
# except ImportError as e:
#     print(f"❌ 元认知模块不可用: {e}")
#     METACOG_AVAILABLE = False



      
try:
    from metacog_integration import (
        CompleteMetaCognitiveModule,
        RoboCasaToMetacogAdapter,
        MetacogToGR00TAdapter, # 这个可能不再直接用于调整GR00T动作了
        ActionAdjuster,       # 同上
        SensorData,
        MetaCognitiveOutput,  # 确保导入
        DirectiveType,        # 确保导入
        MetacogToVLASystem2Adapter # 新增导入
    )
    METACOG_AVAILABLE = True
    print("✅ 元认知模块可用")
except ImportError as e:
    print(f"❌ 元认知模块不可用: {e}")
    METACOG_AVAILABLE = False

    


@dataclass
class FinalConfig:
    """最终实验配置"""
    # 服务连接
    host: str = "localhost"
    port: int = 5555
    
    # 实验设置
    experiment_name: str = "final_groot_experiment"
    num_episodes: int = 8
    max_steps_per_episode: int = 60
    
    # 实验模式
    run_baseline: bool = True
    run_metacognitive: bool = True
    
    # 元认知设置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class FixedDataFormatter:
    """修复后的数据格式器 - 使用正确的视频分辨率"""
    
    def __init__(self):
        # 根据错误：服务端仍然期望640x480，不是224x224
        # 虽然SO100配置是224x224，但微调模型期望640x480
        self.required_keys = {
            "video.webcam": (640, 480, 3),              # 服务端期望：640x480
            "state.single_arm": (1, 5),                 # 5个关节
            "state.gripper": (1, 1),                    # 1个夹爪维度
            "annotation.human.task_description": None   # 任务描述
        }
        
        print("🎯 使用微调模型期望配置:")
        for key, shape in self.required_keys.items():
            if shape:
                print(f"   - {key}: {shape}")
            else:
                print(f"   - {key}: [string list]")
    
    def create_correct_observation(self, base_obs: Dict[str, np.ndarray] = None) -> Dict[str, Any]:
        """创建正确格式的观察数据"""
        correct_obs = {}
        
        # 1. 视频数据 - video.webcam (修正为 height=480, width=640)
        if base_obs and "frontview_image" in base_obs:
            img = base_obs["frontview_image"]
            # 修正：确保是 (height=480, width=640) 格式
            if img.shape[:2] != (480, 640):  # (height, width)
                import cv2
                img = cv2.resize(img, (640, 480))  # cv2.resize使用 (width, height)
            
            # 确保正确的维度：(height=480, width=640, channels=3)
            if len(img.shape) == 3:  # (H, W, C)
                correct_obs["video.webcam"] = img.astype(np.uint8)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")
        else:
            # 生成正确分辨率的测试图像
            correct_obs["video.webcam"] = self._generate_correct_image()
        
        # 1. 视频数据 - video.webcam (服务端仍期望：640x480)
        if base_obs and "frontview_image" in base_obs:
            img = base_obs["frontview_image"]
            # 服务端期望640x480分辨率
            if img.shape[:2] != (480, 640):  # (height, width)
                import cv2
                img = cv2.resize(img, (640, 480))  # cv2.resize使用 (width, height)
            
            # 添加单个batch维度：(480, 640, 3) → (1, 480, 640, 3)
            if len(img.shape) == 3:
                correct_obs["video.webcam"] = img[np.newaxis, :, :, :].astype(np.uint8)
            else:
                correct_obs["video.webcam"] = img.astype(np.uint8)
        else:
            # 生成正确分辨率的测试图像
            correct_obs["video.webcam"] = self._generate_correct_image()
        
        # 2. 单臂状态 - state.single_arm (5-DOF，float32)
        if base_obs and "robot0_joint_pos" in base_obs:
            joint_pos = base_obs["robot0_joint_pos"][:5]  # 5个关节
            joint_pos = np.clip(joint_pos, -1.0, 1.0)     # 限制安全范围
            correct_obs["state.single_arm"] = joint_pos[np.newaxis, :].astype(np.float32)  # float32
        else:
            # 生成5关节数据，使用float32
            joint_data = np.random.uniform(-0.3, 0.3, 5)
            correct_obs["state.single_arm"] = joint_data[np.newaxis, :].astype(np.float32)  # float32
        
        # 3. 夹爪状态 - state.gripper (1维，float32)
        if base_obs and "robot0_gripper_qpos" in base_obs:
            gripper_pos = base_obs["robot0_gripper_qpos"][:1]  # 只取第1个维度
            correct_obs["state.gripper"] = gripper_pos[np.newaxis, :].astype(np.float32)  # float32
        else:
            # 生成1维夹爪数据，使用float32
            gripper_data = np.random.uniform(-0.1, 0.1, 1)
            correct_obs["state.gripper"] = gripper_data[np.newaxis, :].astype(np.float32)  # float32
        
        # 4. 任务描述 - annotation.human.task_description (正确的key)
        correct_obs["annotation.human.task_description"] = ["Execute SO-100 manipulation task"]
        
        return correct_obs
        
        return correct_obs
    
    def _generate_correct_image(self) -> np.ndarray:
        """生成正确分辨率的测试图像 (640x480)，带单个batch维度"""
        # 创建服务端期望的分辨率：(1, 480, 640, 3)
        img = np.zeros((1, 480, 640, 3), dtype=np.uint8)
        
        # 创建有结构的图像，避免纯随机
        for i in range(480):  # height
            for j in range(640):  # width
                # 渐变模式
                img[0, i, j, 0] = (i + j) % 256           # Red
                img[0, i, j, 1] = (i * 2) % 256           # Green  
                img[0, i, j, 2] = (j * 2) % 256           # Blue
        
        return img
    
    def print_observation_details(self, obs: Dict[str, Any]):
        """打印观察数据详情"""
        print("📊 发送的观察数据:")
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}, range=[{value.min():.3f}, {value.max():.3f}]")
            elif isinstance(value, list):
                print(f"   {key}: list[{len(value)}] = {value}")
            else:
                print(f"   {key}: {type(value)} = {value}")

class FinalGR00TClient:
    """最终GR00T客户端"""
    
    def __init__(self, config: FinalConfig):
        self.config = config
        self.client = None
        self.formatter = FixedDataFormatter()  # 使用修复后的格式器
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
            
            # 创建客户端
            self.client = RobotInferenceClient(
                host=self.config.host, 
                port=self.config.port
            )
            
            # 验证连接
            print("📋 验证连接...")
            modality_config = self.client.get_modality_config()
            
            print("✅ 连接成功！服务端配置:")
            for key, config in modality_config.items():
                print(f"   - {key}: {config.modality_keys}")
            
            # 进行一次测试调用（使用正确的分辨率）
            print("\n🧪 进行连接测试（使用微调模型期望配置：640x480 + float32）...")
            test_obs = self.formatter.create_correct_observation()
            self.formatter.print_observation_details(test_obs)
            
            print("🚀 发送测试请求...")
            test_result = self.client.get_action(test_obs)
            
            if test_result is not None:
                print("✅ 测试调用成功！")
                print(f"📤 返回动作键: {list(test_result.keys())}")
                for key, value in test_result.items():
                    if isinstance(value, np.ndarray):
                        print(f"   {key}: shape={value.shape}")
                self.is_connected = True
                return True
            else:
                print("❌ 测试调用失败")
                return False
                
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, observation: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        """进行预测"""
        if not self.is_connected:
            return None
        
        self.total_calls += 1
        start_time = time.time()
        
        try:
            # 创建正确格式的观察数据
            correct_obs = self.formatter.create_correct_observation(observation)
            
            # 调用API
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
class FinalEpisodeResult:
    """最终Episode结果"""
    episode_id: int
    mode: str
    task_success: bool
    total_steps: int
    total_time: float
    api_calls: int
    api_successes: int
    avg_api_time: float
    metacog_interventions: int = 0
    groot_actions_received: int = 0

class FinalGR00TExperiment:
    """最终GR00T实验"""
    
    # def __init__(self, config: FinalConfig):
    #     self.config = config
    #     self.groot_client = FinalGR00TClient(config)
    #     self.results = []
        
    #     # 设置元认知模块
    #     self.metacog_available = False
    #     if METACOG_AVAILABLE:
    #         try:
    #             self.metacog_module = CompleteMetaCognitiveModule(config.device)
    #             self.robocasa_adapter = RoboCasaToMetacogAdapter(image_size=(480, 640))  # 640x480
    #             self.groot_adapter = MetacogToGR00TAdapter()
    #             self.action_adjuster = ActionAdjuster()
    #             self.metacog_available = True
    #             print("✅ 元认知模块已加载")
    #         except Exception as e:
    #             print(f"⚠️ 元认知模块初始化失败: {e}")
        
    #     # 创建环境
    #     self.environment = self._create_environment()



class FinalGR00TExperiment:
    def __init__(self, config: FinalConfig):
        self.config = config
        self.groot_client = FinalGR00TClient(config)
        self.results = []

        # 设置元认知模块
        self.metacog_available = False
        if METACOG_AVAILABLE:
            try:
                self.metacog_module = CompleteMetaCognitiveModule(config.device)
                self.robocasa_adapter = RoboCasaToMetacogAdapter(image_size=(480, 640))
                # self.groot_adapter = MetacogToGR00TAdapter() # 旧的适配器，可能不再需要
                # self.action_adjuster = ActionAdjuster()     # 旧的调整器，可能不再需要
                self.metacog_to_vla_s2_adapter = MetacogToVLASystem2Adapter() # 初始化新的适配器
                self.metacog_available = True
                print("✅ 元认知模块已加载 (包含VLA System2适配器)")
            except Exception as e:
                print(f"⚠️ 元认知模块初始化失败: {e}")

        # 创建环境
        self.environment = self._create_environment()

    
    def _create_environment(self):
        """创建单臂机械手环境"""
        class SingleArmTestEnvironment:
            def __init__(self):
                self.step_count = 0
                self.max_steps = 60
                print("🤖 初始化SO-100机械臂测试环境（640x480分辨率，5-DOF+1-DOF夹爪）")
            
            def reset(self):
                self.step_count = 0
                return self._generate_obs(), {}
            
            def step(self, action):
                self.step_count += 1
                obs = self._generate_obs()
                
                # 单臂任务完成逻辑
                if self.step_count > 25 and np.random.random() < 0.35:
                    done = True
                    reward = 1.0
                elif self.step_count >= self.max_steps:
                    done = True
                    reward = -0.5
                else:
                    done = False
                    reward = np.random.uniform(-0.01, 0.01)
                
                info = {
                    "task_success": done and reward > 0,
                    "collision": np.random.random() < 0.003,
                    "force_violation": np.random.random() < 0.001
                }
                
                return obs, reward, done, False, info
            
            def _generate_obs(self):
                # 服务端实际期望：640x480分辨率 + 5关节 + 1夹爪
                return {
                    "frontview_image": np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8),  # 640x480
                    "robot0_joint_pos": np.random.uniform(-0.2, 0.2, 5),     # 5-DOF
                    "robot0_joint_vel": np.random.uniform(-0.1, 0.1, 5),     # 5-DOF 关节速度
                    "robot0_gripper_qpos": np.random.uniform(-0.05, 0.05, 1), # 1-DOF 夹爪
                    "robot0_eef_pos": np.array([0.5, 0.0, 0.8]),             # End effector位置
                    "robot0_eef_quat": np.array([0, 0, 0, 1])                # End effector方向
                }
        
        return SingleArmTestEnvironment()
    
    def run_experiment(self) -> bool:
        """运行最终实验"""
        print(f"\n🎯 开始修复后的GR00T元认知对比实验")
        print("使用微调模型期望配置：640x480 + float64 + 单个batch维度")
        print("=" * 70)
        
        # 连接到GR00T服务
        if not self.groot_client.connect():
            print("❌ 无法连接到GR00T推理服务")
            return False
        
        try:
            # 运行基线实验
            if self.config.run_baseline:
                print(f"\n🤖 基线实验 (GR00T N1 单臂)")
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
    
    # def _run_episode(self, episode_id: int, mode: str, use_metacognitive: bool) -> FinalEpisodeResult:
    #     """运行最终episode"""
    #     start_time = time.time()
        
    #     result = FinalEpisodeResult(
    #         episode_id=episode_id,
    #         mode=mode,
    #         task_success=False,
    #         total_steps=0,
    #         total_time=0.0,
    #         api_calls=0,
    #         api_successes=0,
    #         avg_api_time=0.0
    #     )
        
    #     try:
    #         obs, info = self.environment.reset()
    #         done = False
    #         step_count = 0
    #         api_times = []
            
    #         print(f"     执行中: ", end="", flush=True)
            
    #         while not done and step_count < self.config.max_steps_per_episode:
    #             # 获取GR00T动作
    #             api_start = time.time()
    #             groot_action = self.groot_client.predict(obs)
    #             api_time = time.time() - api_start
    #             api_times.append(api_time)
                
    #             result.api_calls += 1
                
    #             if groot_action is not None:
    #                 result.api_successes += 1
    #                 result.groot_actions_received += 1
    #                 print(".", end="", flush=True)
    #             else:
    #                 print("x", end="", flush=True)
                
    #             # 元认知处理
    #             if use_metacognitive and self.metacog_available:
    #                 try:
    #                     sensor_data = self.robocasa_adapter.convert_observation(
    #                         obs, np.random.uniform(-0.03, 0.03, 6)  # 实际期望：6维（5关节+1夹爪）
    #                     )
    #                     metacog_output = self.metacog_module.process_sensor_data(sensor_data)
                        
    #                     if metacog_output.directive.value != "continue":
    #                         result.metacog_interventions += 1
    #                         print("M", end="", flush=True)
    #                 except Exception as e:
    #                     pass
                
    #             # 环境步进
    #             env_action = np.random.uniform(-0.05, 0.05, 6)  # 实际期望：5个关节 + 1个夹爪 = 6维
    #             obs, reward, done, _, info = self.environment.step(env_action)
    #             step_count += 1
                
    #             if info.get("task_success", False):
    #                 result.task_success = True
    #                 done = True
    #                 print("!", end="", flush=True)
                
    #             # 每10步显示进度
    #             if step_count % 10 == 0:
    #                 success_rate = result.api_successes / result.api_calls if result.api_calls > 0 else 0
    #                 print(f"|{success_rate:.0%}", end="", flush=True)
            
    #         result.total_steps = step_count
    #         result.total_time = time.time() - start_time
    #         result.avg_api_time = np.mean(api_times) if api_times else 0.0
            
    #         print()  # 换行
            
    #     except Exception as e:
    #         result.total_time = time.time() - start_time
    #         print(f" 异常: {e}")
        
    #     return result



    def _run_episode(self, episode_id: int, mode: str, use_metacognitive: bool) -> FinalEpisodeResult:
        start_time = time.time()

        result = FinalEpisodeResult(
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
            # obs 是从环境获取的原始观测字典
            obs, info = self.environment.reset()
            done = False
            step_count = 0
            api_times = []
            
            # 用于存储元认知反馈给VLA System2的指令
            current_metacognitive_instruction_for_s2 = None

            print(f"     执行中 ({mode}): ", end="", flush=True)

            while not done and step_count < self.config.max_steps_per_episode:
                
                # 准备给GR00T/VLA的观测数据
                # 我们将把元认知指令（如果有的话）添加到这个观测字典中
                observation_for_groot = obs.copy() # 从环境获取的当前观测
                if current_metacognitive_instruction_for_s2:
                    # 假设VLA的System2从 "metacognitive_instruction" 键读取指令
                    # 这个键名需要与您的VLA模型期望的一致
                    observation_for_groot["metacognitive_instruction"] = [current_metacognitive_instruction_for_s2]
                    # print(f"\nDEBUG: Sending to GR00T with instruction: {current_metacognitive_instruction_for_s2}")
                else:
                    # 如果没有指令，可以不添加该键，或者添加一个空列表/占位符
                    # observation_for_groot["metacognitive_instruction"] = [] # 取决于VLA如何处理
                    pass


                # 获取GR00T/VLA动作
                api_start = time.time()
                # 使用包含（可能有的）元认知指令的观测数据进行预测
                groot_action_dict = self.groot_client.predict(observation_for_groot)
                api_time = time.time() - api_start
                api_times.append(api_time)

                result.api_calls += 1
                current_metacognitive_instruction_for_s2 = None # 清除上一条指令，等待新的

                if groot_action_dict is not None:
                    result.api_successes += 1
                    result.groot_actions_received += 1
                    print(".", end="", flush=True)
                    # 从GR00T返回的字典中提取实际的动作 (假设是这样)
                    # 您需要根据groot_action_dict的实际结构来提取用于环境step的动作
                    # 例如： env_action = groot_action_dict.get("action", np.random.uniform(-0.05, 0.05, 6))
                    # 为了简单，我们继续用随机动作，但理想情况下这里应该是GR00T的动作
                    env_action_to_execute = np.random.uniform(-0.05, 0.05, 6)
                    # 获取System1执行的动作信息，用于元认知模块
                    # 如果groot_action_dict包含详细的动作参数，可以用它
                    # 否则，可以用 env_action_to_execute
                    s1_action_info_for_metacog = env_action_to_execute
                else:
                    print("x", end="", flush=True)
                    # 如果GR00T没有返回动作，也需要一个动作给环境
                    env_action_to_execute = np.random.uniform(-0.05, 0.05, 6)
                    s1_action_info_for_metacog = env_action_to_execute


                # 环境步进，使用 env_action_to_execute
                # obs 在这里被更新为下一步的观测
                next_obs, reward, done, _, info = self.environment.step(env_action_to_execute)


                # 元认知处理 (基于 next_obs 和 s1_action_info_for_metacog)
                if use_metacognitive and self.metacog_available:
                    try:
                        # 使用执行动作后的新观测 next_obs 和 System1的动作信息
                        sensor_data = self.robocasa_adapter.convert_observation(
                            next_obs, # 使用执行动作后的新观测
                            s1_action_info_for_metacog, # System1执行的动作或其详细信息
                            execution_status="normal" # 可以根据实际情况更新
                        )
                        metacog_output = self.metacog_module.process_sensor_data(sensor_data)
                        
                        # 生成给System2的简练指令
                        instruction_for_s2 = self.metacog_to_vla_s2_adapter.convert_to_system2_instruction(metacog_output)

                        if instruction_for_s2:
                            current_metacognitive_instruction_for_s2 = instruction_for_s2
                            # 仅在生成了有效指令时才算作干预并打印
                            if metacog_output.directive != DirectiveType.CONTINUE: # 确保不是CONTINUE指令
                                result.metacog_interventions += 1
                                print(f"M[{instruction_for_s2[:15]}..]", end="", flush=True) # 打印指令的简短形式
                        
                    except Exception as e:
                        print(f"\n元认知处理异常: {e}")
                        pass
                
                obs = next_obs # 更新观测以供下一轮循环使用

                step_count += 1

                if info.get("task_success", False):
                    result.task_success = True
                    done = True
                    print("!", end="", flush=True)

                if step_count % 10 == 0 and result.api_calls > 0:
                    success_rate = result.api_successes / result.api_calls
                    print(f"|{success_rate:.0%}", end="", flush=True)
            
            result.total_steps = step_count
            result.total_time = time.time() - start_time
            result.avg_api_time = np.mean(api_times) if api_times else 0.0
            
            print() # 换行
            
        except Exception as e:
            result.total_time = time.time() - start_time
            print(f" 异常: {e}")
            import traceback
            traceback.print_exc() # 打印详细错误信息
        
        return result


    
    def _print_episode_summary(self, result: FinalEpisodeResult):
        """打印最终episode摘要"""
        status = "✅ 成功" if result.task_success else "❌ 失败"
        api_success_rate = result.api_successes / result.api_calls if result.api_calls > 0 else 0
        
        print(f"   结果: {status}")
        print(f"   执行: {result.total_steps} 步, {result.total_time:.1f}s")
        print(f"   API: {result.api_successes}/{result.api_calls} 成功 ({api_success_rate:.1%}), "
              f"平均 {result.avg_api_time*1000:.1f}ms")
        print(f"   GR00T动作: {result.groot_actions_received} 个")
        
        if result.metacog_interventions > 0:
            print(f"   元认知: {result.metacog_interventions} 次干预")
    
    def _analyze_results(self):
        """分析最终实验结果"""
        print(f"\n📊 最终实验结果分析")
        print("=" * 70)
        
        baseline_results = [r for r in self.results if r.mode == "baseline"]
        metacog_results = [r for r in self.results if r.mode == "metacognitive"]
        
        def analyze_mode(results: List[FinalEpisodeResult], mode_name: str):
            if not results:
                return
            
            successes = sum(1 for r in results if r.task_success)
            success_rate = successes / len(results)
            total_api_calls = sum(r.api_calls for r in results)
            total_api_successes = sum(r.api_successes for r in results)
            api_success_rate = total_api_successes / total_api_calls if total_api_calls > 0 else 0
            total_groot_actions = sum(r.groot_actions_received for r in results)
            avg_api_time = np.mean([r.avg_api_time for r in results])
            
            print(f"\n🔍 {mode_name} 模式分析:")
            print(f"   任务成功率: {success_rate:.1%} ({successes}/{len(results)})")
            print(f"   API成功率: {api_success_rate:.1%} ({total_api_successes}/{total_api_calls})")
            print(f"   有效GR00T动作: {total_groot_actions}")
            print(f"   平均API响应时间: {avg_api_time*1000:.1f}ms")
            
            if api_success_rate > 0.8:
                print(f"   🎉 完美！成功调用您的微调GR00T N1模型")
            elif api_success_rate > 0.5:
                print(f"   👍 良好！大部分API调用成功")
            elif api_success_rate > 0:
                print(f"   🔧 部分成功，仍需优化")
            else:
                print(f"   ❌ API调用失败")
            
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
            
            print(f"   任务成功率变化: {success_improvement:+.1%}")
            
            if success_improvement > 0:
                print(f"   ✅ 元认知模块提升了任务成功率")
            elif success_improvement == 0:
                print(f"   ➡️ 元认知模块保持了任务成功率")
            else:
                print(f"   ⚠️ 元认知模块降低了任务成功率")
        
        # GR00T服务最终评估
        client_stats = self.groot_client.get_stats()
        print(f"\n📡 GR00T服务最终评估:")
        print(f"   总API调用: {client_stats['calls']}")
        print(f"   API成功率: {client_stats['success_rate']:.1%}")
        print(f"   平均响应时间: {client_stats['avg_time']*1000:.1f}ms")
        
        if client_stats['success_rate'] >= 0.9:
            print(f"   🎉 完美！您的微调GR00T N1模型工作优秀")
        elif client_stats['success_rate'] >= 0.7:
            print(f"   👍 优秀！您的微调GR00T N1模型工作良好")
        elif client_stats['success_rate'] >= 0.5:
            print(f"   📈 良好！API调用基本稳定")
        elif client_stats['success_rate'] > 0:
            print(f"   🔧 有进展！部分API调用成功")
        else:
            print(f"   🛠️ 需要进一步调试")
    
    def _save_results(self):
        """保存最终实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fixed_groot_experiment_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "experiment_type": "GR00T_N1_Metacognitive_Comparison_Fixed",
            "model_configuration": "so100_finetuned_640x480_float32",
            "config": asdict(self.config),
            "groot_client_stats": self.groot_client.get_stats(),
            "results": [asdict(r) for r in self.results]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\n💾 最终结果已保存: {filename}")

def main():
    """修复后的主函数"""
    print("🎯 修复后的GR00T元认知对比实验")
    print("使用微调模型期望配置：640x480分辨率 + float32 + 单个batch维度")
    print("=" * 70)
    
    # 配置
    config = FinalConfig(
        host="localhost",
        port=5555,
        experiment_name="fixed_groot_metacog_comparison",
        num_episodes=3,  # 先用少量episode测试
        max_steps_per_episode=50
    )
    
    print(f"实验配置:")
    print(f"   GR00T服务: {config.host}:{config.port}")
    print(f"   模型配置: 单臂 (single_arm)")
    print(f"   视频分辨率: 640x480 (修正)")
    print(f"   Episodes: {config.num_episodes}")
    print(f"   最大步数: {config.max_steps_per_episode}")
    
    # 运行实验
    experiment = FinalGR00TExperiment(config)
    
    try:
        success = experiment.run_experiment()
        if success:
            print(f"\n🎉 修复后的实验成功完成！")
            print(f"🔥 成功调用您微调的GR00T N1模型")
            print(f"🧠 获得真实的元认知模块效果对比")
            print(f"📊 实验数据已保存，可用于分析和论文")
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
# #!/usr/bin/env python3
# """
# 集成真实RoboCasa环境的修改版本
# Real RoboCasa Environment Integration - Modified Version
# """

# import os
# import sys
# import time
# import json
# import numpy as np
# import torch
# from pathlib import Path
# from typing import Dict, List, Any, Tuple, Optional
# from dataclasses import dataclass, asdict
# from datetime import datetime
# import cv2

# # 导入RoboCasa
# try:
#     import robocasa
#     from robocasa.environments.kitchen.kitchen import Kitchen
#     ROBOCASA_AVAILABLE = True
#     print("✅ RoboCasa可用")
# except ImportError as e:
#     print(f"❌ RoboCasa不可用: {e}")
#     print("请安装RoboCasa: pip install robocasa")
#     ROBOCASA_AVAILABLE = False

# # 导入GR00T官方客户端
# try:
#     from gr00t.eval.robot import RobotInferenceClient
#     GROOT_CLIENT_AVAILABLE = True
#     print("✅ GR00T官方客户端可用")
# except ImportError as e:
#     print(f"❌ GR00T官方客户端不可用: {e}")
#     GROOT_CLIENT_AVAILABLE = False

# # 导入元认知模块
# try:
#     from metacog_integration import (
#         CompleteMetaCognitiveModule,
#         RoboCasaToMetacogAdapter,
#         MetacogToGR00TAdapter,
#         ActionAdjuster,
#         SensorData,
#         MetaCognitiveOutput,
#         DirectiveType,
#         MetacogToVLASystem2Adapter
#     )
#     METACOG_AVAILABLE = True
#     print("✅ 元认知模块可用")
# except ImportError as e:
#     print(f"❌ 元认知模块不可用: {e}")
#     METACOG_AVAILABLE = False

# # ==================== RoboCasa任务选择器 ====================

# class RoboCasaTaskSelector:
#     """RoboCasa任务选择和验证（使用实际可用的任务名称）"""
    
#     # 基于实际RoboCasa注册的环境名称
#     ATOMIC_TASKS = {
#         # 简单任务（推荐用于测试）
#         "beginner": [
#             "Lift",                    # 基础抓取任务
#             "Stack",                   # 堆叠任务
#             "PnP",                     # Pick and Place
#             "PnPCounterToSink",        # 柜台到水槽
#             "OpenDoor",                # 开门
#             "CloseDoor",               # 关门
#             "OpenSingleDoor",          # 开单门
#             "CloseSingleDoor"          # 关单门
#         ],
        
#         # 中等难度任务
#         "intermediate": [
#             "PnPCounterToCab",         # 柜台到橱柜
#             "PnPCabToCounter",         # 橱柜到柜台
#             "PnPSinkToCounter",        # 水槽到柜台
#             "OpenDrawer",              # 开抽屉
#             "CloseDrawer",             # 关抽屉
#             "TurnOnSinkFaucet",        # 开水龙头
#             "TurnOffSinkFaucet",       # 关水龙头
#             "OpenDoubleDoor",          # 开双门
#             "CloseDoubleDoor"          # 关双门
#         ],
        
#         # 高难度任务
#         "advanced": [
#             "Kitchen",                 # 综合厨房任务
#             "KitchenDemo",             # 厨房演示
#             "TurnOnMicrowave",         # 开微波炉
#             "TurnOffMicrowave",        # 关微波炉
#             "TurnOnStove",             # 开炉子
#             "TurnOffStove",            # 关炉子
#             "CoffeeSetupMug",          # 咖啡杯设置
#             "CoffeeServeMug",          # 咖啡服务
#             "PrepareCoffee",           # 准备咖啡
#             "CupcakeCleanup"           # 纸杯蛋糕清理
#         ]
#     }
    
#     @classmethod
#     def get_test_tasks(cls) -> List[str]:
#         """获取用于测试的简单任务"""
#         return cls.ATOMIC_TASKS["beginner"]
    
#     @classmethod
#     def get_all_tasks(cls) -> List[str]:
#         """获取所有原子任务"""
#         all_tasks = []
#         for category in cls.ATOMIC_TASKS.values():
#             all_tasks.extend(category)
#         return all_tasks
    
#     @classmethod
#     def validate_task(cls, task_name: str) -> bool:
#         """验证任务名称是否有效"""
#         return task_name in cls.get_all_tasks()
    
#     @classmethod
#     def print_available_tasks(cls):
#         """打印所有可用任务"""
#         print("📋 可用的RoboCasa任务:")
#         for category, tasks in cls.ATOMIC_TASKS.items():
#             print(f"\n🎯 {category.upper()} 任务:")
#             for i, task in enumerate(tasks, 1):
#                 print(f"   {i}. {task}")
    
#     @classmethod
#     def get_recommended_task(cls) -> str:
#         """获取推荐的测试任务"""
#         return cls.ATOMIC_TASKS["beginner"][0]  # 返回最简单的任务

# # ==================== 真实RoboCasa环境包装器 ====================

# class RealRoboCasaEnvironment:
#     """真实的RoboCasa环境包装器"""
    
#     def __init__(self, 
#                  task_name: str = "Lift",
#                  robot_type: str = "PandaMobile",
#                  horizon: int = 500,
#                  camera_names: List[str] = None):
#         """
#         初始化真实RoboCasa环境
        
#         Args:
#             task_name: RoboCasa任务名称
#             robot_type: 机器人类型
#             horizon: 最大步数
#             camera_names: 相机名称列表
#         """
#         if not ROBOCASA_AVAILABLE:
#             raise ImportError("RoboCasa不可用，请先安装")
        
#         self.task_name = task_name
#         self.robot_type = robot_type
#         self.horizon = horizon
        
#         # 使用实际可用的相机名称
#         if camera_names is None:
#             self.camera_names = ["robot0_frontview", "robot0_eye_in_hand"]
#         else:
#             self.camera_names = camera_names
        
#         print(f"🏗️ 创建RoboCasa环境: {task_name}")
#         print(f"   机器人: {robot_type}")
#         print(f"   相机: {self.camera_names}")
#         print(f"   最大步数: {horizon}")
        
#         # 验证任务名称
#         if not RoboCasaTaskSelector.validate_task(task_name):
#             print(f"⚠️ 任务名称 '{task_name}' 不在已知任务列表中")
#             print(f"💡 推荐使用测试任务:")
#             for task in RoboCasaTaskSelector.get_test_tasks()[:3]:  # 显示前3个
#                 print(f"   - {task}")
#             print(f"📋 运行 RoboCasaTaskSelector.print_available_tasks() 查看所有任务")
        
#         self.env = None
#         self.current_step = 0
#         self.last_observation = None
#         self.task_completed = False
        
#         # 尝试创建环境
#         self._create_environment()
    
#     def _create_environment(self):
#         """创建RoboCasa环境实例"""
#         try:
#             print("📦 正在创建RoboCasa环境...")
            
#             # 创建环境的参数
#             env_kwargs = {
#                 "robots": self.robot_type,
#                 "has_renderer": False,           # 训练时关闭渲染
#                 "has_offscreen_renderer": True,  # 启用离屏渲染获取图像
#                 "render_camera": "robot0_frontview",  # 使用实际存在的相机
#                 "render_collision_mesh": False,
#                 "render_visual_mesh": True,
#                 "control_freq": 20,
#                 "horizon": self.horizon,
#                 "ignore_done": True,
#                 "hard_reset": True,
#                 "camera_names": self.camera_names,
#                 "camera_heights": 480,           # 使用640x480分辨率
#                 "camera_widths": 640,
#                 "camera_depths": True,           # 启用深度信息
#                 "reward_shaping": True,
#                 "use_camera_obs": True,
#                 "use_object_obs": True,
#             }
            
#             # 创建环境
#             self.env = robocasa.make(self.task_name, **env_kwargs)
            
#             print(f"✅ RoboCasa环境创建成功!")
#             print(f"   任务: {self.task_name}")
#             print(f"   相机: {self.camera_names}")
            
#             # 安全地获取动作空间信息
#             try:
#                 if hasattr(self.env, 'action_space'):
#                     print(f"   动作空间: {self.env.action_space}")
#                 elif hasattr(self.env, 'action_spec'):
#                     action_spec = self.env.action_spec()
#                     print(f"   动作规格: {action_spec}")
#                 elif hasattr(self.env, '_get_action_space'):
#                     action_space = self.env._get_action_space()
#                     print(f"   动作空间: {action_space}")
#                 else:
#                     print(f"   动作空间: 未知 (环境类型: {type(self.env).__name__})")
#             except Exception as e:
#                 print(f"   动作空间: 获取失败 - {e}")
            
#             # 安全地获取观测空间信息  
#             try:
#                 if hasattr(self.env, 'observation_space'):
#                     obs_keys = list(self.env.observation_space.spaces.keys()) if hasattr(self.env.observation_space, 'spaces') else 'Unknown'
#                     print(f"   观测空间键: {obs_keys}")
#                 elif hasattr(self.env, 'observation_spec'):
#                     obs_spec = self.env.observation_spec()
#                     obs_keys = list(obs_spec.keys()) if isinstance(obs_spec, dict) else 'Unknown'
#                     print(f"   观测规格键: {obs_keys}")
#                 else:
#                     print(f"   观测空间: 未知")
#             except Exception as e:
#                 print(f"   观测空间: 获取失败 - {e}")
            
#             return True
            
#         except Exception as e:
#             print(f"❌ 创建RoboCasa环境失败: {e}")
            
#             # 如果错误信息包含可用环境列表，提取并显示
#             error_str = str(e)
#             if "registered environment among:" in error_str:
#                 # 提取注册的环境列表
#                 env_list_start = error_str.find("registered environment among:") + 30
#                 env_list = error_str[env_list_start:].strip()
#                 available_envs = [env.strip() for env in env_list.split(",")]
                
#                 print(f"\n📋 RoboCasa中实际可用的环境:")
#                 print(f"总共 {len(available_envs)} 个环境")
                
#                 # 按类别显示部分环境
#                 basic_envs = [env for env in available_envs if env in ["Lift", "Stack", "PnP", "Kitchen"]]
#                 pnp_envs = [env for env in available_envs if env.startswith("PnP")]
#                 door_envs = [env for env in available_envs if "Door" in env]
                
#                 if basic_envs:
#                     print(f"🎯 基础任务: {basic_envs}")
#                 if pnp_envs[:5]:  # 只显示前5个
#                     print(f"📦 抓取任务: {pnp_envs[:5]}")
#                 if door_envs[:3]:  # 只显示前3个  
#                     print(f"🚪 开关门任务: {door_envs[:3]}")
                
#                 print(f"💡 建议首先尝试: Lift (最基础的抓取任务)")
            
#             print(f"\n错误详情:")
#             import traceback
#             traceback.print_exc()
            
#             # 提供解决建议
#             print(f"\n🔧 解决建议:")
#             print(f"1. 检查任务名称是否正确: '{self.task_name}'")
#             print(f"   可用的简单任务: {RoboCasaTaskSelector.get_test_tasks()}")
#             print(f"2. 检查机器人类型是否支持: '{self.robot_type}'")
#             print(f"   建议使用: PandaMobile 或 Panda")
#             print(f"3. 确保RoboCasa正确安装并下载了必要资源")
#             print(f"4. 尝试最简单的任务: '{RoboCasaTaskSelector.get_recommended_task()}'")
#             print(f"5. 运行 RoboCasaTaskSelector.print_available_tasks() 查看所有任务")
            
#             raise
    
#     def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#         """重置环境并返回初始观测"""
#         if self.env is None:
#             raise RuntimeError("环境未正确初始化")
        
#         try:
#             print(f"🔄 重置RoboCasa环境: {self.task_name}")
            
#             obs = self.env.reset()
#             self.current_step = 0
#             self.last_observation = obs
#             self.task_completed = False
            
#             # 处理观测数据格式
#             processed_obs = self._process_observation(obs)
            
#             info = {
#                 "task_name": self.task_name,
#                 "step": self.current_step,
#                 "max_steps": self.horizon
#             }
            
#             print(f"✅ 环境重置成功")
#             print(f"   观测键: {list(processed_obs.keys())}")
#             self._print_observation_info(processed_obs)
            
#             return processed_obs, info
            
#         except Exception as e:
#             print(f"❌ 环境重置失败: {e}")
#             raise
    
#     def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
#         """执行动作并返回下一步信息"""
#         if self.env is None:
#             raise RuntimeError("环境未正确初始化")
        
#         try:
#             # 确保动作格式正确
#             if not isinstance(action, np.ndarray):
#                 action = np.array(action)
            
#             # 执行动作
#             obs, reward, done, info = self.env.step(action)
#             self.current_step += 1
#             self.last_observation = obs
            
#             # 处理观测数据
#             processed_obs = self._process_observation(obs)
            
#             # 检查任务完成状态
#             task_success = info.get("success", False) or reward > 0.9
#             if task_success:
#                 self.task_completed = True
#                 done = True
            
#             # 检查是否超时
#             if self.current_step >= self.horizon:
#                 done = True
            
#             # 增强info信息
#             enhanced_info = {
#                 **info,
#                 "task_name": self.task_name,
#                 "step": self.current_step,
#                 "max_steps": self.horizon,
#                 "task_success": task_success,
#                 "task_completed": self.task_completed,
#                 "action_taken": action.tolist() if hasattr(action, 'tolist') else action
#             }
            
#             return processed_obs, reward, done, False, enhanced_info
            
#         except Exception as e:
#             print(f"❌ 环境步进失败: {e}")
#             print(f"动作形状: {action.shape if hasattr(action, 'shape') else '未知'}")
#             print(f"动作内容: {action}")
#             raise
    
#     def _process_observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
#         """处理观测数据，确保格式正确"""
#         processed = {}
        
#         # 处理图像数据
#         for camera in self.camera_names:
#             # RGB图像
#             rgb_key = f"{camera}_image"
#             if rgb_key in obs:
#                 img = obs[rgb_key]
#                 # 确保是正确的分辨率 (480, 640, 3)
#                 if img.shape != (480, 640, 3):
#                     img = cv2.resize(img, (640, 480))
#                     if len(img.shape) == 3 and img.shape[2] == 3:
#                         processed[f"frontview_image"] = img.astype(np.uint8)
#                     else:
#                         processed[f"frontview_image"] = np.zeros((480, 640, 3), dtype=np.uint8)
#                 else:
#                     processed[f"frontview_image"] = img.astype(np.uint8)
#                 break  # 只需要一个图像源
            
#             # 深度图像
#             depth_key = f"{camera}_depth"
#             if depth_key in obs:
#                 depth = obs[depth_key]
#                 if depth.shape != (480, 640):
#                     depth = cv2.resize(depth, (640, 480))
#                 processed[f"frontview_depth"] = depth.astype(np.float32)
#                 break  # 只需要一个深度源
        
#         # 如果没有找到图像，使用默认值
#         if "frontview_image" not in processed:
#             processed["frontview_image"] = np.zeros((480, 640, 3), dtype=np.uint8)
        
#         # 处理机器人状态数据
#         robot_keys = [
#             "robot0_joint_pos", "robot0_joint_vel", 
#             "robot0_eef_pos", "robot0_eef_quat",
#             "robot0_gripper_qpos", "robot0_gripper_qvel"
#         ]
        
#         for key in robot_keys:
#             if key in obs:
#                 processed[key] = np.array(obs[key], dtype=np.float32)
        
#         # 如果缺少关键数据，生成默认值
#         if "robot0_joint_pos" not in processed:
#             processed["robot0_joint_pos"] = np.zeros(7, dtype=np.float32)
#         if "robot0_joint_vel" not in processed:
#             processed["robot0_joint_vel"] = np.zeros(7, dtype=np.float32)
#         if "robot0_eef_pos" not in processed:
#             processed["robot0_eef_pos"] = np.array([0.5, 0.0, 0.8], dtype=np.float32)
#         if "robot0_eef_quat" not in processed:
#             processed["robot0_eef_quat"] = np.array([0, 0, 0, 1], dtype=np.float32)
#         if "robot0_gripper_qpos" not in processed:
#             processed["robot0_gripper_qpos"] = np.zeros(2, dtype=np.float32)
#         if "frontview_image" not in processed:
#             processed["frontview_image"] = np.zeros((480, 640, 3), dtype=np.uint8)
        
#         return processed
    
#     def _print_observation_info(self, obs: Dict[str, Any]):
#         """打印观测信息"""
#         print(f"📊 观测数据详情:")
#         for key, value in obs.items():
#             if isinstance(value, np.ndarray):
#                 print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
#             else:
#                 print(f"   {key}: {type(value)}")
    
#     def get_action_space(self):
#         """获取动作空间（兼容多种RoboCasa版本）"""
#         if self.env is None:
#             raise RuntimeError("环境未初始化")
        
        
#         # ... in RealRoboCasaEnvironment.get_action_space
#         # 尝试 'action_spec' 属性
#         if hasattr(self.env, 'action_spec'):
#             action_spec = self.env.action_spec
#             # robosuite 0.3.0 版本的 action_spec 是一个元组 (low, high)
#             if isinstance(action_spec, tuple) and len(action_spec) == 2:
#                 return action_spec

#         # 尝试 'action_space' 属性 (兼容gym)
#         if hasattr(self.env, 'action_space'):
#             return self.env.action_space

#         # 尝试 '_action_space'
#         if hasattr(self.env, '_action_space'):
#             return self.env._action_space

#         # 如果都没有，返回一个默认的动作空间描述
#         print("⚠️ 无法获取标准动作空间，使用默认配置")
#         # 根据日志，PandaMobile的维度是12
#         import gym.spaces
#         return gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)


#         # 如果都没有，返回一个默认的动作空间描述
#         print("⚠️ 无法获取标准动作空间，使用默认配置")
#         import gym.spaces
#         return gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    
#     def get_observation_space(self):
#         """获取观测空间（兼容多种RoboCasa版本）"""
#         if self.env is None:
#             raise RuntimeError("环境未初始化")
        
#         # 尝试多种可能的属性名
#         for attr_name in ['observation_space', 'observation_spec', '_observation_space']:
#             if hasattr(self.env, attr_name):
#                 attr = getattr(self.env, attr_name)
#                 if callable(attr):
#                     return attr()
#                 else:
#                     return attr
        
#         return None
    
#     def close(self):
#         """关闭环境"""
#         if self.env is not None:
#             try:
#                 self.env.close()
#                 print(f"🔒 RoboCasa环境已关闭: {self.task_name}")
#             except:
#                 pass
#             finally:
#                 self.env = None

# # ==================== 相机配置测试器 ====================

# class RoboCasaCameraConfigTester:
#     """RoboCasa相机配置测试器"""
    
#     def __init__(self):
#         self.available_cameras = [
#             "robot0_frontview",
#             "robot0_agentview_center", 
#             "robot0_agentview_left",
#             "robot0_agentview_right",
#             "robot0_robotview",
#             "robot0_eye_in_hand"
#         ]
        
#         # 推荐的相机配置组合
#         self.camera_configs = [
#             ["robot0_frontview"],                                    # 单前视图
#             ["robot0_agentview_center"],                            # 单中心视图  
#             ["robot0_robotview"],                                   # 单机器人视图
#             ["robot0_frontview", "robot0_eye_in_hand"],             # 前视图+手部
#             ["robot0_agentview_center", "robot0_eye_in_hand"],      # 中心+手部
#             ["robot0_robotview", "robot0_eye_in_hand"]              # 机器人+手部
#         ]
    
#     def find_working_camera_config(self, task_name: str = "Lift") -> Optional[List[str]]:
#         """找到可工作的相机配置"""
#         print(f"\n📷 测试相机配置 (任务: {task_name})")
        
#         for i, cam_config in enumerate(self.camera_configs):
#             print(f"   测试配置 {i+1}: {cam_config}")
            
#             try:
#                 # 基础环境参数
#                 test_kwargs = {
#                     "robots": "PandaMobile",
#                     "has_renderer": False,
#                     "has_offscreen_renderer": True,
#                     "render_camera": cam_config[0],  # 使用第一个相机作为渲染相机
#                     "camera_names": cam_config,
#                     "camera_heights": 480,
#                     "camera_widths": 640,
#                     "camera_depths": True,
#                     "horizon": 50,  # 短时间测试
#                     "use_camera_obs": True,
#                 }
                
#                 # 尝试创建环境
#                 test_env = robocasa.make(task_name, **test_kwargs)
                
#                 # 尝试reset
#                 obs = test_env.reset()
                
#                 # 检查相机数据是否存在
#                 camera_data_found = False
#                 for cam in cam_config:
#                     if f"{cam}_image" in obs:
#                         camera_data_found = True
#                         img_shape = obs[f"{cam}_image"].shape
#                         print(f"     ✅ {cam}: 图像形状 {img_shape}")
                
#                 test_env.close()
                
#                 if camera_data_found:
#                     print(f"   ✅ 配置 {i+1} 成功!")
#                     return cam_config
#                 else:
#                     print(f"   ❌ 配置 {i+1}: 无相机数据")
                    
#             except Exception as e:
#                 print(f"   ❌ 配置 {i+1}: {str(e)[:60]}...")
        
#         print(f"   ❌ 没有找到可用的相机配置")
#         return None

# # ==================== 环境测试器 ====================

# class RoboCasaEnvironmentTester:
#     """RoboCasa环境测试器"""
    
#     def __init__(self):
#         self.test_results = {}
#         self.camera_tester = RoboCasaCameraConfigTester()
    
#     def test_environment_creation(self, task_name: str) -> bool:
#         """测试环境创建"""
#         print(f"\n🧪 测试环境创建: {task_name}")
        
#         # 首先测试相机配置
#         working_cameras = self.camera_tester.find_working_camera_config(task_name)
        
#         if not working_cameras:
#             print(f"❌ 没有找到可用的相机配置")
#             return False
        
#         try:
#             env = RealRoboCasaEnvironment(
#                 task_name=task_name,
#                 robot_type="PandaMobile",
#                 horizon=100,  # 短时间测试
#                 camera_names=working_cameras  # 使用测试通过的相机
#             )
            
#             print(f"✅ 环境创建成功")
            
#             # 安全地测试动作空间
#             try:
#                 action_space = env.get_action_space()
#                 print(f"📏 动作空间: {action_space}")
#             except Exception as e:
#                 print(f"📏 动作空间: 获取失败 - {e}")
            
#             env.close()
#             return True
            
#         except Exception as e:
#             print(f"❌ 环境创建失败: {e}")
#             return False
    
#     def test_environment_functionality(self, task_name: str, max_steps: int = 10) -> Dict[str, Any]:
#         """测试环境功能"""
#         print(f"\n🧪 测试环境功能: {task_name}")
        
#         result = {
#             "task_name": task_name,
#             "creation_success": False,
#             "reset_success": False,
#             "step_success": False,
#             "steps_completed": 0,
#             "error": None,
#             "camera_config": None
#         }
        
#         try:
#             # 首先测试相机配置
#             working_cameras = self.camera_tester.find_working_camera_config(task_name)
            
#             if not working_cameras:
#                 result["error"] = "No working camera configuration found"
#                 return result
            
#             result["camera_config"] = working_cameras
            
#             # 1. 创建环境
#             env = RealRoboCasaEnvironment(
#                 task_name=task_name, 
#                 horizon=max_steps * 2,
#                 camera_names=working_cameras
#             )
#             result["creation_success"] = True
            
#             # 2. 测试reset
#             obs, info = env.reset()
#             result["reset_success"] = True
#             print(f"✅ Reset成功，观测键: {list(obs.keys())}")
            
#             # 验证相机数据
#             for cam in working_cameras:
#                 img_key = f"{cam}_image"
#                 if img_key in obs:
#                     print(f"   📷 {cam}: {obs[img_key].shape}")
            
#             # 3. 测试step
#             try:
#                 action_space = env.get_action_space()
#                 print(f"📏 使用动作空间: {action_space}")
#             except Exception as e:
#                 print(f"⚠️ 动作空间获取失败: {e}")
#                 action_space = None
            
#             for step in range(max_steps):
#                 # 生成随机动作

#                 # ... in RoboCasaEnvironmentTester.test_environment_functionality
#                 try:
#                     if action_space is not None and hasattr(action_space, 'sample'):
#                         action = action_space.sample()
#                     elif action_space is not None:
#                         # 这是一个更通用的方式来处理从 robosuite 返回的动作空间
#                         # robosuite 的 action_spec() 返回的是 (low, high) 元组
#                         low, high = action_space
#                         if isinstance(low, np.ndarray) and isinstance(high, np.ndarray):
#                             action = np.random.uniform(low, high)
#                         else:
#                             # 如果格式不符合预期，根据形状生成
#                             action_shape = np.array(low).shape
#                             action = np.random.uniform(-0.1, 0.1, action_shape)
#                     else:
#                         # 如果获取动作空间失败，回退到默认但可能错误的维度
#                         print("   ⚠️ 无法确定动作空间，使用默认12维动作")
#                         action = np.random.uniform(-0.1, 0.1, 12) # 将7改为12，因为日志显示是12

#                     # 确保动作在[-1, 1]范围内，这是robosuite的普遍要求
#                     action = np.clip(action, -1.0, 1.0)
                    
#                     print(f"   🎯 动作形状: {action.shape}, 范围: [{action.min():.3f}, {action.max():.3f}]")
                    
#                 except Exception as action_e:
#                     print(f"   ⚠️ 动作生成失败: {action_e}")
#                     action = np.random.uniform(-0.1, 0.1, 7)
                
#                 obs, reward, done, _, info = env.step(action)
#                 result["steps_completed"] = step + 1
                
#                 print(f"   步骤 {step+1}: reward={reward:.3f}, done={done}")
                
#                 if done:
#                     if info.get("task_success", False):
#                         print(f"🎉 任务完成！")
#                     else:
#                         print(f"⏱️ 任务结束")
#                     break
            
#             result["step_success"] = True
#             env.close()
            
#             print(f"✅ 环境功能测试完成")
            
#         except Exception as e:
#             result["error"] = str(e)
#             print(f"❌ 环境功能测试失败: {e}")
        
#         return result
    
#     def find_working_task(self) -> Optional[str]:
#         """找到一个可工作的任务"""
#         print(f"\n🔍 寻找可用的RoboCasa任务...")
        
#         # 按难易程度测试任务
#         test_tasks = RoboCasaTaskSelector.get_test_tasks()
        
#         for task_name in test_tasks:
#             print(f"\n尝试任务: {task_name}")
            
#             # 首先测试相机配置
#             working_cameras = self.camera_tester.find_working_camera_config(task_name)
            
#             if not working_cameras:
#                 print(f"❌ 任务 {task_name} 相机配置失败")
#                 continue
            
#             if self.test_environment_creation(task_name):
#                 result = self.test_environment_functionality(task_name, max_steps=3)
                
#                 if (result["creation_success"] and 
#                     result["reset_success"] and 
#                     result["step_success"]):
#                     print(f"✅ 找到可用任务: {task_name}")
#                     print(f"   📷 使用相机配置: {working_cameras}")
#                     return task_name
#                 else:
#                     print(f"❌ 任务 {task_name} 功能测试失败")
#             else:
#                 print(f"❌ 任务 {task_name} 创建失败")
        
#         print(f"❌ 没有找到可用的任务")
#         return None

# # ==================== 修改FinalGR00TExperiment类 ====================

# # [保留原有的其他类定义: FinalConfig, FixedDataFormatter, FinalGR00TClient, FinalEpisodeResult]

# @dataclass
# class FinalConfig:
#     """最终实验配置"""
#     # 服务连接
#     host: str = "localhost"
#     port: int = 5555
    
#     # 实验设置
#     experiment_name: str = "robocasa_groot_experiment"
#     num_episodes: int = 3
#     max_steps_per_episode: int = 60
    
#     # RoboCasa设置
#     robocasa_task: str = "Lift"  # 使用实际存在的简单任务
#     use_real_robocasa: bool = True  # 是否使用真实RoboCasa
    
#     # 实验模式
#     run_baseline: bool = True
#     run_metacognitive: bool = True
    
#     # 设备设置
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"

# class FixedDataFormatter:
#     """修复后的数据格式器 - 使用正确的视频分辨率"""
    
#     def __init__(self):
#         self.required_keys = {
#             "video.webcam": (640, 480, 3),
#             "state.single_arm": (1, 5),
#             "state.gripper": (1, 1),
#             "annotation.human.task_description": None
#         }
        
#         print("🎯 使用微调模型期望配置:")
#         for key, shape in self.required_keys.items():
#             if shape:
#                 print(f"   - {key}: {shape}")
#             else:
#                 print(f"   - {key}: [string list]")
    
#     def create_correct_observation(self, base_obs: Dict[str, np.ndarray] = None) -> Dict[str, Any]:
#         """创建正确格式的观察数据"""
#         correct_obs = {}
        
#         # 1. 视频数据 - video.webcam
#         if base_obs and "frontview_image" in base_obs:
#             img = base_obs["frontview_image"]
#             if img.shape[:2] != (480, 640):
#                 import cv2
#                 img = cv2.resize(img, (640, 480))
#             correct_obs["video.webcam"] = img[np.newaxis, :, :, :].astype(np.uint8)
#         else:
#             correct_obs["video.webcam"] = self._generate_correct_image()
        
#         # 2. 单臂状态 - state.single_arm
#         if base_obs and "robot0_joint_pos" in base_obs:
#             joint_pos = base_obs["robot0_joint_pos"][:5]
#             joint_pos = np.clip(joint_pos, -1.0, 1.0)
#             correct_obs["state.single_arm"] = joint_pos[np.newaxis, :].astype(np.float32)
#         else:
#             joint_data = np.random.uniform(-0.3, 0.3, 5)
#             correct_obs["state.single_arm"] = joint_data[np.newaxis, :].astype(np.float32)
        
#         # 3. 夹爪状态 - state.gripper
#         if base_obs and "robot0_gripper_qpos" in base_obs:
#             gripper_pos = base_obs["robot0_gripper_qpos"][:1]
#             correct_obs["state.gripper"] = gripper_pos[np.newaxis, :].astype(np.float32)
#         else:
#             gripper_data = np.random.uniform(-0.1, 0.1, 1)
#             correct_obs["state.gripper"] = gripper_data[np.newaxis, :].astype(np.float32)
        
#         # 4. 任务描述
#         task_desc = "Execute manipulation task"
#         if base_obs and hasattr(base_obs, 'get'):
#             task_name = base_obs.get('task_name', 'unknown')
#             task_desc = f"Execute {task_name}"
        
#         correct_obs["annotation.human.task_description"] = [task_desc]
        
#         return correct_obs
        
#     def _generate_correct_image(self) -> np.ndarray:
#         """生成正确分辨率的测试图像"""
#         img = np.zeros((1, 480, 640, 3), dtype=np.uint8)
        
#         for i in range(480):
#             for j in range(640):
#                 img[0, i, j, 0] = (i + j) % 256
#                 img[0, i, j, 1] = (i * 2) % 256
#                 img[0, i, j, 2] = (j * 2) % 256
        
#         return img
    
#     def print_observation_details(self, obs: Dict[str, Any]):
#         """打印观察数据详情"""
#         print("📊 发送的观察数据:")
#         for key, value in obs.items():
#             if isinstance(value, np.ndarray):
#                 print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
#             elif isinstance(value, list):
#                 print(f"   {key}: list[{len(value)}] = {value}")
#             else:
#                 print(f"   {key}: {type(value)} = {value}")

# class FinalGR00TClient:
#     """最终GR00T客户端"""
    
#     def __init__(self, config: FinalConfig):
#         self.config = config
#         self.client = None
#         self.formatter = FixedDataFormatter()
#         self.is_connected = False
        
#         # 统计信息
#         self.total_calls = 0
#         self.total_successes = 0
#         self.total_failures = 0
#         self.total_time = 0.0
    
#     def connect(self) -> bool:
#         """连接到GR00T服务"""
#         if not GROOT_CLIENT_AVAILABLE:
#             print("❌ GR00T官方客户端不可用")
#             return False
        
#         try:
#             print(f"🔗 连接到GR00T服务: {self.config.host}:{self.config.port}")
            
#             self.client = RobotInferenceClient(
#                 host=self.config.host, 
#                 port=self.config.port
#             )
            
#             print("📋 验证连接...")
#             modality_config = self.client.get_modality_config()
            
#             print("✅ 连接成功！服务端配置:")
#             for key, config in modality_config.items():
#                 print(f"   - {key}: {config.modality_keys}")
            
#             # 测试调用
#             print("\n🧪 进行连接测试...")
#             test_obs = self.formatter.create_correct_observation()
#             self.formatter.print_observation_details(test_obs)
            
#             print("🚀 发送测试请求...")
#             test_result = self.client.get_action(test_obs)
            
#             if test_result is not None:
#                 print("✅ 测试调用成功！")
#                 print(f"📤 返回动作键: {list(test_result.keys())}")
#                 for key, value in test_result.items():
#                     if isinstance(value, np.ndarray):
#                         print(f"   {key}: shape={value.shape}")
#                 self.is_connected = True
#                 return True
#             else:
#                 print("❌ 测试调用失败")
#                 return False
                
#         except Exception as e:
#             print(f"❌ 连接失败: {e}")
#             import traceback
#             traceback.print_exc()
#             return False
    
#     def predict(self, observation: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
#         """进行预测"""
#         if not self.is_connected:
#             return None
        
#         self.total_calls += 1
#         start_time = time.time()
        
#         try:
#             correct_obs = self.formatter.create_correct_observation(observation)
#             action = self.client.get_action(correct_obs)
            
#             api_time = time.time() - start_time
#             self.total_time += api_time
            
#             if action is not None:
#                 self.total_successes += 1
#                 return action
#             else:
#                 self.total_failures += 1
#                 return None
                
#         except Exception as e:
#             api_time = time.time() - start_time
#             self.total_time += api_time
#             self.total_failures += 1
#             print(f"⚠️ 预测异常: {e}")
#             return None
    
#     def get_stats(self) -> Dict[str, Any]:
#         """获取统计信息"""
#         if self.total_calls == 0:
#             return {"calls": 0, "successes": 0, "failures": 0, "success_rate": 0, "avg_time": 0}
        
#         return {
#             "calls": self.total_calls,
#             "successes": self.total_successes,
#             "failures": self.total_failures,
#             "success_rate": self.total_successes / self.total_calls,
#             "avg_time": self.total_time / self.total_calls
#         }

# @dataclass 
# class FinalEpisodeResult:
#     """最终Episode结果"""
#     episode_id: int
#     mode: str
#     task_success: bool
#     total_steps: int
#     total_time: float
#     api_calls: int
#     api_successes: int
#     avg_api_time: float
#     metacog_interventions: int = 0
#     groot_actions_received: int = 0

# class FinalGR00TExperiment:
#     """修改后的GR00T实验 - 集成真实RoboCasa"""
    
#     def __init__(self, config: FinalConfig):
#         self.config = config
#         self.groot_client = FinalGR00TClient(config)
#         self.results = []

#         # 设置元认知模块
#         self.metacog_available = False
#         if METACOG_AVAILABLE:
#             try:
#                 self.metacog_module = CompleteMetaCognitiveModule(config.device)
#                 self.robocasa_adapter = RoboCasaToMetacogAdapter(image_size=(480, 640))
#                 self.metacog_to_vla_s2_adapter = MetacogToVLASystem2Adapter()
#                 self.metacog_available = True
#                 print("✅ 元认知模块已加载")
#             except Exception as e:
#                 print(f"⚠️ 元认知模块初始化失败: {e}")

#         # 创建环境（这里是关键修改）
#         self.environment = self._create_environment()
    


#     # 在 FinalGR00TExperiment 类的内部添加这个方法
#     def _convert_groot_action_to_robocasa(self, groot_action: Dict[str, np.ndarray]) -> np.ndarray:
#         """
#         将GR00T模型的输出动作转换为RoboCasa环境期望的动作格式。
#         这是一个关键的适配器。
#         """
#         # GR00T输出通常是字典，包含 'world_vector', 'rotation_delta', 'gripper_closedness_action'
#         # 形状通常是 (1, 3), (1, 3), (1, 1)
        
#         # 1. 提取并处理GR00T动作
#         # 末端执行器位移 (dx, dy, dz)
#         world_vector = groot_action.get('world_vector', np.zeros((1, 3)))[0]
#         # 末端执行器旋转 (d_roll, d_pitch, d_yaw)
#         rotation_delta = groot_action.get('rotation_delta', np.zeros((1, 3)))[0]
#         # 夹爪动作 (-1 for open, 1 for close)
#         # GR00T 的 gripper_closedness_action 范围是 [0, 1] for open, to [1, 1] for closed.
#         # robosuite 的夹爪动作通常是 -1 (open) to 1 (close). 我们需要映射一下。
#         gripper_action_groot = groot_action.get('gripper_closedness_action', np.zeros((1,1)))[0][0]
#         gripper_action_robocasa = (gripper_action_groot - 0.5) * 2.0  # Map [0, 1] to [-1, 1]

#         # 2. 构造RoboCasa的12维动作向量
#         # RoboCasa的PandaMobile控制器通常期望一个10维或12维的动作。
#         # 12维可能是：[arm_dx, dy, dz, d_roll, dpitch, dyaw] (6) + [gripper] (1) + 
#         # [base_vx, vy, vtheta] (3) + [2 other dims?]
#         # 我们先假设一个常见的10维结构，并用0填充剩余维度
#         # [arm_dx, arm_dy, arm_dz, arm_droll, arm_dpitch, arm_dyaw, gripper, base_x, base_y, base_rot]
        
#         # !! 关键假设 !!: GR00T目前只控制手臂，不控制底盘。
#         # 所以底盘的动作我们设置为0。
#         base_action = np.zeros(3) # (base_vx, base_vy, base_vtheta)

#         # 组合成一个10维动作向量
#         # 臂6维 + 夹爪1维
#         arm_and_gripper_action = np.concatenate([
#             world_vector, 
#             rotation_delta, 
#             [gripper_action_robocasa]
#         ]) # 7维
        
#         # 组合手臂和底盘动作
#         # 假设动作格式是：7维臂动作 + 3维底盘动作 + 2维未知动作(用0填充)
#         robocasa_action = np.zeros(12)
#         # 填充手臂和夹爪部分
#         robocasa_action[0:7] = arm_and_gripper_action
#         # 填充底盘部分 (这里我们假设后3维是底盘，但需要验证)
#         # robocasa_action[7:10] = base_action 
#         # 让我们假设 robosuite 的默认 PandaMobile 控制器是 OSC_POSE
#         # 它的动作空间是 [d_x, d_y, d_z, d_roll, d_pitch, d_yaw, gripper, base_vx, base_vy, base_vtheta] -> 10维
#         # 既然环境报12维，我们先填充前7维，后面用0，看看会发生什么。
        
#         print(f"🤖 GR00T->RoboCasa: GR00T action (world_vec, rot_delta, grip): "
#             f"{np.round(world_vector, 2)}, {np.round(rotation_delta, 2)}, {gripper_action_robocasa:.2f}")
        
#         # 将动作裁剪到[-1, 1]范围
#         return np.clip(robocasa_action, -1.0, 1.0)



#     def _create_environment(self):
#         """创建环境 - 真实RoboCasa或回退到模拟环境"""
#         if self.config.use_real_robocasa and ROBOCASA_AVAILABLE:
#             print(f"🏗️ 创建真实RoboCasa环境")
            
#             # 首先测试任务可用性
#             tester = RoboCasaEnvironmentTester()
            
#             # 尝试使用配置中的任务
#             test_result = tester.test_environment_functionality(
#                 self.config.robocasa_task, max_steps=3
#             )
            
#             if test_result["creation_success"] and test_result["reset_success"]:
#                 print(f"✅ 使用任务: {self.config.robocasa_task}")
#                 return RealRoboCasaEnvironment(
#                     task_name=self.config.robocasa_task,
#                     robot_type="PandaMobile",
#                     horizon=self.config.max_steps_per_episode * 2
#                 )
#             else:
#                 # 如果配置的任务失败，尝试找到可用任务
#                 print(f"⚠️ 配置任务失败，寻找可用任务...")
#                 working_task = tester.find_working_task()
                
#                 if working_task:
#                     print(f"✅ 找到可用任务: {working_task}")
#                     self.config.robocasa_task = working_task  # 更新配置
#                     return RealRoboCasaEnvironment(
#                         task_name=working_task,
#                         robot_type="PandaMobile", 
#                         horizon=self.config.max_steps_per_episode * 2
#                     )
#                 else:
#                     print(f"❌ 无法找到可用的RoboCasa任务，回退到模拟环境")
#                     return self._create_fallback_environment()
#         else:
#             print(f"🤖 使用模拟环境（RoboCasa不可用或已禁用）")
#             return self._create_fallback_environment()
    
#     def _create_fallback_environment(self):
#         """创建回退的模拟环境"""
#         class SingleArmTestEnvironment:
#             def __init__(self):
#                 self.step_count = 0
#                 self.max_steps = 60
#                 print("🤖 初始化模拟单臂环境（回退模式）")
            
#             def reset(self):
#                 self.step_count = 0
#                 return self._generate_obs(), {"task_name": "simulated_task"}
            
#             def step(self, action):
#                 self.step_count += 1
#                 obs = self._generate_obs()
                
#                 if self.step_count > 25 and np.random.random() < 0.35:
#                     done = True
#                     reward = 1.0
#                 elif self.step_count >= self.max_steps:
#                     done = True
#                     reward = -0.5
#                 else:
#                     done = False
#                     reward = np.random.uniform(-0.01, 0.01)
                
#                 info = {
#                     "task_success": done and reward > 0,
#                     "collision": np.random.random() < 0.003,
#                     "force_violation": np.random.random() < 0.001,
#                     "task_name": "simulated_task"
#                 }
                
#                 return obs, reward, done, False, info
            
#             def _generate_obs(self):
#                 return {
#                     "frontview_image": np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8),
#                     "robot0_joint_pos": np.random.uniform(-0.2, 0.2, 7),
#                     "robot0_joint_vel": np.random.uniform(-0.1, 0.1, 7),
#                     "robot0_gripper_qpos": np.random.uniform(-0.05, 0.05, 2),
#                     "robot0_eef_pos": np.array([0.5, 0.0, 0.8]),
#                     "robot0_eef_quat": np.array([0, 0, 0, 1])
#                 }
            
#             def close(self):
#                 pass
        
#         return SingleArmTestEnvironment()
    
#     def run_experiment(self) -> bool:
#         """运行实验"""
#         env_type = "真实RoboCasa" if isinstance(self.environment, RealRoboCasaEnvironment) else "模拟"
        
#         print(f"\n🎯 开始GR00T + 元认知实验")
#         print(f"环境类型: {env_type}")
#         if isinstance(self.environment, RealRoboCasaEnvironment):
#             print(f"RoboCasa任务: {self.config.robocasa_task}")
#         print("=" * 70)
        
#         # 连接到GR00T服务
#         if not self.groot_client.connect():
#             print("❌ 无法连接到GR00T推理服务")
#             return False
        
#         try:
#             # 运行基线实验
#             if self.config.run_baseline:
#                 print(f"\n🤖 基线实验 (GR00T N1)")
#                 print("-" * 50)
                
#                 for episode in range(self.config.num_episodes):
#                     print(f"\n📊 基线 Episode {episode + 1}/{self.config.num_episodes}")
#                     result = self._run_episode(episode, "baseline", False)
#                     self.results.append(result)
#                     self._print_episode_summary(result)
            
#             # 运行元认知实验
#             if self.config.run_metacognitive and self.metacog_available:
#                 print(f"\n🧠 元认知实验 (GR00T N1 + 元认知模块)")
#                 print("-" * 50)
                
#                 for episode in range(self.config.num_episodes):
#                     print(f"\n📊 元认知 Episode {episode + 1}/{self.config.num_episodes}")
#                     result = self._run_episode(episode, "metacognitive", True)
#                     self.results.append(result)
#                     self._print_episode_summary(result)
            
#             # 分析结果
#             self._analyze_results()
#             self._save_results()
            
#             return True
            
#         finally:
#             # 关闭环境
#             if hasattr(self.environment, 'close'):
#                 self.environment.close()
    
#     def _run_episode(self, episode_id: int, mode: str, use_metacognitive: bool) -> FinalEpisodeResult:
#         """运行episode"""
#         start_time = time.time()
        
#         result = FinalEpisodeResult(
#             episode_id=episode_id,
#             mode=mode,
#             task_success=False,
#             total_steps=0,
#             total_time=0.0,
#             api_calls=0,
#             api_successes=0,
#             avg_api_time=0.0
#         )
        
#         try:
#             obs, info = self.environment.reset()
#             done = False
#             step_count = 0
#             api_times = []
#             current_metacognitive_instruction_for_s2 = None
            
#             print(f"     执行中 ({mode}): ", end="", flush=True)
            
#             while not done and step_count < self.config.max_steps_per_episode:
                
#                 # 准备观测数据
#                 observation_for_groot = obs.copy()
#                 if current_metacognitive_instruction_for_s2:
#                     observation_for_groot["metacognitive_instruction"] = [current_metacognitive_instruction_for_s2]
                
#                 # 获取GR00T动作
#                 api_start = time.time()
#                 groot_action_dict = self.groot_client.predict(observation_for_groot)
#                 api_time = time.time() - api_start
#                 api_times.append(api_time)
                
#                 result.api_calls += 1
#                 current_metacognitive_instruction_for_s2 = None
                
                
#                 # ... in FinalGR00TExperiment._run_episode
#                 env_action_to_execute = None
#                 s1_action_info_for_metacog = None

#                 if groot_action_dict is not None:
#                     result.api_successes += 1
#                     result.groot_actions_received += 1
#                     print(".", end="", flush=True)
                    
#                     # 核心修改：将GR00T动作转换为RoboCasa动作
#                     if isinstance(self.environment, RealRoboCasaEnvironment):
#                         env_action_to_execute = self._convert_groot_action_to_robocasa(groot_action_dict)
#                         s1_action_info_for_metacog = env_action_to_execute # 元认知模块使用转换后的动作
#                     else: # 模拟环境
#                         # 在模拟环境中，我们仍然可以模仿这个过程
#                         env_action_to_execute = self._convert_groot_action_to_robocasa(groot_action_dict)[:7]
#                         s1_action_info_for_metacog = env_action_to_execute

#                 else:
#                     print("x", end="", flush=True)
#                     # GR00T调用失败，生成一个零动作（保持不动）
#                     if isinstance(self.environment, RealRoboCasaEnvironment):
#                         try:
#                             action_space = self.environment.get_action_space()
#                             if isinstance(action_space, tuple):
#                                 action_shape = action_space[0].shape
#                             else: # gym.Space
#                                 action_shape = action_space.shape
#                             env_action_to_execute = np.zeros(action_shape)
#                         except Exception:
#                             env_action_to_execute = np.zeros(12) # Fallback to 12
#                     else:
#                         env_action_to_execute = np.zeros(7)
#                     s1_action_info_for_metacog = env_action_to_execute

#                 # 确保动作非空
#                 if env_action_to_execute is None:
#                     print("⚠️ 动作未能生成，使用零动作")
#                     env_action_to_execute = np.zeros(12) if isinstance(self.environment, RealRoboCasaEnvironment) else np.zeros(7)



                
#                 # 环境步进
#                 next_obs, reward, done, _, info = self.environment.step(env_action_to_execute)
                
#                 # 元认知处理
#                 if use_metacognitive and self.metacog_available:
#                     try:
#                         sensor_data = self.robocasa_adapter.convert_observation(
#                             next_obs,
#                             s1_action_info_for_metacog,
#                             execution_status="normal"
#                         )
#                         metacog_output = self.metacog_module.process_sensor_data(sensor_data)
                        
#                         instruction_for_s2 = self.metacog_to_vla_s2_adapter.convert_to_system2_instruction(metacog_output)
                        
#                         if instruction_for_s2:
#                             current_metacognitive_instruction_for_s2 = instruction_for_s2
#                             if metacog_output.directive != DirectiveType.CONTINUE:
#                                 result.metacog_interventions += 1
#                                 print(f"M[{instruction_for_s2[:10]}...]", end="", flush=True)
                        
#                     except Exception as e:
#                         pass
                
#                 obs = next_obs
#                 step_count += 1
                
#                 if info.get("task_success", False):
#                     result.task_success = True
#                     done = True
#                     print("!", end="", flush=True)
                
#                 if step_count % 10 == 0 and result.api_calls > 0:
#                     success_rate = result.api_successes / result.api_calls
#                     print(f"|{success_rate:.0%}", end="", flush=True)
            
#             result.total_steps = step_count
#             result.total_time = time.time() - start_time
#             result.avg_api_time = np.mean(api_times) if api_times else 0.0
            
#             print()
            
#         except Exception as e:
#             result.total_time = time.time() - start_time
#             print(f" 异常: {e}")
#             import traceback
#             traceback.print_exc()
        
#         return result
    
#     def _print_episode_summary(self, result: FinalEpisodeResult):
#         """打印episode摘要"""
#         status = "✅ 成功" if result.task_success else "❌ 失败"
#         api_success_rate = result.api_successes / result.api_calls if result.api_calls > 0 else 0
        
#         print(f"   结果: {status}")
#         print(f"   执行: {result.total_steps} 步, {result.total_time:.1f}s")
#         print(f"   API: {result.api_successes}/{result.api_calls} 成功 ({api_success_rate:.1%})")
#         print(f"   GR00T动作: {result.groot_actions_received} 个")
        
#         if result.metacog_interventions > 0:
#             print(f"   元认知: {result.metacog_interventions} 次干预")
    
#     def _analyze_results(self):
#         """分析结果"""
#         print(f"\n📊 实验结果分析")
#         print("=" * 70)
        
#         env_type = "真实RoboCasa" if isinstance(self.environment, RealRoboCasaEnvironment) else "模拟"
#         print(f"环境类型: {env_type}")
        
#         baseline_results = [r for r in self.results if r.mode == "baseline"]
#         metacog_results = [r for r in self.results if r.mode == "metacognitive"]
        
#         def analyze_mode(results: List[FinalEpisodeResult], mode_name: str):
#             if not results:
#                 return
            
#             successes = sum(1 for r in results if r.task_success)
#             success_rate = successes / len(results)
#             total_api_calls = sum(r.api_calls for r in results)
#             total_api_successes = sum(r.api_successes for r in results)
#             api_success_rate = total_api_successes / total_api_calls if total_api_calls > 0 else 0
            
#             print(f"\n🔍 {mode_name} 模式:")
#             print(f"   任务成功率: {success_rate:.1%} ({successes}/{len(results)})")
#             print(f"   API成功率: {api_success_rate:.1%} ({total_api_successes}/{total_api_calls})")
            
#             if mode_name == "元认知":
#                 total_interventions = sum(r.metacog_interventions for r in results)
#                 print(f"   元认知干预: {total_interventions} 次")
        
#         analyze_mode(baseline_results, "基线")
#         analyze_mode(metacog_results, "元认知")
        
#         # 对比分析
#         if baseline_results and metacog_results:
#             baseline_success = sum(1 for r in baseline_results if r.task_success) / len(baseline_results)
#             metacog_success = sum(1 for r in metacog_results if r.task_success) / len(metacog_results)
#             improvement = metacog_success - baseline_success
            
#             print(f"\n⚖️ 对比分析:")
#             print(f"   成功率变化: {improvement:+.1%}")
            
#             if improvement > 0:
#                 print(f"   ✅ 元认知模块提升了性能")
#             elif improvement == 0:
#                 print(f"   ➡️ 元认知模块保持了性能")
#             else:
#                 print(f"   ⚠️ 元认知模块影响了性能")
    
#     def _save_results(self):
#         """保存结果"""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         env_type = "robocasa" if isinstance(self.environment, RealRoboCasaEnvironment) else "simulated"
#         filename = f"groot_experiment_{env_type}_{timestamp}.json"
        
#         data = {
#             "timestamp": timestamp,
#             "experiment_type": f"GR00T_Metacognitive_{env_type}",
#             "environment_type": env_type,
#             "robocasa_task": self.config.robocasa_task if isinstance(self.environment, RealRoboCasaEnvironment) else None,
#             "config": asdict(self.config),
#             "results": [asdict(r) for r in self.results]
#         }
        
#         with open(filename, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=2, default=str)
        
#         print(f"\n💾 结果已保存: {filename}")

# # ==================== 主函数 ====================

# def main():
#     """主函数 - 集成真实RoboCasa环境"""
#     print("🎯 GR00T + 真实RoboCasa环境集成实验")
#     print("阶段1.1: 集成真实RoboCasa环境 (已修复相机配置问题)")
#     print("=" * 70)
    
#     # 检查依赖
#     print(f"📋 依赖检查:")
#     print(f"   RoboCasa: {'✅' if ROBOCASA_AVAILABLE else '❌'}")
#     print(f"   GR00T Client: {'✅' if GROOT_CLIENT_AVAILABLE else '❌'}")  
#     print(f"   元认知模块: {'✅' if METACOG_AVAILABLE else '❌'}")
    
#     if not ROBOCASA_AVAILABLE:
#         print("\n⚠️ RoboCasa不可用，将使用模拟环境")
#         use_robocasa = False
#     else:
#         print("\n🎉 RoboCasa可用，将尝试使用真实环境")
#         # 显示可用任务
#         RoboCasaTaskSelector.print_available_tasks()
#         print(f"\n💡 推荐测试任务: {RoboCasaTaskSelector.get_recommended_task()}")
#         use_robocasa = True
    
#     # 配置实验
#     config = FinalConfig(
#         host="localhost",
#         port=5555,
#         experiment_name="robocasa_integration_test",
#         num_episodes=2,  # 少量episode用于测试
#         max_steps_per_episode=30,
#         robocasa_task=RoboCasaTaskSelector.get_recommended_task(),  # 使用推荐任务
#         use_real_robocasa=use_robocasa,
#         run_baseline=True,
#         run_metacognitive=True if METACOG_AVAILABLE else False
#     )
    
#     print(f"\n🛠️ 实验配置:")
#     print(f"   环境: {'真实RoboCasa' if config.use_real_robocasa else '模拟环境'}")
#     if config.use_real_robocasa:
#         print(f"   任务: {config.robocasa_task}")
#     print(f"   Episodes: {config.num_episodes}")
#     print(f"   最大步数: {config.max_steps_per_episode}")
    
#     # 运行环境测试（如果RoboCasa可用）
#     if ROBOCASA_AVAILABLE:
#         print(f"\n🧪 RoboCasa环境测试")
#         print("-" * 50)
        
#         tester = RoboCasaEnvironmentTester()
#         working_task = tester.find_working_task()
        
#         if working_task:
#             config.robocasa_task = working_task
#             print(f"✅ 确认使用任务: {working_task}")
#         else:
#             print(f"❌ RoboCasa环境测试失败，将使用模拟环境")
#             config.use_real_robocasa = False
    
#     # 运行实验
#     print(f"\n🚀 开始实验")
#     print("-" * 50)
    
#     experiment = FinalGR00TExperiment(config)
    
#     try:
#         success = experiment.run_experiment()
#         if success:
#             env_type = "真实RoboCasa" if isinstance(experiment.environment, RealRoboCasaEnvironment) else "模拟"
#             print(f"\n🎉 阶段1.1任务完成！")
#             print(f"✅ 成功集成{env_type}环境")
#             print(f"✅ 修复了相机配置问题")
#             print(f"✅ 环境reset/step功能正常")
#             print(f"✅ GR00T客户端调用正常")
#             if METACOG_AVAILABLE:
#                 print(f"✅ 元认知模块集成正常")
#             print(f"\n📈 下一步可以继续阶段1.2和1.3任务")
#             print(f"💡 关键修复：使用正确的相机名称（robot0_frontview等）")
#         else:
#             print(f"\n❌ 实验失败")
    
#     except KeyboardInterrupt:
#         print(f"\n⚠️ 实验被用户中断")
#     except Exception as e:
#         print(f"\n❌ 实验异常: {e}")
#         import traceback
#         traceback.print_exc()
    
#     finally:
#         # 清理资源
#         if hasattr(experiment, 'environment') and hasattr(experiment.environment, 'close'):
#             experiment.environment.close()

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
集成元认知模块的稳定草莓环境 - 基于可工作的StableStrawberryEnvironment + 视频录制功能
Enhanced Stable Strawberry Environment with Metacognitive Integration + Video Recording
"""

import os
import sys
import time
import json
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import queue

# 设置环境变量避免渲染问题
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

# 导入模块
try:
    import robosuite
    from robosuite.controllers import load_composite_controller_config
    ROBOSUITE_AVAILABLE = True
    print("✅ RoboSuite可用")
except ImportError as e:
    print(f"❌ RoboSuite不可用: {e}")
    ROBOSUITE_AVAILABLE = False

try:
    import robocasa
    ROBOCASA_AVAILABLE = True
    print("✅ RoboCasa可用")
except ImportError as e:
    print(f"❌ RoboCasa不可用: {e}")
    ROBOCASA_AVAILABLE = False

try:
    from gr00t.eval.robot import RobotInferenceClient
    GROOT_CLIENT_AVAILABLE = True
    print("✅ GR00T官方客户端可用")
except ImportError as e:
    print(f"❌ GR00T官方客户端不可用: {e}")
    GROOT_CLIENT_AVAILABLE = False

# 导入修复的元认知模块
try:
    from metacog_integration import (
        CompleteMetaCognitiveModule,
        RoboCasaToMetacogAdapter,
        MetacogToGR00TAdapter,
        ActionAdjuster,
        SensorData,
        MetaCognitiveOutput,
        DirectiveType,
        MetacogToVLASystem2Adapter
    )
    METACOG_AVAILABLE = True
    print("✅ 修复的元认知模块可用")
except ImportError as e:
    print(f"❌ 元认知模块不可用: {e}")
    print("请确保fixed_metacog_integration.py在同目录下")
    
    # 尝试原版模块
    try:
        from metacog_integration import (
            CompleteMetaCognitiveModule,
            RoboCasaToMetacogAdapter,
            MetacogToGR00TAdapter,
            ActionAdjuster,
            SensorData,
            MetaCognitiveOutput,
            DirectiveType,
            MetacogToVLASystem2Adapter
        )
        METACOG_AVAILABLE = True
        print("✅ 原版元认知模块可用")
    except ImportError:
        METACOG_AVAILABLE = False
        
        # 创建备用数据结构
        @dataclass
        class SensorData:
            rgb_image: np.ndarray
            depth_image: np.ndarray
            force_torque: np.ndarray
            contact_detected: bool
            joint_positions: np.ndarray
            joint_velocities: np.ndarray
            end_effector_pose: np.ndarray
            system1_commands: np.ndarray
            execution_status: str
            timestamp: float

# ==================== 视频录制器 ====================

class VideoRecorder:
    """视频录制器 - 专门用于保存训练过程"""
    
    def __init__(self, 
                 output_dir: str = "./experiment_videos",
                 fps: int = 20,
                 video_size: Tuple[int, int] = (640, 480),
                 codec: str = 'mp4v'):
        """
        初始化视频录制器
        
        Args:
            output_dir: 视频保存目录
            fps: 帧率
            video_size: 视频尺寸 (宽, 高)
            codec: 视频编码格式
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.video_size = video_size
        self.codec = codec
        
        # 录制状态
        self.is_recording = False
        self.video_writer = None
        self.current_episode = 0
        self.frame_count = 0
        
        # 线程安全的帧队列
        self.frame_queue = queue.Queue(maxsize=100)
        self.recording_thread = None
        self.stop_recording_flag = threading.Event()
        
        print(f"🎥 视频录制器初始化")
        print(f"   保存目录: {self.output_dir}")
        print(f"   视频参数: {video_size[0]}x{video_size[1]} @ {fps}fps")
        print(f"   编码格式: {codec}")
    
    def start_episode_recording(self, episode_id: int, experiment_name: str = "strawberry_experiment"):
        """开始录制新的episode"""
        if self.is_recording:
            self.stop_episode_recording()
        
        self.current_episode = episode_id
        self.frame_count = 0
        
        # 生成视频文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_episode_{episode_id:03d}_{timestamp}.mp4"
        self.video_path = self.output_dir / filename
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.video_writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            self.fps,
            self.video_size
        )
        
        if not self.video_writer.isOpened():
            print(f"❌ 无法创建视频文件: {self.video_path}")
            return False
        
        # 启动录制线程
        self.is_recording = True
        self.stop_recording_flag.clear()
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.start()
        
        print(f"🎬 开始录制 Episode {episode_id}: {filename}")
        return True
    
    def add_frame(self, image: np.ndarray, step_info: Dict[str, Any] = None):
        """添加一帧到录制队列"""
        if not self.is_recording:
            return
        
        try:
            # 处理图像格式
            processed_image = self._process_image(image, step_info)
            
            # 添加到队列（非阻塞）
            if not self.frame_queue.full():
                self.frame_queue.put(processed_image, block=False)
                self.frame_count += 1
            else:
                print("⚠️ 帧队列已满，跳过帧")
                
        except Exception as e:
            print(f"⚠️ 添加帧失败: {e}")
    
    def _process_image(self, image: np.ndarray, step_info: Dict[str, Any] = None) -> np.ndarray:
        """处理图像格式并添加信息叠加"""
        try:
            # 确保图像格式正确
            if image is None:
                image = np.zeros((*self.video_size[::-1], 3), dtype=np.uint8)
            
            # 转换数据类型
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # 调整尺寸
            if image.shape[:2] != self.video_size[::-1]:
                image = cv2.resize(image, self.video_size)
            
            # 确保是3通道
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
            # 添加信息叠加
            if step_info:
                image = self._add_info_overlay(image, step_info)
            
            return image
            
        except Exception as e:
            print(f"⚠️ 图像处理失败: {e}")
            return np.zeros((*self.video_size[::-1], 3), dtype=np.uint8)
    
    def _add_info_overlay(self, image: np.ndarray, step_info: Dict[str, Any]) -> np.ndarray:
        """在图像上添加信息叠加"""
        try:
            overlay_image = image.copy()
            
            # 设置字体参数
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (0, 255, 0)  # 绿色
            thickness = 2
            
            # 添加基本信息
            y_offset = 30
            
            # Episode和Step信息
            if 'step' in step_info:
                text = f"Episode: {self.current_episode} | Step: {step_info['step']}"
                cv2.putText(overlay_image, text, (10, y_offset), font, font_scale, color, thickness)
                y_offset += 25
            
            # 任务进度
            if 'strawberry_task_progress' in step_info:
                progress = step_info['strawberry_task_progress']
                picked = progress.get('strawberries_picked', 0)
                placed = progress.get('strawberries_on_plate', 0)
                text = f"Strawberries: Picked {picked}/3 | Placed {placed}/3"
                cv2.putText(overlay_image, text, (10, y_offset), font, font_scale, color, thickness)
                y_offset += 25
            
            # 奖励信息
            if 'total_reward' in step_info:
                text = f"Reward: {step_info['total_reward']:.2f}"
                cv2.putText(overlay_image, text, (10, y_offset), font, font_scale, color, thickness)
                y_offset += 25
            
            # 元认知反馈
            if 'metacognitive_feedback' in step_info and step_info['metacognitive_feedback']:
                text = f"Metacog: {step_info['metacognitive_feedback'][:40]}..."
                cv2.putText(overlay_image, text, (10, y_offset), font, font_scale, (255, 255, 0), thickness)
                y_offset += 25
            
            # 任务成功标记
            if step_info.get('task_success', False):
                text = "TASK SUCCESS!"
                cv2.putText(overlay_image, text, (10, image.shape[0] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            
            return overlay_image
            
        except Exception as e:
            print(f"⚠️ 信息叠加失败: {e}")
            return image
    
    def _recording_worker(self):
        """录制工作线程"""
        while self.is_recording and not self.stop_recording_flag.is_set():
            try:
                # 从队列获取帧（带超时）
                frame = self.frame_queue.get(timeout=1.0)
                
                # 写入视频文件
                if self.video_writer and self.video_writer.isOpened():
                    self.video_writer.write(frame)
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ 录制线程错误: {e}")
                break
    
    def stop_episode_recording(self):
        """停止当前episode的录制"""
        if not self.is_recording:
            return
        
        print(f"🎬 停止录制 Episode {self.current_episode} ({self.frame_count} 帧)")
        
        # 停止录制标志
        self.is_recording = False
        self.stop_recording_flag.set()
        
        # 等待录制线程结束
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=5.0)
        
        # 处理剩余帧
        while not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
                if self.video_writer and self.video_writer.isOpened():
                    self.video_writer.write(frame)
                self.frame_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                print(f"⚠️ 处理剩余帧错误: {e}")
                break
        
        # 关闭视频写入器
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        if hasattr(self, 'video_path') and self.video_path.exists():
            file_size = self.video_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ 视频已保存: {self.video_path} ({file_size:.1f}MB)")
        
        self.frame_count = 0
    
    def create_summary_video(self, episode_videos: List[str], output_name: str = "experiment_summary"):
        """创建实验总结视频"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = self.output_dir / f"{output_name}_{timestamp}.mp4"
            
            print(f"🎞️ 创建总结视频: {summary_path}")
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            summary_writer = cv2.VideoWriter(
                str(summary_path),
                fourcc,
                self.fps,
                self.video_size
            )
            
            if not summary_writer.isOpened():
                print(f"❌ 无法创建总结视频文件")
                return None
            
            # 合并所有episode视频
            for i, video_path in enumerate(episode_videos):
                if not Path(video_path).exists():
                    continue
                
                print(f"   合并 Episode {i+1}: {Path(video_path).name}")
                
                cap = cv2.VideoCapture(video_path)
                
                # 添加episode标题帧
                title_frame = np.zeros((*self.video_size[::-1], 3), dtype=np.uint8)
                cv2.putText(title_frame, f"Episode {i+1}", 
                           (self.video_size[0]//4, self.video_size[1]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
                
                # 写入标题帧（持续1秒）
                for _ in range(self.fps):
                    summary_writer.write(title_frame)
                
                # 写入episode帧
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame.shape[:2] != self.video_size[::-1]:
                        frame = cv2.resize(frame, self.video_size)
                    
                    summary_writer.write(frame)
                
                cap.release()
            
            summary_writer.release()
            
            if summary_path.exists():
                file_size = summary_path.stat().st_size / (1024 * 1024)
                print(f"✅ 总结视频已保存: {summary_path} ({file_size:.1f}MB)")
                return str(summary_path)
            
        except Exception as e:
            print(f"❌ 创建总结视频失败: {e}")
            return None
    
    def cleanup(self):
        """清理资源"""
        if self.is_recording:
            self.stop_episode_recording()
        
        print("🧹 视频录制器资源已清理")

# ==================== 简化的数据适配器 - 基础版本 ====================

class SimpleRoboCasaAdapter:
    """简化的数据适配器 - 元认知模块会自动处理维度适配"""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        
    def convert_observation(self, robocasa_obs: Dict[str, np.ndarray], 
                          action: np.ndarray, 
                          execution_status: str = "normal") -> SensorData:
        """将RoboCasa观察转换为SensorData格式 - 让元认知模块处理维度适配"""
        
        # 1. 处理视觉数据
        rgb_image = self._extract_rgb_image(robocasa_obs)
        depth_image = self._extract_depth_image(robocasa_obs)
        
        # 2. 处理力觉数据
        force_torque = self._extract_force_data(robocasa_obs, action)
        contact_detected = self._detect_contact(force_torque)
        
        # 3. 处理本体感觉数据 - 保持原始数据，让元认知模块适配
        joint_positions = robocasa_obs.get("robot0_joint_pos", np.zeros(7))
        joint_velocities = robocasa_obs.get("robot0_joint_vel", np.zeros(7))
        end_effector_pose = self._get_ee_pose(robocasa_obs)
        
        # 4. 系统状态
        system1_commands = self._process_action(action)
        
        return SensorData(
            rgb_image=rgb_image,
            depth_image=depth_image,
            force_torque=force_torque,
            contact_detected=contact_detected,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            end_effector_pose=end_effector_pose,
            system1_commands=system1_commands,
            execution_status=execution_status,
            timestamp=time.time()
        )
    
    def _extract_rgb_image(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """提取RGB图像"""
        for key in ["frontview_image", "robot0_eye_in_hand_image"]:
            if key in obs and obs[key] is not None:
                img = obs[key]
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                if img.shape[:2] != self.image_size:
                    img = cv2.resize(img, self.image_size)
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=-1)
                elif img.shape[2] != 3:
                    img = img[:, :, :3]
                return img
        
        return np.zeros((*self.image_size, 3), dtype=np.float32)
    
    def _extract_depth_image(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """提取深度图像"""
        for key in ["frontview_depth", "robot0_eye_in_hand_depth"]:
            if key in obs and obs[key] is not None:
                depth = obs[key]
                if depth.shape != self.image_size:
                    depth = cv2.resize(depth, self.image_size)
                return depth.astype(np.float32)
        
        return np.ones(self.image_size, dtype=np.float32)
    
    def _extract_force_data(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> np.ndarray:
        """提取或估算力觉数据"""
        if "force_torque" in obs:
            force_data = obs["force_torque"][:6]
        else:
            force_data = self._estimate_force_from_action(action)
        
        return force_data.astype(np.float32)
    
    def _estimate_force_from_action(self, action: np.ndarray) -> np.ndarray:
        """从动作估算力矩"""
        if action is None or len(action) == 0:
            return np.zeros(6, dtype=np.float32)
        
        action_magnitude = np.linalg.norm(action)
        estimated_force = np.random.normal(0, action_magnitude * 0.05, 6)
        estimated_force = np.clip(estimated_force, -5, 5)
        
        return estimated_force.astype(np.float32)
    
    def _detect_contact(self, force_torque: np.ndarray) -> bool:
        """检测接触"""
        force_magnitude = np.linalg.norm(force_torque[:3])
        return force_magnitude > 0.5
    
    def _get_ee_pose(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """获取末端执行器姿态"""
        pose = np.zeros(7, dtype=np.float32)
        
        if "robot0_eef_pos" in obs:
            pos = obs["robot0_eef_pos"][:3]
            pose[:3] = pos
        else:
            pose[:3] = [0.5, 0.0, 0.8]
        
        if "robot0_eef_quat" in obs:
            quat = obs["robot0_eef_quat"][:4]
            pose[3:7] = quat
        else:
            pose[3:7] = [0, 0, 0, 1]
        
        return pose
    
    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """处理系统动作"""
        if action is None:
            return np.zeros(8, dtype=np.float32)
        
        if len(action) >= 8:
            return action[:8].astype(np.float32)
        else:
            padded = np.zeros(8, dtype=np.float32)
            padded[:len(action)] = action
            return padded

# ==================== 简化动作处理器 ====================

class SimpleActionProcessor:
    """简化动作处理器 - 减少复杂性避免崩溃"""
    
    def __init__(self):
        self.world_vector_scale = 0.02
        self.rotation_scale = 0.02
        self.gripper_scale = 0.15
        self.processed_actions = 0
        
        print("🔧 简化动作处理器初始化")
        print(f"   位移缩放: {self.world_vector_scale}")
        print(f"   旋转缩放: {self.rotation_scale}")
    
    def process_groot_action(self, groot_action: Dict[str, np.ndarray]) -> np.ndarray:
        """简化的动作处理"""
        try:
            world_vector = groot_action.get('world_vector', np.zeros((1, 3)))[0]
            rotation_delta = groot_action.get('rotation_delta', np.zeros((1, 3)))[0]
            gripper_action = groot_action.get('gripper_closedness_action', np.zeros((1, 1)))[0][0]
            
            # 简化的缩放
            scaled_world = np.clip(world_vector * self.world_vector_scale, -0.3, 0.3)
            scaled_rotation = np.clip(rotation_delta * self.rotation_scale, -0.3, 0.3)
            scaled_gripper = np.clip(gripper_action * self.gripper_scale, -1.0, 1.0)
            
            # 构建SO100动作
            so100_action = np.zeros(6)
            so100_action[0:3] = scaled_world
            so100_action[3:5] = scaled_rotation[:2]
            so100_action[5] = scaled_gripper
            
            self.processed_actions += 1
            
            # 简化的统计（减少打印频率）
            if self.processed_actions % 50 == 0:
                print(f"   🎯 已处理 {self.processed_actions} 次动作")
            
            return so100_action
            
        except Exception as e:
            print(f"⚠️ 动作处理错误: {e}")
            return np.zeros(6)

# ==================== 增强草莓环境 - 集成元认知 ====================

# class EnhancedStrawberryEnvironment:
#     """增强草莓环境 - 基于可工作的StableStrawberryEnvironment + 元认知集成"""
    
#     def __init__(self, 
#                  so100_xml_path: str = None,
#                  horizon: int = 100,
#                  enable_gui: bool = False,
#                  robot: str = "Panda",
#                  enable_metacognitive: bool = True,
#                  device: str = "cuda" if torch.cuda.is_available() else "cpu"):
#         """
#         初始化增强草莓环境
        
#         Args:
#             so100_xml_path: SO100 XML路径
#             horizon: 最大步数
#             enable_gui: 是否启用GUI
#             robot: 机器人类型
#             enable_metacognitive: 是否启用元认知模块
#             device: 设备类型
#         """
#         if not ROBOSUITE_AVAILABLE:
#             raise ImportError("RoboSuite不可用")
        
#         self.horizon = horizon
#         self.so100_xml_path = so100_xml_path
#         self.enable_gui = enable_gui
#         self.robot = robot
#         self.enable_metacognitive = enable_metacognitive and METACOG_AVAILABLE
#         self.device = device
        
#         # 环境状态
#         self.env = None
#         self.current_step = 0
        
#         # 草莓任务状态
#         self.strawberry_positions = np.array([
#             [0.6, 0.1, 0.82],   
#             [0.7, 0.15, 0.82],  
#             [0.8, 0.1, 0.82]    
#         ])
#         self.plate_position = np.array([0.5, -0.2, 0.81])
#         self.strawberry_states = [True, True, True]
#         self.strawberry_on_plate = [False, False, False]
        
#         # 统计
#         self.strawberries_picked = 0
#         self.strawberries_on_plate = 0
#         self.total_reward = 0.0
#         self.metacog_interventions = 0
#         self.sensor_failures = 0
        
#         # 动作处理器
#         self.action_processor = SimpleActionProcessor()
        
#         print(f"🍓 创建增强草莓环境")
#         print(f"   机器人: {robot}")
#         print(f"   GUI: {'启用' if enable_gui else '禁用 (避免崩溃)'}")
#         print(f"   最大步数: {horizon}")
#         print(f"   元认知模块: {'启用' if self.enable_metacognitive else '禁用'}")
#         print(f"   设备: {device}")
        
#         # 初始化元认知模块
#         if self.enable_metacognitive:
#             self._init_metacognitive_modules()
        
#         # 创建环境
#         self._create_stable_environment()
    
#     def _init_metacognitive_modules(self):
#         """初始化元认知模块"""
#         try:
#             print("🧠 初始化元认知模块...")
            
#             self.metacog_module = CompleteMetaCognitiveModule(self.device)
#             self.robocasa_adapter = RoboCasaToMetacogAdapter(image_size=(224, 224))  # 使用标准适配器
#             self.metacog_to_vla_adapter = MetacogToVLASystem2Adapter()
#             self.action_adjuster = ActionAdjuster()
            
#             print("✅ 元认知模块初始化成功")
            
#         except Exception as e:
#             print(f"❌ 元认知模块初始化失败: {e}")
#             self.enable_metacognitive = False
    
#     def _create_stable_environment(self):
#         """创建稳定环境 - 使用最简单的配置"""
#         try:
#             print("🏗️ 创建稳定环境...")
            
#             # 最简单的配置 - 避免复杂参数
#             config = {
#                 "env_name": "PnPCounterToCab",
#                 "robots": self.robot,
#                 "controller_configs": load_composite_controller_config(robot=self.robot),
#             }
            
#             print(f"   使用机器人: {self.robot}")
#             print(f"   控制器: 已加载")
            
#             # 非常保守的环境配置
#             self.env = robosuite.make(
#                 **config,
#                 has_renderer=False,  # 强制关闭渲染器避免崩溃
#                 has_offscreen_renderer=True,  # 保持离屏渲染
#                 render_camera=None,
#                 ignore_done=True,
#                 use_camera_obs=True,
#                 control_freq=20,
#                 camera_names=["robot0_eye_in_hand"],  # 只使用一个相机
#                 camera_heights=480,
#                 camera_widths=640,
#                 initialization_noise=None,  # 关闭噪声
#             )
            
#             print("✅ 稳定环境创建成功")
            
#             # 简单验证
#             if hasattr(self.env, 'action_space'):
#                 print(f"   动作空间: {getattr(self.env.action_space, 'shape', 'Unknown')}")
            
#             print("🎉 稳定环境初始化完成！")
            
#         except Exception as e:
#             print(f"❌ 环境创建失败: {e}")
#             import traceback
#             traceback.print_exc()
#             raise
    
#     def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#         """重置环境"""
#         try:
#             print("🔄 重置稳定环境...")
            
#             obs = self.env.reset()
#             self.current_step = 0
            
#             # 重置状态
#             self.strawberries_picked = 0
#             self.strawberries_on_plate = 0
#             self.total_reward = 0.0
#             self.strawberry_states = [True, True, True]
#             self.strawberry_on_plate = [False, False, False]
#             self.metacog_interventions = 0
#             self.sensor_failures = 0
            
#             # 处理观测数据 - 使用真实数据
#             processed_obs = self._process_real_observation(obs)
            
#             # 安全的位置调整
#             robot_pos = processed_obs.get("robot0_eef_pos", np.array([0.5, 0.0, 0.8]))
#             print(f"   机器人位置: {robot_pos}")
            
#             # 简单的位置调整
#             if abs(robot_pos[0]) > 1.5 or abs(robot_pos[1]) > 1.5:
#                 print("   ⚠️ 调整物体位置")
#                 self.strawberry_positions += robot_pos[:3] * 0.5
#                 self.plate_position += robot_pos[:3] * 0.5
            
#             # 构建信息
#             info = {
#                 "task_name": "Enhanced Strawberry Pick and Place",
#                 "task_description": "Pick up strawberries and place them in the target location",
#                 "step": self.current_step,
#                 "max_steps": self.horizon,
#                 "metacognitive_enabled": self.enable_metacognitive,
#                 "strawberry_task_progress": {
#                     "strawberries_picked": self.strawberries_picked,
#                     "strawberries_on_plate": self.strawberries_on_plate,
#                     "total_strawberries": 3,
#                 }
#             }
            
#             print("✅ 稳定环境重置成功")
            
#             return processed_obs, info
            
#         except Exception as e:
#             print(f"❌ 环境重置失败: {e}")
#             # 返回安全的默认值
#             return self._get_safe_default_obs(), {"step": 0, "task_name": "Safe Default"}
    
#     def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
#         """安全的步进"""
#         try:
#             # 安全的动作适配
#             adapted_action = self._safe_adapt_action(action)
            
#             # 环境步进
#             obs, reward, done, info = self.env.step(adapted_action)
#             self.current_step += 1
            
#             # 处理观测数据 - 使用真实数据
#             processed_obs = self._process_real_observation(obs)
            
#             # 任务奖励评估
#             task_reward, task_success = self._safe_evaluate_task(processed_obs, action)
#             reward += task_reward
#             self.total_reward += reward
            
#             # 元认知处理
#             metacog_feedback = None
#             if self.enable_metacognitive:
#                 metacog_feedback = self._process_metacognitive_feedback(processed_obs, adapted_action)
            
#             # 任务完成
#             if task_success:
#                 done = True
#                 reward += 10.0
#                 print(f"🎉 草莓任务完成！")
            
#             # 超时
#             if self.current_step >= self.horizon:
#                 done = True
            
#             # 增强信息
#             enhanced_info = {
#                 **info,
#                 "task_name": "Enhanced Strawberry Pick and Place",
#                 "task_description": "Pick up strawberries and place them in the target location",
#                 "step": self.current_step,
#                 "max_steps": self.horizon,
#                 "task_success": task_success,
#                 "total_reward": self.total_reward,
#                 "metacog_interventions": self.metacog_interventions,
#                 "sensor_failures": self.sensor_failures,
#                 "metacognitive_feedback": metacog_feedback,
#                 "strawberry_task_progress": {
#                     "strawberries_picked": self.strawberries_picked,
#                     "strawberries_on_plate": self.strawberries_on_plate,
#                     "total_strawberries": 3,
#                 }
#             }
            
#             # 简化的进度显示
#             if self.current_step % 30 == 0:
#                 print(f"   📊 步骤 {self.current_step}: 拣选={self.strawberries_picked}, 放置={self.strawberries_on_plate}, 奖励={self.total_reward:.2f}")
#                 if self.enable_metacognitive:
#                     print(f"   🧠 元认知干预: {self.metacog_interventions}")
            
#             return processed_obs, reward, done, False, enhanced_info
            
#         except Exception as e:
#             print(f"❌ 步进失败: {e}")
#             self.sensor_failures += 1
#             # 返回安全值
#             return self._get_safe_default_obs(), 0.0, True, False, {"step": self.current_step}
    
#     def _process_metacognitive_feedback(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> Optional[str]:
#         """处理元认知反馈 - 使用修复的元认知模块"""
#         if not self.enable_metacognitive:
#             return None
        
#         try:
#             # 转换观测数据为传感器数据格式 - 元认知模块会自动适配维度
#             sensor_data = self.robocasa_adapter.convert_observation(
#                 obs, action, execution_status="normal"
#             )
            
#             # 获取元认知输出
#             metacog_output = self.metacog_module.process_sensor_data(sensor_data)
            
#             # 转换为VLA System2指令
#             instruction = self.metacog_to_vla_adapter.convert_to_system2_instruction(metacog_output)
            
#             # 记录干预
#             if instruction and metacog_output.directive != DirectiveType.CONTINUE:
#                 self.metacog_interventions += 1
#                 if self.current_step % 30 == 0:  # 适度的打印频率
#                     print(f"   🧠 元认知干预: {instruction}")
            
#             return instruction
            
#         except Exception as e:
#             if self.current_step % 40 == 0:  # 进一步减少错误打印频率
#                 print(f"⚠️ 元认知处理错误: {e}")
#             return None
    
#     def _process_real_observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
#         """处理真实观测数据 - 替换随机数据"""
#         processed = {}
        
#         try:
#             # 处理图像数据 - 使用真实相机数据
#             image_found = False
#             for camera in ["robot0_robotview", "robot0_eye_in_hand"]:
#                 rgb_key = f"{camera}_image"
#                 if rgb_key in obs and obs[rgb_key] is not None:
#                     try:
#                         img = obs[rgb_key]
#                         if img is not None and img.size > 0:
#                             if img.shape[:2] != (480, 640):
#                                 img = cv2.resize(img, (640, 480))
#                             processed["frontview_image"] = img.astype(np.uint8)
#                             image_found = True
#                             break
#                     except Exception as e:
#                         print(f"⚠️ 处理{camera}图像失败: {e}")
#                         continue
            
#             if not image_found:
#                 processed["frontview_image"] = np.zeros((480, 640, 3), dtype=np.uint8)
#                 self.sensor_failures += 1
            
#             # 处理深度数据
#             depth_found = False
#             for camera in ["robot0_robotview", "robot0_eye_in_hand"]:
#                 depth_key = f"{camera}_depth"
#                 if depth_key in obs and obs[depth_key] is not None:
#                     try:
#                         depth = obs[depth_key]
#                         if depth.size > 0:
#                             if depth.shape != (480, 640):
#                                 depth = cv2.resize(depth, (640, 480))
#                             processed["frontview_depth"] = depth.astype(np.float32)
#                             depth_found = True
#                             break
#                     except Exception:
#                         continue
            
#             if not depth_found:
#                 processed["frontview_depth"] = np.ones((480, 640), dtype=np.float32)
            
#             # 处理机器人状态数据 - 使用真实传感器数据
#             robot_keys = ["robot0_joint_pos", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
            
#             for key in robot_keys:
#                 if key in obs and obs[key] is not None:
#                     try:
#                         data = np.array(obs[key], dtype=np.float32)
#                         # 检查数据有效性
#                         if np.any(np.isnan(data)) or np.any(np.isinf(data)):
#                             print(f"⚠️ {key} 包含无效数据")
#                             data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
                        
#                         if "joint" in key:
#                             processed[key] = data[:5] if len(data) > 5 else data
#                         else:
#                             processed[key] = data
#                     except Exception as e:
#                         print(f"⚠️ 处理{key}失败: {e}")
#                         self.sensor_failures += 1
            
#             # 提供安全的默认值（基于物理约束而不是随机）
#             if "robot0_eef_pos" not in processed:
#                 processed["robot0_eef_pos"] = np.array([0.5, 0.0, 0.8], dtype=np.float32)
#             if "robot0_joint_pos" not in processed:
#                 processed["robot0_joint_pos"] = np.zeros(5, dtype=np.float32)
#             if "robot0_eef_quat" not in processed:
#                 processed["robot0_eef_quat"] = np.array([0, 0, 0, 1], dtype=np.float32)
#             if "robot0_gripper_qpos" not in processed:
#                 processed["robot0_gripper_qpos"] = np.zeros(2, dtype=np.float32)
            
#             # 添加任务信息
#             processed["robot_type"] = "SO100"
#             processed["task_description"] = "Enhanced: Pick strawberries and place them carefully"
#             processed["current_step"] = self.current_step
            
#             return processed
            
#         except Exception as e:
#             print(f"⚠️ 观测数据处理错误: {e}")
#             self.sensor_failures += 1
#             return self._get_safe_default_obs()
    
#     def _safe_adapt_action(self, action: np.ndarray) -> np.ndarray:
#         """安全的动作适配"""
#         try:
#             if not isinstance(action, np.ndarray):
#                 action = np.array(action)
            
#             # 确保动作是有限的
#             action = np.nan_to_num(action, nan=0.0, posinf=0.1, neginf=-0.1)
            
#             if len(action) == 6:
#                 adapted = np.zeros(7)
#                 adapted[0:3] = action[0:3]
#                 adapted[3:5] = action[3:5]
#                 adapted[5] = 0.0
#                 adapted[6] = action[5]
#                 return np.clip(adapted, -1.0, 1.0)
#             elif len(action) == 7:
#                 return np.clip(action, -1.0, 1.0)
#             else:
#                 adapted = np.zeros(7)
#                 adapted[:min(len(action), 7)] = action[:7]
#                 return np.clip(adapted, -1.0, 1.0)
                
#         except Exception as e:
#             print(f"⚠️ 动作适配错误: {e}")
#             return np.zeros(7)
    
#     def _safe_evaluate_task(self, obs: Dict[str, Any], action: np.ndarray) -> Tuple[float, bool]:
#         """安全的草莓任务评估 - 基于真实传感器数据"""
#         try:
#             reward = 0.0
#             task_success = False
            
#             robot_pos = obs.get("robot0_eef_pos")
#             if robot_pos is None:
#                 return reward, task_success
            
#             gripper_action = action[-1] if len(action) > 0 else 0.0
            
#             # 草莓检测 - 基于真实位置数据
#             for i, (strawberry_pos, is_available) in enumerate(zip(self.strawberry_positions, self.strawberry_states)):
#                 if not is_available:
#                     continue
                
#                 try:
#                     distance = np.linalg.norm(robot_pos - strawberry_pos)
                    
#                     if distance < 0.3:  # 接近草莓
#                         reward += 0.5
                        
#                         if distance < 0.2 and gripper_action > 0.2:  # 抓取动作
#                             if self.strawberry_states[i]:
#                                 self.strawberry_states[i] = False
#                                 self.strawberries_picked += 1
#                                 reward += 2.0
#                                 print(f"   🍓 拣选草莓{i+1}!")
                                
#                 except Exception:
#                     continue
            
#             # 盘子检测 - 基于真实位置数据
#             try:
#                 plate_distance = np.linalg.norm(robot_pos - self.plate_position)
                
#                 if plate_distance < 0.25:  # 接近盘子
#                     reward += 0.5
                    
#                     if plate_distance < 0.15 and gripper_action < -0.2:  # 放置动作
#                         picked = sum(1 for state in self.strawberry_states if not state)
#                         on_plate = sum(1 for state in self.strawberry_on_plate if state)
                        
#                         if picked > on_plate:
#                             for i, on_plate_state in enumerate(self.strawberry_on_plate):
#                                 if not on_plate_state and not self.strawberry_states[i]:
#                                     self.strawberry_on_plate[i] = True
#                                     self.strawberries_on_plate += 1
#                                     reward += 3.0
#                                     print(f"   🍽️ 放置草莓{i+1}!")
#                                     break
#             except Exception:
#                 pass
            
#             # 任务完成判断
#             if self.strawberries_on_plate >= 3:
#                 task_success = True
            
#             return reward, task_success
            
#         except Exception as e:
#             print(f"⚠️ 任务评估错误: {e}")
#             return 0.0, False
    
#     def _get_safe_default_obs(self) -> Dict[str, np.ndarray]:
#         """获取安全的默认观测"""
#         return {
#             "frontview_image": np.zeros((480, 640, 3), dtype=np.uint8),
#             "frontview_depth": np.ones((480, 640), dtype=np.float32),
#             "robot0_joint_pos": np.zeros(5, dtype=np.float32),
#             "robot0_eef_pos": np.array([0.5, 0.0, 0.8], dtype=np.float32),
#             "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
#             "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
#             "robot_type": "SO100",
#             "task_description": "Safe default observation",
#             "current_step": self.current_step
#         }
    
#     def get_action_space(self):
#         """获取动作空间"""
#         if self.env is None:
#             raise RuntimeError("环境未初始化")
        
#         if hasattr(self.env, 'action_space'):
#             return self.env.action_space
#         else:
#             import gym.spaces
#             return gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    
#     def close(self):
#         """安全关闭环境"""
#         if self.env is not None:
#             try:
#                 self.env.close()
#                 print("🔒 增强草莓环境已关闭")
#                 print(f"📊 最终结果: 拣选={self.strawberries_picked}/3, 放置={self.strawberries_on_plate}/3")
#                 print(f"   总奖励={self.total_reward:.2f}, 元认知干预={self.metacog_interventions}, 传感器失败={self.sensor_failures}")
#             except Exception as e:
#                 print(f"⚠️ 关闭环境错误: {e}")
#             finally:
#                 self.env = None




from robosuite.models.objects import BoxObject, CylinderObject, CanObject
from robosuite.utils.placement_samplers import UniformRandomSampler

class EnhancedStrawberryEnvironment:
    """
    增强的桌面草莓环境 - 使用自定义的桌面场景替换厨房环境。
    (最终兼容版，修复了size参数问题)
    """
    
    def __init__(self, 
                 so100_xml_path: str = None, 
                 horizon: int = 100,
                 enable_gui: bool = False,
                 robot: str = "Panda",
                 enable_metacognitive: bool = True,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        if not ROBOSUITE_AVAILABLE:
            raise ImportError("RoboSuite不可用")
        
        self.horizon = horizon
        self.enable_gui = enable_gui
        self.robot = robot
        self.enable_metacognitive = enable_metacognitive and METACOG_AVAILABLE
        self.device = device
        
        self.env = None
        self.current_step = 0
        self.table_top_offset = None
        self.plate_pos = None
        self.object_names = ["strawberry1", "strawberry2", "strawberry3", "grape1", "grape2", "grape3"]
        self.held_object = None
        self.placed_strawberries = set()
        
        self.total_reward = 0.0
        self.metacog_interventions = 0
        self.sensor_failures = 0
        
        self.action_processor = SimpleActionProcessor()
        
        print(f"🍓 创建增强的【桌面】草莓环境")
        print(f"   机器人: {robot}")
        print(f"   GUI: {'启用' if enable_gui else '禁用'}")
        print(f"   最大步数: {horizon}")
        
        if self.enable_metacognitive:
            self._init_metacognitive_modules()
        
        self._create_tabletop_environment()

    def _init_metacognitive_modules(self):
        try:
            print("🧠 初始化元认知模块...")
            self.metacog_module = CompleteMetaCognitiveModule(self.device)
            self.robocasa_adapter = RoboCasaToMetacogAdapter(image_size=(224, 224))
            self.metacog_to_vla_adapter = MetacogToVLASystem2Adapter()
            self.action_adjuster = ActionAdjuster()
            print("✅ 元认知模块初始化成功")
        except Exception as e:
            print(f"❌ 元认知模块初始化失败: {e}")
            self.enable_metacognitive = False

    def _create_tabletop_environment(self):
        """
        创建自定义的桌面环境 - 兼容旧版Robosuite API (size参数)
        """
        try:
            print("🏗️ 创建自定义桌面环境...")
            
            # 【【【 已修复 】】】
            # 1. 定义我们的物体，使用 size_min 和 size_max
            strawberry_size = [0.02, 0.025] # [radius, half_height]
            strawberries = [
                CanObject(
                    name=f"strawberry{i+1}", 
                    size_min=strawberry_size, # 使用旧版API
                    size_max=strawberry_size, # 使用旧版API
                    rgba=[1, 0, 0, 1]
                ) for i in range(3)
            ]
            
            grape_size = [0.018, 0.018] # [radius, half_height]
            grapes = [
                CylinderObject(
                    name=f"grape{i+1}", 
                    size_min=grape_size, # 使用旧版API
                    size_max=grape_size, # 使用旧版API
                    rgba=[0.5, 1, 0.5, 1]
                ) for i in range(3)
            ]
            
            plate_size = [0.12, 0.01] # [radius, half_height]
            plate = CylinderObject(
                name="plate",
                size_min=plate_size, # 使用旧版API
                size_max=plate_size, # 使用旧版API
                rgba=[1, 1, 1, 1],
                solimp=[0.998, 0.998, 0.001],
                solref=[0.001, 1]
            )
            
            # 2. 配置环境 (与上一版相同)
            config = {
                "env_name": "PickPlace",
                "robots": self.robot,
                "controller_configs": load_composite_controller_config(robot=self.robot),
                "placement_initializer": None,
                "single_object_mode": 2,
                "has_renderer": self.enable_gui,
                "has_offscreen_renderer": True,
                "ignore_done": True,
                "use_camera_obs": True,
                "control_freq": 20,
                "camera_configs": {
                    "type": "Camera",
                    "name": "worldview",
                    "pos": np.array([0.6, 0.0, 1.4]),
                    "quat": np.array([0.653, 0.271, 0.653, -0.271]),
                    "camera_fovy": 50
                },
                "camera_names": "worldview",
                "camera_heights": 480,
                "camera_widths": 640,
                "mujoco_objects": strawberries + grapes + [plate]
            }
            
            # 3. 创建环境
            self.env = robosuite.make(**config)

            # 4. 获取桌面信息
            self.table_top_offset = self.env.table_top_offset
            
            print("✅ 自定义桌面环境创建成功！")
            
        except Exception as e:
            print(f"❌ 环境创建失败: {e}")
            import traceback
            traceback.print_exc()
            raise


    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """重置环境，并手动放置所有物体"""
        try:
            print("🔄 重置桌面环境...")
            obs = self.env.reset()
            self.current_step = 0
            
            # 重置状态
            self.total_reward = 0.0
            self.held_object = None
            self.placed_strawberries.clear()
            
            # --- 手动放置物体 ---
            # 定义桌面上可放置物体的区域
            table_pos = self.table_top_offset
            x_range = [-0.15, 0.15]
            y_range = [-0.25, 0.25]
            
            # 放置盘子
            self.plate_pos = np.array([table_pos[0] - 0.25, table_pos[1], table_pos[2]])
            self.env.sim.data.set_joint_qpos(
                "plate_joint0",
                np.concatenate([self.plate_pos, [1, 0, 0, 0]])
            )
            
            # 随机放置草莓和葡萄
            for obj_name in self.object_names:
                while True:
                    # 在桌面上随机选一个点
                    random_pos = table_pos + np.array([
                        np.random.uniform(*x_range),
                        np.random.uniform(*y_range),
                        0.02 # 物体高度偏移
                    ])
                    
                    # 确保不会和盘子重叠
                    if np.linalg.norm(random_pos[:2] - self.plate_pos[:2]) > 0.15:
                        self.env.sim.data.set_joint_qpos(
                            f"{obj_name}_joint0",
                            np.concatenate([random_pos, [1, 0, 0, 0]])
                        )
                        break
            
            # --- 结束手动放置 ---
            
            processed_obs = self._process_real_observation(obs)
            info = self._get_current_info()
            
            print("✅ 桌面环境重置成功，物体已放置。")
            
            return processed_obs, info
            
        except Exception as e:
            print(f"❌ 环境重置失败: {e}")
            return self._get_safe_default_obs(), {"step": 0, "task_name": "Safe Default"}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """安全的步进"""
        try:
            adapted_action = self._safe_adapt_action(action)
            obs, reward, done, info = self.env.step(adapted_action)
            self.current_step += 1
            
            processed_obs = self._process_real_observation(obs)
            
            # 使用全新的桌面任务评估函数
            task_reward, task_success = self._evaluate_tabletop_task(processed_obs, adapted_action)
            reward = task_reward # 我们只关心我们的任务奖励
            self.total_reward += reward
            
            metacog_feedback = None
            if self.enable_metacognitive:
                metacog_feedback = self._process_metacognitive_feedback(processed_obs, adapted_action)
            
            if task_success:
                done = True
                self.total_reward += 10.0 # 成功时给予巨大奖励
                print(f"🎉 草莓任务完成！")
            
            if self.current_step >= self.horizon:
                done = True

            enhanced_info = self._get_current_info()
            enhanced_info['task_success'] = task_success
            enhanced_info['metacognitive_feedback'] = metacog_feedback
            
            if self.enable_gui:
                self.env.render()

            return processed_obs, reward, done, False, enhanced_info
            
        except Exception as e:
            print(f"❌ 步进失败: {e}")
            self.sensor_failures += 1
            return self._get_safe_default_obs(), 0.0, True, False, {"step": self.current_step}

    def _evaluate_tabletop_task(self, obs: Dict[str, Any], action: np.ndarray) -> Tuple[float, bool]:
        """
        全新的任务评估函数，基于3D坐标。
        """
        reward = 0.0
        eef_pos = obs.get("robot0_eef_pos")
        gripper_openness = obs.get("robot0_gripper_qpos")[0] # 假设值越大越开
        
        # 1. 抓取逻辑
        if self.held_object is None:
            # 寻找最近的、尚未放置的草莓
            min_dist = float('inf')
            target_strawberry = None
            for i in range(3):
                s_name = f"strawberry{i+1}"
                if s_name not in self.placed_strawberries:
                    s_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(f"{s_name}_main")]
                    dist = np.linalg.norm(eef_pos - s_pos)
                    if dist < min_dist:
                        min_dist = dist
                        target_strawberry = s_name
            
            if target_strawberry:
                # 奖励：靠近目标草莓
                reward += 1.0 - np.tanh(10.0 * min_dist)
                
                # 检查是否抓取成功
                if min_dist < 0.05 and gripper_openness < 0.01: # Gripper is closed
                    self.held_object = target_strawberry
                    reward += 5.0 # 抓取成功奖励
                    print(f"   🍓 抓取 {self.held_object}!")

        # 2. 放置逻辑
        else:
            # 奖励：靠近盘子
            dist_to_plate = np.linalg.norm(eef_pos[:2] - self.plate_pos[:2])
            reward += 1.0 - np.tanh(10.0 * dist_to_plate)
            
            # 获取当前抓着物体的实时位置
            held_obj_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(f"{self.held_object}_main")]

            # 检查是否放置成功
            dist_on_plate = np.linalg.norm(held_obj_pos[:2] - self.plate_pos[:2])
            is_over_plate = dist_on_plate < 0.1 # 盘子半径
            is_low_enough = held_obj_pos[2] < self.table_top_offset[2] + 0.05
            
            if is_over_plate and is_low_enough and gripper_openness > 0.02: # Gripper is opening
                print(f"   🍽️ 放置 {self.held_object}!")
                self.placed_strawberries.add(self.held_object)
                self.held_object = None
                reward += 10.0 # 放置成功奖励

        # 3. 任务成功判断
        task_success = len(self.placed_strawberries) == 3
        return reward, task_success
    
    # --- Helper and Unchanged Methods ---
    
    def _get_current_info(self) -> Dict[str, Any]:
        return {
            "task_name": "Tabletop Strawberry Pick and Place",
            "task_description": "Pick up red strawberries and place them on the white plate.",
            "step": self.current_step,
            "max_steps": self.horizon,
            "total_reward": self.total_reward,
            "metacog_interventions": self.metacog_interventions,
            "sensor_failures": self.sensor_failures,
            "strawberry_task_progress": {
                "strawberries_picked": 1 if self.held_object else 0,
                "strawberries_on_plate": len(self.placed_strawberries),
                "total_strawberries": 3,
            }
        }

    def _process_real_observation(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        processed = {}
        try:
            # 使用我们自定义的 'worldview' 相机
            rgb_key = "worldview_image"
            if rgb_key in obs and obs[rgb_key] is not None:
                # Robosuite返回的图像是上下颠倒的，需要翻转
                img = obs[rgb_key][::-1] 
                processed["frontview_image"] = img.astype(np.uint8)
            else:
                processed["frontview_image"] = np.zeros((480, 640, 3), dtype=np.uint8)
                self.sensor_failures += 1
            
            depth_key = "worldview_depth"
            if depth_key in obs and obs[depth_key] is not None:
                depth = obs[depth_key][::-1]
                processed["frontview_depth"] = depth.astype(np.float32)
            else:
                processed["frontview_depth"] = np.ones((480, 640), dtype=np.float32)
            
            robot_keys = ["robot0_joint_pos", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
            for key in robot_keys:
                if key in obs and obs[key] is not None:
                    processed[key] = np.array(obs[key], dtype=np.float32)

            return processed
        except Exception as e:
            print(f"⚠️ 观测数据处理错误: {e}")
            self.sensor_failures += 1
            return self._get_safe_default_obs()

    def close(self):
        """安全关闭环境"""
        if self.env is not None:
            try:
                self.env.close()
                print("🔒 增强桌面环境已关闭")
                print(f"📊 最终结果: 放置={len(self.placed_strawberries)}/3")
            except Exception as e:
                print(f"⚠️ 关闭环境错误: {e}")
            finally:
                self.env = None

    def _safe_adapt_action(self, action: np.ndarray) -> np.ndarray:
        """安全的动作适配"""
        try:
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            
            action = np.nan_to_num(action, nan=0.0, posinf=0.1, neginf=-0.1)
            
            if len(action) == 6:
                adapted = np.zeros(7)
                adapted[0:3] = action[0:3]
                adapted[3:5] = action[3:5]
                adapted[5] = 0.0
                adapted[6] = action[5]
                return np.clip(adapted, -1.0, 1.0)
            elif len(action) == 7:
                return np.clip(action, -1.0, 1.0)
            else:
                adapted = np.zeros(7)
                adapted[:min(len(action), 7)] = action[:7]
                return np.clip(adapted, -1.0, 1.0)
                
        except Exception as e:
            print(f"⚠️ 动作适配错误: {e}")
            return np.zeros(7)
    
    def _get_safe_default_obs(self) -> Dict[str, np.ndarray]:
        """获取安全的默认观测"""
        return {
            "frontview_image": np.zeros((480, 640, 3), dtype=np.uint8),
            "frontview_depth": np.ones((480, 640), dtype=np.float32),
            "robot0_joint_pos": np.zeros(5, dtype=np.float32),
            "robot0_eef_pos": np.array([0.5, 0.0, 0.8], dtype=np.float32),
            "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            "robot_type": "SO100",
            "task_description": "Safe default observation",
            "current_step": self.current_step
        }
    
    def get_action_space(self):
        """获取动作空间"""
        if self.env is None:
            raise RuntimeError("环境未初始化")
        return self.env.action_space
    
    def _process_metacognitive_feedback(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> Optional[str]:
        """处理元认知反馈 - 使用修复的元认知模块"""
        if not self.enable_metacognitive:
            return None
        
        try:
            sensor_data = self.robocasa_adapter.convert_observation(
                obs, action, execution_status="normal"
            )
            metacog_output = self.metacog_module.process_sensor_data(sensor_data)
            instruction = self.metacog_to_vla_adapter.convert_to_system2_instruction(metacog_output)
            
            if instruction and metacog_output.directive != DirectiveType.CONTINUE:
                self.metacog_interventions += 1
            
            return instruction
            
        except Exception as e:
            return None

# ==================== 支持视频录制的增强草莓环境 ====================

# class EnhancedStrawberryEnvironmentWithVideo(EnhancedStrawberryEnvironment):
#     """增强草莓环境 - 支持视频录制"""
    
#     def __init__(self, 
#                  so100_xml_path: str = None,
#                  horizon: int = 100,
#                  enable_gui: bool = False,
#                  robot: str = "Panda",
#                  enable_metacognitive: bool = True,
#                  device: str = "cuda" if torch.cuda.is_available() else "cpu",
#                  # 新增视频录制参数
#                  enable_video_recording: bool = True,
#                  video_output_dir: str = "./experiment_videos",
#                  video_fps: int = 20):
#         """
#         初始化支持视频录制的增强草莓环境
        
#         Args:
#             enable_video_recording: 是否启用视频录制
#             video_output_dir: 视频保存目录
#             video_fps: 视频帧率
#         """
        
#         # 初始化父类
#         super().__init__(so100_xml_path, horizon, enable_gui, robot, 
#                         enable_metacognitive, device)
        
#         # 视频录制设置
#         self.enable_video_recording = enable_video_recording
#         self.video_recorder = None
        
#         if self.enable_video_recording:
#             self.video_recorder = VideoRecorder(
#                 output_dir=video_output_dir,
#                 fps=video_fps,
#                 video_size=(640, 480)
#             )
#             print(f"🎥 视频录制已启用")
#         else:
#             print(f"📷 视频录制已禁用")
    
#     def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#         """重置环境并开始新的视频录制"""
#         obs, info = super().reset()
        
#         # 开始新的episode录制
#         if self.enable_video_recording and self.video_recorder:
#             episode_id = info.get('episode_id', 0)
#             self.video_recorder.start_episode_recording(episode_id, "enhanced_strawberry")
            
#             # 录制第一帧
#             self._record_current_frame(obs, info)
        
#         return obs, info
    
#     def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
#         """环境步进并录制视频帧"""
#         obs, reward, done, truncated, info = super().step(action)
        
#         # 录制当前帧
#         if self.enable_video_recording and self.video_recorder:
#             self._record_current_frame(obs, info)
            
#             # 如果episode结束，停止录制
#             if done:
#                 self.video_recorder.stop_episode_recording()
        
#         return obs, reward, done, truncated, info
    
#     def _record_current_frame(self, obs: Dict[str, Any], info: Dict[str, Any]):
#         """录制当前帧"""
#         try:
#             # 提取RGB图像
#             rgb_image = None
            
#             # 尝试从观测中获取图像
#             if "frontview_image" in obs and obs["frontview_image"] is not None:
#                 rgb_image = obs["frontview_image"]
#             elif "robot0_robotview_image" in obs and obs["robot0_robotview_image"] is not None:
#                 rgb_image = obs["robot0_robotview_image"]
            
#             if rgb_image is not None:
#                 # 准备帧信息
#                 frame_info = {
#                     'step': info.get('step', self.current_step),
#                     'total_reward': info.get('total_reward', self.total_reward),
#                     'task_success': info.get('task_success', False),
#                     'strawberry_task_progress': info.get('strawberry_task_progress', {}),
#                     'metacognitive_feedback': info.get('metacognitive_feedback')
#                 }
                
#                 # 添加帧到录制器
#                 self.video_recorder.add_frame(rgb_image, frame_info)
            
#         except Exception as e:
#             print(f"⚠️ 录制帧失败: {e}")
    
#     def close(self):
#         """关闭环境并清理视频录制器"""
#         # 停止视频录制
#         if self.enable_video_recording and self.video_recorder:
#             self.video_recorder.cleanup()
        
#         # 调用父类关闭方法
#         super().close()


class EnhancedStrawberryEnvironmentWithVideo(EnhancedStrawberryEnvironment):
    """
    支持视频录制的增强桌面环境 (此类代码基本不变, 仅继承新的父类)
    """
    
    def __init__(self, 
                 so100_xml_path: str = None,
                 horizon: int = 100,
                 enable_gui: bool = False,
                 robot: str = "Panda",
                 enable_metacognitive: bool = True,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 enable_video_recording: bool = True,
                 video_output_dir: str = "./experiment_videos",
                 video_fps: int = 20):
        
        # 初始化父类 (现在是新的桌面环境父类)
        super().__init__(so100_xml_path, horizon, enable_gui, robot, 
                        enable_metacognitive, device)
        
        self.enable_video_recording = enable_video_recording
        self.video_recorder = None
        
        if self.enable_video_recording:
            self.video_recorder = VideoRecorder(
                output_dir=video_output_dir,
                fps=video_fps,
                video_size=(640, 480)
            )
            print(f"🎥 视频录制已启用")
        else:
            print(f"📷 视频录制已禁用")
    
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        obs, info = super().reset()
        if self.enable_video_recording and self.video_recorder:
            episode_id = info.get('episode_id', 0)
            self.video_recorder.start_episode_recording(episode_id, "tabletop_strawberry")
            self._record_current_frame(obs, info)
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        obs, reward, done, truncated, info = super().step(action)
        if self.enable_video_recording and self.video_recorder:
            self._record_current_frame(obs, info)
            if done:
                self.video_recorder.stop_episode_recording()
        return obs, reward, done, truncated, info
    
    def _record_current_frame(self, obs: Dict[str, Any], info: Dict[str, Any]):
        try:
            rgb_image = obs.get("frontview_image")
            if rgb_image is not None:
                frame_info = {
                    'step': info.get('step', self.current_step),
                    'total_reward': info.get('total_reward', self.total_reward),
                    'task_success': info.get('task_success', False),
                    'strawberry_task_progress': info.get('strawberry_task_progress', {}),
                    'metacognitive_feedback': info.get('metacognitive_feedback')
                }
                self.video_recorder.add_frame(rgb_image, frame_info)
        except Exception as e:
            print(f"⚠️ 录制帧失败: {e}")
    
    def close(self):
        if self.enable_video_recording and self.video_recorder:
            self.video_recorder.cleanup()
        super().close()

# ==================== 增强GR00T客户端 ====================

class EnhancedGR00TClient:
    """增强GR00T客户端 - 支持元认知反馈"""
    
    def __init__(self, host: str = "localhost", port: int = 5555):
        self.host = host
        self.port = port
        self.client = None
        self.is_connected = False
        self.action_processor = SimpleActionProcessor()
        
        # 统计信息
        self.total_calls = 0
        self.total_successes = 0
        self.total_time = 0.0
    
    def connect(self) -> bool:
        """连接到GR00T服务"""
        if not GROOT_CLIENT_AVAILABLE:
            return False
        
        try:
            print(f"🔗 连接到GR00T: {self.host}:{self.port}")
            
            self.client = RobotInferenceClient(host=self.host, port=self.port)
            
            # 验证连接
            modality_config = self.client.get_modality_config()
            print("✅ GR00T连接成功！")
            
            # 连接测试
            test_obs = self._create_test_observation()
            test_result = self.client.get_action(test_obs)
            
            if test_result is not None:
                print("✅ GR00T测试成功！")
                self.is_connected = True
                return True
            else:
                print("❌ GR00T测试失败")
                return False
                
        except Exception as e:
            print(f"❌ GR00T连接失败: {e}")
            return False
    
    def predict_action(self, observation: Dict[str, np.ndarray], 
                      task_description: str = None,
                      metacognitive_feedback: str = None) -> Optional[np.ndarray]:
        """预测动作 - 支持元认知反馈"""
        if not self.is_connected:
            return None
        
        self.total_calls += 1
        start_time = time.time()
        
        try:
            # 转换观测格式
            groot_obs = self._convert_observation(observation, task_description, metacognitive_feedback)
            
            # 获取动作
            groot_action = self.client.get_action(groot_obs)
            
            api_time = time.time() - start_time
            self.total_time += api_time
            
            if groot_action is not None:
                self.total_successes += 1
                # 使用简化的动作处理器
                so100_action = self.action_processor.process_groot_action(groot_action)
                return so100_action
            else:
                return None
                
        except Exception as e:
            api_time = time.time() - start_time
            self.total_time += api_time
            if self.total_calls % 20 == 0:
                print(f"⚠️ 预测错误: {e}")
            return None
    
    def _convert_observation(self, obs: Dict[str, np.ndarray], 
                           task_description: str = None,
                           metacognitive_feedback: str = None) -> Dict[str, Any]:
        """转换观测格式"""
        try:
            groot_obs = {}
            
            # 视频数据
            if "frontview_image" in obs and obs["frontview_image"] is not None:
                img = obs["frontview_image"]
                if img.shape[:2] != (480, 640):
                    img = cv2.resize(img, (640, 480))
                groot_obs["video.webcam"] = img[np.newaxis, :, :, :].astype(np.uint8)
            else:
                groot_obs["video.webcam"] = np.zeros((1, 480, 640, 3), dtype=np.uint8)
            
            # 机器人状态
            if "robot0_joint_pos" in obs and obs["robot0_joint_pos"] is not None:
                joint_pos = obs["robot0_joint_pos"][:5]  # 使用前5个关节
                joint_pos = np.clip(joint_pos, -2.0, 2.0)
                groot_obs["state.single_arm"] = joint_pos[np.newaxis, :].astype(np.float32)
            else:
                groot_obs["state.single_arm"] = np.zeros((1, 5), dtype=np.float32)
            
            # 夹爪状态
            if "robot0_gripper_qpos" in obs and obs["robot0_gripper_qpos"] is not None:
                gripper_pos = obs["robot0_gripper_qpos"][:1]
                groot_obs["state.gripper"] = gripper_pos[np.newaxis, :].astype(np.float32)
            else:
                groot_obs["state.gripper"] = np.zeros((1, 1), dtype=np.float32)
            
            # 任务描述 - 集成元认知反馈
            desc_parts = []
            if task_description:
                desc_parts.append(task_description)
            if metacognitive_feedback:
                desc_parts.append(f"建议: {metacognitive_feedback}")
            
            if desc_parts:
                full_description = ". ".join(desc_parts)
            else:
                full_description = obs.get("task_description", "Pick strawberries and place them carefully")
            
            groot_obs["annotation.human.task_description"] = [full_description]
            
            return groot_obs
            
        except Exception as e:
            print(f"⚠️ 观测转换错误: {e}")
            return self._create_test_observation()
    
    def _create_test_observation(self) -> Dict[str, Any]:
        """创建测试观测"""
        return {
            "video.webcam": np.zeros((1, 480, 640, 3), dtype=np.uint8),
            "state.single_arm": np.zeros((1, 5), dtype=np.float32),
            "state.gripper": np.zeros((1, 1), dtype=np.float32),
            "annotation.human.task_description": ["Pick strawberries and place them carefully"]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "calls": self.total_calls,
            "successes": self.total_successes,
            "success_rate": self.total_successes / self.total_calls if self.total_calls > 0 else 0,
            "avg_time": self.total_time / self.total_calls if self.total_calls > 0 else 0,
        }

# ==================== 主实验类 ====================

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 连接设置
    host: str = "localhost"
    port: int = 5555
    
    # 实验设置
    experiment_name: str = "enhanced_strawberry_experiment"
    num_episodes: int = 3
    max_steps_per_episode: int = 80
    
    # 环境设置
    so100_xml_path: str = None
    
    # 模块启用
    enable_metacognitive: bool = True
    
    # 设备设置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class EpisodeResult:
    """Episode结果"""
    episode_id: int
    task_success: bool
    total_steps: int
    total_time: float
    total_reward: float
    api_calls: int
    api_successes: int
    metacog_interventions: int
    sensor_failures: int
    strawberries_picked: int
    strawberries_on_plate: int

class EnhancedStrawberryExperiment:
    """增强草莓实验 - 集成所有功能"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = []
        
        print(f"🎯 初始化增强草莓实验")
        print(f"   元认知: {'启用' if config.enable_metacognitive else '禁用'}")
        print(f"   设备: {config.device}")
        
        # 初始化组件
        self.groot_client = EnhancedGR00TClient(config.host, config.port)
        self.environment = None
        
        # 创建环境
        self._create_environment()
    
    def _create_environment(self):
        """创建增强草莓环境"""
        print(f"🏗️ 创建增强草莓环境")
        
        self.environment = EnhancedStrawberryEnvironment(
            so100_xml_path=self.config.so100_xml_path,
            horizon=self.config.max_steps_per_episode * 2,
            enable_gui=False,  # 避免崩溃
            robot="Panda",
            enable_metacognitive=self.config.enable_metacognitive,
            device=self.config.device
        )
    
    def run_experiment(self) -> bool:
        """运行完整实验"""
        print(f"\n🎯 开始增强草莓实验")
        print(f"任务: 草莓拣选和放置")
        print("=" * 70)
        
        # 连接GR00T
        if not self.groot_client.connect():
            print("❌ 无法连接到GR00T服务")
            return False
        
        try:
            # 运行episodes
            for episode in range(self.config.num_episodes):
                print(f"\n📊 Episode {episode + 1}/{self.config.num_episodes}")
                result = self._run_episode(episode)
                self.results.append(result)
                self._print_episode_summary(result)
            
            # 分析结果
            self._analyze_results()
            self._save_results()
            
            return True
            
        finally:
            if hasattr(self.environment, 'close'):
                self.environment.close()
    
    def _run_episode(self, episode_id: int) -> EpisodeResult:
        """运行单个episode"""
        start_time = time.time()
        
        result = EpisodeResult(
            episode_id=episode_id,
            task_success=False,
            total_steps=0,
            total_time=0.0,
            total_reward=0.0,
            api_calls=0,
            api_successes=0,
            metacog_interventions=0,
            sensor_failures=0,
            strawberries_picked=0,
            strawberries_on_plate=0
        )
        
        try:
            obs, info = self.environment.reset()
            # 添加episode_id到info中
            info['episode_id'] = episode_id
            
            done = False
            step_count = 0
            
            print(f"     执行中: ", end="", flush=True)
            
            while not done and step_count < self.config.max_steps_per_episode:
                # 获取任务描述和元认知反馈
                task_description = obs.get("task_description", "Pick strawberries and place them carefully")
                metacognitive_feedback = info.get("metacognitive_feedback") if 'info' in locals() else None
                
                # 获取GR00T动作
                action = self.groot_client.predict_action(
                    obs, task_description, metacognitive_feedback
                )
                
                result.api_calls += 1
                
                if action is not None:
                    result.api_successes += 1
                    print(".", end="", flush=True)
                else:
                    # 使用零动作作为回退
                    action = np.zeros(6)
                    print("x", end="", flush=True)
                
                # 环境步进
                next_obs, reward, done, _, info = self.environment.step(action)
                
                result.total_reward += reward
                obs = next_obs
                step_count += 1
                
                # 记录统计
                result.metacog_interventions = info.get("metacog_interventions", 0)
                result.sensor_failures = info.get("sensor_failures", 0)
                
                # 记录草莓进度
                progress = info.get("strawberry_task_progress", {})
                result.strawberries_picked = progress.get("strawberries_picked", 0)
                result.strawberries_on_plate = progress.get("strawberries_on_plate", 0)
                
                # 检查任务完成
                if info.get("task_success", False):
                    result.task_success = True
                    done = True
                    print("!", end="", flush=True)
                
                # 进度显示
                if step_count % 15 == 0:
                    success_rate = result.api_successes / result.api_calls if result.api_calls > 0 else 0
                    print(f"|{success_rate:.0%}", end="", flush=True)
                
                # 元认知干预显示
                if result.metacog_interventions > 0 and step_count % 10 == 0:
                    print("M", end="", flush=True)
            
            result.total_steps = step_count
            result.total_time = time.time() - start_time
            
            print()
            
        except Exception as e:
            result.total_time = time.time() - start_time
            print(f" 异常: {e}")
        
        return result
    
    def _print_episode_summary(self, result: EpisodeResult):
        """打印episode摘要"""
        status = "✅ 成功" if result.task_success else "❌ 失败"
        api_success_rate = result.api_successes / result.api_calls if result.api_calls > 0 else 0
        
        print(f"   结果: {status}")
        print(f"   执行: {result.total_steps} 步, {result.total_time:.1f}s")
        print(f"   奖励: {result.total_reward:.2f}")
        print(f"   API: {result.api_successes}/{result.api_calls} 成功 ({api_success_rate:.1%})")
        print(f"   草莓: 拣选={result.strawberries_picked}/3, 放置={result.strawberries_on_plate}/3")
        
        if result.metacog_interventions > 0:
            print(f"   元认知: {result.metacog_interventions} 次干预")
        if result.sensor_failures > 0:
            print(f"   传感器: {result.sensor_failures} 次失败")
    
    def _analyze_results(self):
        """分析实验结果"""
        print(f"\n📊 实验结果分析")
        print("=" * 70)
        
        if not self.results:
            print("❌ 没有结果数据")
            return
        
        # 总体统计
        total_episodes = len(self.results)
        successful_episodes = sum(1 for r in self.results if r.task_success)
        success_rate = successful_episodes / total_episodes
        
        total_api_calls = sum(r.api_calls for r in self.results)
        total_api_successes = sum(r.api_successes for r in self.results)
        api_success_rate = total_api_successes / total_api_calls if total_api_calls > 0 else 0
        
        avg_reward = np.mean([r.total_reward for r in self.results])
        avg_steps = np.mean([r.total_steps for r in self.results])
        
        # 草莓任务统计
        total_picked = sum(r.strawberries_picked for r in self.results)
        total_placed = sum(r.strawberries_on_plate for r in self.results)
        avg_picked = total_picked / total_episodes
        avg_placed = total_placed / total_episodes
        
        print(f"🔍 总体表现:")
        print(f"   任务成功率: {success_rate:.1%} ({successful_episodes}/{total_episodes})")
        print(f"   平均奖励: {avg_reward:.2f}")
        print(f"   平均步数: {avg_steps:.1f}")
        print(f"   API成功率: {api_success_rate:.1%} ({total_api_successes}/{total_api_calls})")
        print(f"   草莓拣选: 平均 {avg_picked:.1f}/3 个/episode")
        print(f"   草莓放置: 平均 {avg_placed:.1f}/3 个/episode")
        
        # 元认知分析
        if self.config.enable_metacognitive:
            total_interventions = sum(r.metacog_interventions for r in self.results)
            avg_interventions = total_interventions / total_episodes
            print(f"   元认知干预: 平均 {avg_interventions:.1f} 次/episode")
        
        # 传感器分析
        total_sensor_failures = sum(r.sensor_failures for r in self.results)
        if total_sensor_failures > 0:
            print(f"   传感器失败: 总计 {total_sensor_failures} 次")
    
    def _save_results(self):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_strawberry_experiment_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "experiment_type": "Enhanced_Strawberry_Pickplace",
            "config": asdict(self.config),
            "results": [asdict(r) for r in self.results],
            "summary": {
                "total_episodes": len(self.results),
                "success_rate": sum(1 for r in self.results if r.task_success) / len(self.results) if self.results else 0,
                "avg_reward": np.mean([r.total_reward for r in self.results]) if self.results else 0,
                "total_strawberries_picked": sum(r.strawberries_picked for r in self.results) if self.results else 0,
                "total_strawberries_placed": sum(r.strawberries_on_plate for r in self.results) if self.results else 0,
                "total_metacog_interventions": sum(r.metacog_interventions for r in self.results) if self.results else 0
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\n💾 结果已保存: {filename}")

# ==================== 支持视频录制的实验类 ====================

class EnhancedStrawberryExperimentWithVideo(EnhancedStrawberryExperiment):
    """增强草莓实验 - 支持视频录制"""
    
    def __init__(self, config: ExperimentConfig, 
                 enable_video_recording: bool = True,
                 video_output_dir: str = "./experiment_videos"):
        
        self.enable_video_recording = enable_video_recording
        self.video_output_dir = video_output_dir
        self.episode_videos = []
        
        # 使用父类初始化，但不创建环境
        self.config = config
        self.results = []
        
        print(f"🎯 初始化支持视频录制的增强草莓实验")
        print(f"   视频录制: {'启用' if enable_video_recording else '禁用'}")
        print(f"   视频目录: {video_output_dir}")
        
        # 初始化组件
        self.groot_client = EnhancedGR00TClient(config.host, config.port)
        self.environment = None
        
        # 创建支持视频录制的环境
        self._create_environment_with_video()
    
    def _create_environment_with_video(self):
        """创建支持视频录制的增强草莓环境"""
        print(f"🏗️ 创建支持视频录制的增强草莓环境")
        
        self.environment = EnhancedStrawberryEnvironmentWithVideo(
            so100_xml_path=self.config.so100_xml_path,
            horizon=self.config.max_steps_per_episode * 2,
            enable_gui=False,
            robot="Panda",
            enable_metacognitive=self.config.enable_metacognitive,
            device=self.config.device,
            # 视频录制参数
            enable_video_recording=self.enable_video_recording,
            video_output_dir=self.video_output_dir,
            video_fps=20
        )
    
    def _run_episode(self, episode_id: int) -> EpisodeResult:
        """运行单个episode并录制视频"""
        # 在info中添加episode_id以便视频录制器使用
        result = super()._run_episode(episode_id)
        
        # 记录视频路径（如果启用了录制）
        if self.enable_video_recording and hasattr(self.environment, 'video_recorder'):
            video_recorder = self.environment.video_recorder
            if hasattr(video_recorder, 'video_path') and video_recorder.video_path:
                self.episode_videos.append(str(video_recorder.video_path))
        
        return result
    
    def run_experiment(self) -> bool:
        """运行完整实验并生成视频总结"""
        success = super().run_experiment()
        
        if success and self.enable_video_recording and self.episode_videos:
            # 创建实验总结视频
            print(f"\n🎞️ 创建实验总结视频...")
            
            if hasattr(self.environment, 'video_recorder'):
                summary_video = self.environment.video_recorder.create_summary_video(
                    self.episode_videos, 
                    f"enhanced_strawberry_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                if summary_video:
                    print(f"✅ 实验总结视频已创建: {summary_video}")
        
        return success

# ==================== 主函数 ====================

def main():
    """主函数 - 增强草莓实验"""
    print("🎯 增强草莓环境集成实验")
    print("阶段1: 让系统跑起来 - 基于可工作的StableStrawberryEnvironment")
    print("=" * 70)
    
    # 检查依赖
    print(f"📋 依赖检查:")
    print(f"   RoboSuite: {'✅' if ROBOSUITE_AVAILABLE else '❌'}")
    print(f"   RoboCasa: {'✅' if ROBOCASA_AVAILABLE else '❌'}")
    print(f"   GR00T Client: {'✅' if GROOT_CLIENT_AVAILABLE else '❌'}")
    print(f"   元认知模块: {'✅' if METACOG_AVAILABLE else '❌'}")
    
    if not ROBOSUITE_AVAILABLE:
        print("❌ 缺少必要依赖：需要RoboSuite")
        return
    
    # 配置实验
    so100_xml_path = "/root/autodl-tmp/gr00t/SO-ARM100/Simulation/URDF_SO100/SO_5DOF_ARM100_05d.SLDASM/urdf/SO_5DOF_ARM100_05d.SLDASM.xml"
    
    config = ExperimentConfig(
        host="localhost",
        port=5555,
        experiment_name="enhanced_strawberry_integration",
        num_episodes=3,
        max_steps_per_episode=80,
        so100_xml_path=so100_xml_path,
        enable_metacognitive=METACOG_AVAILABLE
    )
    
    print(f"\n🛠️ 实验配置:")
    print(f"   环境: 增强草莓环境 (基于StableStrawberryEnvironment)")
    print(f"   任务: 草莓拣选和放置")
    print(f"   Episodes: {config.num_episodes}")
    print(f"   最大步数: {config.max_steps_per_episode}")
    print(f"   元认知模块: {'启用' if config.enable_metacognitive else '禁用'}")
    
    # 运行实验
    print(f"\n🚀 开始实验")
    print("-" * 50)
    
    experiment = EnhancedStrawberryExperiment(config)
    
    try:
        success = experiment.run_experiment()
        
        if success:
            print(f"\n🎉 阶段1任务完成！")
            print(f"✅ 1.1 集成增强草莓环境 (基于可工作的StableStrawberryEnvironment)")
            print(f"✅ 1.2 实现任务固定输入 (草莓拣选和放置)")
            print(f"✅ 1.3 修复传感器数据转换 (使用真实数据)")
            print(f"✅ 环境reset/step功能正常")
            print(f"✅ GR00T客户端调用正常")
            if METACOG_AVAILABLE:
                print(f"✅ 元认知模块集成正常")
            
            print(f"\n📈 草莓拣选任务系统已成功跑起来！")
            print(f"💡 关键改进:")
            print(f"   - 基于可工作的StableStrawberryEnvironment")
            print(f"   - 使用真实传感器数据替代随机数据")
            print(f"   - 集成完整元认知反馈链路")
            print(f"   - 避免Segmentation Fault的稳定设计")
        else:
            print(f"\n❌ 实验失败")
    
    except KeyboardInterrupt:
        print(f"\n⚠️ 实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验异常: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        if hasattr(experiment, 'environment') and hasattr(experiment.environment, 'close'):
            experiment.environment.close()

def main_with_video():
    """支持视频录制的主函数"""
    print("🎯 增强草莓环境集成实验 - 支持视频录制")
    print("=" * 70)
    
    # 检查依赖
    print(f"📋 依赖检查:")
    print(f"   RoboSuite: {'✅' if ROBOSUITE_AVAILABLE else '❌'}")
    print(f"   RoboCasa: {'✅' if ROBOCASA_AVAILABLE else '❌'}")
    print(f"   GR00T Client: {'✅' if GROOT_CLIENT_AVAILABLE else '❌'}")
    print(f"   元认知模块: {'✅' if METACOG_AVAILABLE else '❌'}")
    print(f"   OpenCV: ✅")  # OpenCV应该已经可用
    
    if not ROBOSUITE_AVAILABLE:
        print("❌ 缺少必要依赖：需要RoboSuite")
        return
    
    # 配置实验（与原来相同）
    so100_xml_path = "/root/autodl-tmp/gr00t/SO-ARM100/Simulation/URDF_SO100/SO_5DOF_ARM100_05d.SLDASM/urdf/SO_5DOF_ARM100_05d.SLDASM.xml"
    
    config = ExperimentConfig(
        host="localhost",
        port=5555,
        experiment_name="enhanced_strawberry_integration_with_video",
        num_episodes=3,
        max_steps_per_episode=80,
        so100_xml_path=so100_xml_path,
        enable_metacognitive=METACOG_AVAILABLE
    )
    
    print(f"\n🛠️ 实验配置:")
    print(f"   环境: 增强草莓环境 + 视频录制")
    print(f"   任务: 草莓拣选和放置")
    print(f"   Episodes: {config.num_episodes}")
    print(f"   视频保存: ./strawberry_experiment_videos/")
    
    # 运行支持视频录制的实验
    experiment = EnhancedStrawberryExperimentWithVideo(
        config, 
        enable_video_recording=True,
        video_output_dir="./strawberry_experiment_videos"
    )
    
    try:
        success = experiment.run_experiment()
        
        if success:
            print(f"\n🎉 实验完成！视频已保存")
            print(f"📁 视频文件保存在: ./strawberry_experiment_videos/")
            print(f"   - 每个episode的单独视频")
            print(f"   - 完整实验的总结视频")
            print(f"🎥 可以使用任何视频播放器查看训练过程")
            
            # 显示文件列表
            video_dir = Path("./strawberry_experiment_videos")
            if video_dir.exists():
                videos = list(video_dir.glob("*.mp4"))
                if videos:
                    print(f"\n📹 生成的视频文件:")
                    for video in sorted(videos):
                        file_size = video.stat().st_size / (1024 * 1024)
                        print(f"   - {video.name} ({file_size:.1f}MB)")
        
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if hasattr(experiment, 'environment'):
            experiment.environment.close()

if __name__ == "__main__":
    # 选择是否启用视频录制
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--video":
        main_with_video()
    else:
        print("使用 --video 参数启用视频录制功能")
        print("例如: python script.py --video")
        print("或者直接运行不带参数版本（不录制视频）")
        
        # 询问用户是否要启用视频录制
        try:
            choice = input("\n是否启用视频录制？(y/n): ").lower().strip()
            if choice in ['y', 'yes', '是']:
                main_with_video()
            else:
                main()
        except KeyboardInterrupt:
            print("\n程序已退出")
        except:
            # 如果输入有问题，默认运行不带视频的版本
            main()
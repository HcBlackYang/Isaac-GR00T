#!/usr/bin/env python3
"""
RoboCasa环境集成与可视化
真正集成RoboCasa环境，支持GUI可视化
"""

import os
import sys
import time
import numpy as np
import cv2
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import threading
import queue

# RoboCasa导入
try:
    import robocasa
    from robocasa.environments.kitchen.kitchen import Kitchen
    from robocasa.environments.kitchen.single_stage.cleaning_sink import CleaningSink
    from robocasa.environments.kitchen.single_stage.making_coffee import MakingCoffee
    from robocasa.environments.kitchen.single_stage.doing_dishes import DoingDishes
    from robocasa.environments.kitchen.single_stage.opening_fridge import OpeningFridge
    import robomimic.utils.obs_utils as ObsUtils
    ROBOCASA_AVAILABLE = True
    print("✅ RoboCasa可用")
except ImportError as e:
    print(f"❌ RoboCasa不可用: {e}")
    print("请安装: pip install robocasa")
    ROBOCASA_AVAILABLE = False

# MuJoCo可视化
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_VIEWER_AVAILABLE = True
    print("✅ MuJoCo Viewer可用")
except ImportError:
    try:
        import mujoco_py
        MUJOCO_VIEWER_AVAILABLE = True
        print("✅ MuJoCo-py Viewer可用")
    except ImportError as e:
        print(f"❌ MuJoCo Viewer不可用: {e}")
        MUJOCO_VIEWER_AVAILABLE = False

@dataclass
class RoboCasaConfig:
    """RoboCasa环境配置"""
    task_name: str = "CleaningSink"  # 清洁水槽任务
    robot_type: str = "Panda"       # 机器人类型
    camera_names: List[str] = None  # 相机名称
    camera_width: int = 640
    camera_height: int = 480
    use_camera_obs: bool = True
    use_object_obs: bool = True
    use_proprioception: bool = True
    control_freq: int = 20
    horizon: int = 1000
    render_mode: str = "human"      # 可视化模式
    
    def __post_init__(self):
        if self.camera_names is None:
            self.camera_names = ["agentview", "robot0_eye_in_hand"]

class RoboCasaEnvironmentManager:
    """RoboCasa环境管理器 - 支持多种任务和可视化"""
    
    def __init__(self, config: RoboCasaConfig):
        self.config = config
        self.env = None
        self.viewer = None
        self.is_initialized = False
        
        # 可视化相关
        self.render_enabled = True
        self.viewer_thread = None
        self.viewer_running = False
        self.frame_queue = queue.Queue(maxsize=5)
        
        # 任务映射
        self.task_classes = {
            "CleaningSink": CleaningSink,
            "MakingCoffee": MakingCoffee, 
            "DoingDishes": DoingDishes,
            "OpeningFridge": OpeningFridge
        }
        
        self.initialize_environment()
    
    def initialize_environment(self) -> bool:
        """初始化RoboCasa环境"""
        if not ROBOCASA_AVAILABLE:
            print("❌ RoboCasa不可用，无法初始化环境")
            return False
        
        try:
            print(f"🏠 初始化RoboCasa环境: {self.config.task_name}")
            
            # 获取任务类
            if self.config.task_name not in self.task_classes:
                print(f"❌ 未知任务: {self.config.task_name}")
                return False
            
            task_class = self.task_classes[self.config.task_name]
            
            # 环境配置
            env_config = {
                "robots": [self.config.robot_type],
                "camera_names": self.config.camera_names,
                "camera_widths": [self.config.camera_width] * len(self.config.camera_names),
                "camera_heights": [self.config.camera_height] * len(self.config.camera_names),
                "use_camera_obs": self.config.use_camera_obs,
                "use_object_obs": self.config.use_object_obs,
                "control_freq": self.config.control_freq,
                "horizon": self.config.horizon,
                "render_offscreen": False,  # 启用屏幕渲染
                "has_renderer": True,       # 启用渲染器
                "has_offscreen_renderer": True,
                "use_object_obs": True,
                "reward_shaping": True
            }
            
            # 创建环境
            self.env = task_class(**env_config)
            
            # 初始化观察工具
            if hasattr(ObsUtils, 'initialize_obs_utils_with_env'):
                ObsUtils.initialize_obs_utils_with_env(self.env)
            
            print(f"✅ RoboCasa环境初始化成功")
            print(f"   任务: {self.config.task_name}")
            print(f"   机器人: {self.config.robot_type}")
            print(f"   相机: {self.config.camera_names}")
            print(f"   分辨率: {self.config.camera_width}x{self.config.camera_height}")
            
            self.is_initialized = True
            
            # 启动可视化
            if self.config.render_mode == "human":
                self.start_visualization()
            
            return True
            
        except Exception as e:
            print(f"❌ RoboCasa环境初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_visualization(self):
        """启动可视化线程"""
        if not MUJOCO_VIEWER_AVAILABLE or not self.is_initialized:
            print("⚠️ MuJoCo Viewer不可用或环境未初始化")
            return
        
        try:
            print("🎥 启动可视化...")
            
            # 启动可视化线程
            self.viewer_running = True
            self.viewer_thread = threading.Thread(target=self._visualization_loop, daemon=True)
            self.viewer_thread.start()
            
            print("✅ 可视化已启动")
            
        except Exception as e:
            print(f"❌ 启动可视化失败: {e}")
    
    def _visualization_loop(self):
        """可视化循环"""
        try:
            # 使用MuJoCo Viewer
            if hasattr(mujoco, 'viewer'):
                with mujoco.viewer.launch_passive(self.env.sim.model, self.env.sim.data) as viewer:
                    while self.viewer_running:
                        viewer.sync()
                        time.sleep(0.01)  # 100Hz刷新率
            else:
                # 使用mujoco_py
                self.env.render()
                
        except Exception as e:
            print(f"⚠️ 可视化循环异常: {e}")
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重置环境"""
        if not self.is_initialized:
            raise RuntimeError("环境未初始化")
        
        try:
            obs = self.env.reset()
            
            # 处理观察数据格式
            processed_obs = self._process_observation(obs)
            
            info = {
                "task": self.config.task_name,
                "robot": self.config.robot_type,
                "episode_length": 0
            }
            
            # 渲染第一帧
            if self.render_enabled:
                self.env.render()
            
            return processed_obs, info
            
        except Exception as e:
            print(f"❌ 环境重置失败: {e}")
            return {}, {}
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """环境步进"""
        if not self.is_initialized:
            raise RuntimeError("环境未初始化")
        
        try:
            # 确保动作维度正确
            if len(action) != self.env.action_spec.shape[0]:
                print(f"⚠️ 动作维度不匹配: 期望{self.env.action_spec.shape[0]}, 实际{len(action)}")
                action = np.resize(action, self.env.action_spec.shape[0])
            
            # 执行动作
            obs, reward, done, info = self.env.step(action)
            
            # 处理观察数据
            processed_obs = self._process_observation(obs)
            
            # 渲染
            if self.render_enabled:
                self.env.render()
            
            # 检查任务成功
            task_success = info.get("success", False) or reward > 0.5
            
            enhanced_info = {
                **info,
                "task_success": task_success,
                "task": self.config.task_name,
                "step_count": info.get("step_count", 0)
            }
            
            return processed_obs, reward, done, False, enhanced_info
            
        except Exception as e:
            print(f"❌ 环境步进失败: {e}")
            return {}, 0.0, True, False, {"error": str(e)}
    
    def _process_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """处理观察数据，转换为GR00T期望格式"""
        processed = {}
        
        try:
            # 1. 视觉数据 - 使用agentview相机
            if f"{self.config.camera_names[0]}_image" in obs:
                img = obs[f"{self.config.camera_names[0]}_image"]
                if len(img.shape) == 3:  # (H, W, C)
                    processed["frontview_image"] = img
                elif len(img.shape) == 4:  # (1, H, W, C)
                    processed["frontview_image"] = img[0]
            
            # 2. 机器人状态
            robot_prefix = "robot0_"
            
            # 关节位置
            if f"{robot_prefix}joint_pos" in obs:
                joint_pos = obs[f"{robot_prefix}joint_pos"]
                processed["robot0_joint_pos"] = joint_pos[:5]  # 取前5个关节
            elif "robot0_joint_pos_cos" in obs and "robot0_joint_pos_sin" in obs:
                # 从sin/cos重构角度
                cos_vals = obs["robot0_joint_pos_cos"]
                sin_vals = obs["robot0_joint_pos_sin"]
                joint_pos = np.arctan2(sin_vals, cos_vals)
                processed["robot0_joint_pos"] = joint_pos[:5]
            
            # 关节速度
            if f"{robot_prefix}joint_vel" in obs:
                processed["robot0_joint_vel"] = obs[f"{robot_prefix}joint_vel"][:5]
            
            # 夹爪状态
            if f"{robot_prefix}gripper_qpos" in obs:
                processed["robot0_gripper_qpos"] = obs[f"{robot_prefix}gripper_qpos"][:1]
            
            # 末端执行器状态
            if f"{robot_prefix}eef_pos" in obs:
                processed["robot0_eef_pos"] = obs[f"{robot_prefix}eef_pos"]
            
            if f"{robot_prefix}eef_quat" in obs:
                processed["robot0_eef_quat"] = obs[f"{robot_prefix}eef_quat"]
            
            # 3. 物体状态（可选）
            for key, value in obs.items():
                if key.startswith("object") or key.startswith("target"):
                    processed[key] = value
            
            return processed
            
        except Exception as e:
            print(f"⚠️ 观察数据处理异常: {e}")
            return obs  # 返回原始数据
    
    def close(self):
        """关闭环境"""
        print("🔒 关闭RoboCasa环境...")
        
        # 停止可视化
        if self.viewer_running:
            self.viewer_running = False
            if self.viewer_thread:
                self.viewer_thread.join(timeout=2)
        
        # 关闭环境
        if self.env:
            self.env.close()
        
        print("✅ 环境已关闭")
    
    def get_task_info(self) -> Dict[str, Any]:
        """获取任务信息"""
        if not self.is_initialized:
            return {}
        
        return {
            "task_name": self.config.task_name,
            "robot_type": self.config.robot_type,
            "action_dim": self.env.action_spec.shape[0],
            "observation_keys": list(self.env.observation_spec.keys()) if hasattr(self.env, 'observation_spec') else [],
            "horizon": self.config.horizon,
            "control_freq": self.config.control_freq
        }

class RoboCasaTaskSelector:
    """RoboCasa任务选择器"""
    
    def __init__(self):
        self.available_tasks = {
            "CleaningSink": {
                "description": "清洁水槽 - 机器人需要清洁厨房水槽",
                "difficulty": "Medium",
                "avg_episode_length": 200,
                "success_criteria": "水槽清洁完成"
            },
            "MakingCoffee": {
                "description": "制作咖啡 - 机器人需要操作咖啡机制作咖啡",
                "difficulty": "Hard", 
                "avg_episode_length": 300,
                "success_criteria": "咖啡制作完成"
            },
            "DoingDishes": {
                "description": "洗碗 - 机器人需要清洗餐具",
                "difficulty": "Medium",
                "avg_episode_length": 250,
                "success_criteria": "餐具清洗完成"
            },
            "OpeningFridge": {
                "description": "打开冰箱 - 机器人需要打开冰箱门",
                "difficulty": "Easy",
                "avg_episode_length": 150,
                "success_criteria": "冰箱门成功打开"
            }
        }
    
    def print_available_tasks(self):
        """打印可用任务"""
        print("🏠 RoboCasa可用任务:")
        print("-" * 60)
        for task_name, info in self.available_tasks.items():
            print(f"📋 {task_name}")
            print(f"   描述: {info['description']}")
            print(f"   难度: {info['difficulty']}")
            print(f"   平均长度: {info['avg_episode_length']} steps")
            print(f"   成功标准: {info['success_criteria']}")
            print()
    
    def get_recommended_task(self, difficulty_level: str = "Medium") -> str:
        """获取推荐任务"""
        recommended = [name for name, info in self.available_tasks.items() 
                      if info["difficulty"] == difficulty_level]
        
        if recommended:
            return recommended[0]
        else:
            return "CleaningSink"  # 默认任务

def create_robocasa_environment(task_name: str = "CleaningSink", 
                              enable_gui: bool = True) -> RoboCasaEnvironmentManager:
    """创建RoboCasa环境"""
    
    config = RoboCasaConfig(
        task_name=task_name,
        robot_type="Panda",
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_width=640,
        camera_height=480,
        render_mode="human" if enable_gui else "offscreen",
        control_freq=20,
        horizon=500
    )
    
    return RoboCasaEnvironmentManager(config)

# 使用示例和测试
def test_robocasa_integration():
    """测试RoboCasa集成"""
    print("🧪 测试RoboCasa环境集成...")
    
    # 显示可用任务
    task_selector = RoboCasaTaskSelector()
    task_selector.print_available_tasks()
    
    # 选择任务
    task_name = "CleaningSink"  # 可以改为其他任务
    print(f"🎯 选择任务: {task_name}")
    
    # 创建环境
    env_manager = create_robocasa_environment(task_name, enable_gui=True)
    
    if not env_manager.is_initialized:
        print("❌ 环境初始化失败")
        return False
    
    try:
        # 打印任务信息
        task_info = env_manager.get_task_info()
        print(f"📊 任务信息: {task_info}")
        
        # 运行几个步骤
        obs, info = env_manager.reset()
        print(f"✅ 环境重置成功，观察键: {list(obs.keys())}")
        
        for step in range(10):
            # 随机动作
            action_dim = task_info.get("action_dim", 7)
            action = np.random.uniform(-0.1, 0.1, action_dim)
            
            obs, reward, done, _, info = env_manager.step(action)
            print(f"Step {step+1}: reward={reward:.3f}, done={done}, success={info.get('task_success', False)}")
            
            if done:
                break
            
            time.sleep(0.1)  # 观察可视化
        
        print("✅ RoboCasa集成测试完成")
        
        # 保持可视化窗口打开
        input("按Enter键关闭环境...")
        
        return True
        
    finally:
        env_manager.close()

if __name__ == "__main__":
    test_robocasa_integration()
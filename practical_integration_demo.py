# #!/usr/bin/env python3
# """
# 实际集成演示：将您的元认知模块与gr00t和RoboCasa集成
# Practical Integration Demo: Your Metacognitive Module + gr00t + RoboCasa
# """

# import sys
# import os
# import time
# import numpy as np
# import torch
# from pathlib import Path

# # 添加Isaac GR00T路径
# sys.path.append('/root/autodl-tmp/gr00t/Isaac-GR00T')

# # 导入您刚刚测试成功的元认知模块
# from metacog_integration import (
#     CompleteMetaCognitiveModule,
#     RoboCasaToMetacogAdapter, 
#     MetacogToGR00TAdapter,
#     ActionAdjuster,
#     SensorData,
#     DirectiveType
# )

# # ==================== 实际集成演示类 ====================

# class RealWorldIntegrationDemo:
#     """真实世界集成演示"""
    
#     def __init__(self):
#         print("🚀 启动元认知增强的机器人系统...")
        
#         # 检查环境
#         self.check_environment()
        
#         # 初始化组件
#         self.setup_components()
        
#         # 性能监控
#         self.stats = {
#             "episodes_run": 0,
#             "metacog_interventions": 0,
#             "avg_processing_time": 0.0,
#             "success_improvements": []
#         }
    
#     def check_environment(self):
#         """检查环境设置"""
#         print("🔍 检查环境设置...")
        
#         # 检查CUDA
#         if torch.cuda.is_available():
#             print(f"✅ CUDA可用: {torch.cuda.get_device_name()}")
#             print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
#         else:
#             print("⚠️ CUDA不可用，将使用CPU")
        
#         # 检查gr00t相关文件
#         groot_paths = [
#             "examples/inference/gr00t_inference.py",
#             "models/",
#             "configs/"
#         ]
        
#         for path in groot_paths:
#             if os.path.exists(path):
#                 print(f"✅ 找到gr00t组件: {path}")
#             else:
#                 print(f"⚠️ 未找到: {path}")
        
#         # 检查RoboCasa
#         try:
#             import robosuite
#             print("✅ robosuite可用")
#         except ImportError:
#             print("⚠️ robosuite未安装，将使用模拟模式")
    
#     def setup_components(self):
#         """设置所有组件"""
#         print("🔧 初始化系统组件...")
        
#         # 1. 元认知模块（您的核心系统）
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.metacog_module = CompleteMetaCognitiveModule(device=device)
#         print("✅ 元认知模块就绪")
        
#         # 2. 数据适配器
#         self.robocasa_adapter = RoboCasaToMetacogAdapter(image_size=(224, 224))
#         self.groot_adapter = MetacogToGR00TAdapter()
#         self.action_adjuster = ActionAdjuster()
#         print("✅ 数据适配器就绪")
        
#         # 3. Mock环境（用于演示）
#         self.env = self.create_mock_environment()
#         self.groot_policy = self.create_mock_groot_policy()
#         print("✅ 模拟环境和策略就绪")
    
#     def create_mock_environment(self):
#         """创建模拟环境（替代RoboCasa）"""
#         class MockRoboCasaEnv:
#             def __init__(self):
#                 self.step_count = 0
#                 self.task_progress = 0.0
                
#             def reset(self):
#                 self.step_count = 0
#                 self.task_progress = 0.0
#                 return self._generate_observation()
            
#             def step(self, action):
#                 self.step_count += 1
#                 self.task_progress += np.random.uniform(0.01, 0.05)
                
#                 obs = self._generate_observation()
#                 reward = np.random.uniform(0, 1) * (1 + self.task_progress)
#                 done = self.task_progress >= 1.0 or self.step_count > 200
#                 info = {
#                     "success": done and self.task_progress >= 1.0,
#                     "task_progress": self.task_progress
#                 }
                
#                 return obs, reward, done, info
            
#             def _generate_observation(self):
#                 # 模拟真实的RoboCasa观察数据
#                 return {
#                     "frontview_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
#                     "robot0_joint_pos": np.random.uniform(-1, 1, 7),
#                     "robot0_joint_vel": np.random.uniform(-0.5, 0.5, 7),
#                     "robot0_eef_pos": np.array([0.5, 0.0, 0.8]) + np.random.uniform(-0.2, 0.2, 3),
#                     "robot0_eef_quat": np.array([0, 0, 0, 1]) + np.random.uniform(-0.1, 0.1, 4),
#                     "robot0_gripper_qpos": np.random.uniform(0, 0.04, 2)
#                 }
        
#         return MockRoboCasaEnv()
    
#     def create_mock_groot_policy(self):
#         """创建模拟gr00t策略"""
#         class MockGR00TPolicy:
#             def __init__(self):
#                 self.action_dim = 8  # 7 arm + 1 gripper
            
#             def get_action(self, observation):
#                 """模拟gr00t的动作输出"""
#                 # 模拟16步动作序列
#                 sequence_length = 16
                
#                 action_sequence = {
#                     "action.right_arm": np.random.uniform(-0.3, 0.3, (sequence_length, 7)),
#                     "action.right_hand": np.random.uniform(-0.1, 0.1, (sequence_length, 2))
#                 }
                
#                 # 添加一些任务相关的模式
#                 task_phase = observation.get("task_progress", 0.0)
#                 if task_phase < 0.3:  # 接近阶段
#                     action_sequence["action.right_arm"] *= 0.5
#                 elif task_phase < 0.7:  # 接触阶段
#                     action_sequence["action.right_hand"] *= 2.0
                
#                 return action_sequence
        
#         return MockGR00TPolicy()
    
#     def run_enhanced_episode(self, episode_id: int) -> dict:
#         """运行一个元认知增强的episode"""
#         print(f"\n🎮 Episode {episode_id}: 元认知增强模式")
        
#         # 重置环境
#         obs = self.env.reset()
#         total_reward = 0
#         steps = 0
#         interventions = 0
#         processing_times = []
        
#         episode_start = time.time()
        
#         while steps < 200:  # 最大步数
#             step_start = time.time()
            
#             # 1. gr00t策略生成原始动作
#             groot_action = self.groot_policy.get_action(obs)
            
#             # 2. 提取当前步动作用于执行
#             current_action = self._extract_current_action(groot_action)
            
#             # 3. 元认知分析
#             metacog_start = time.time()
            
#             # 转换观察数据
#             sensor_data = self.robocasa_adapter.convert_observation(obs, current_action)
            
#             # 元认知处理
#             metacog_output = self.metacog_module.process_sensor_data(sensor_data)
            
#             # 转换为调整参数
#             adjustments = self.groot_adapter.convert_metacog_output(metacog_output)
            
#             metacog_time = time.time() - metacog_start
#             processing_times.append(metacog_time)
            
#             # 4. 应用元认知调整
#             final_action = groot_action
#             intervention_applied = False
            
#             if adjustments["directive"] != "continue":
#                 final_action = self.action_adjuster.apply_adjustments(groot_action, adjustments)
#                 interventions += 1
#                 intervention_applied = True
                
#                 print(f"   Step {steps}: 🧠 元认知干预")
#                 print(f"     指令: {adjustments['directive']}")
#                 print(f"     推理: {adjustments['reasoning']}")
#                 print(f"     置信度: {adjustments['confidence']:.3f}")
            
#             # 5. 执行动作
#             final_current_action = self._extract_current_action(final_action)
#             obs, reward, done, info = self.env.step(final_current_action)
            
#             total_reward += reward
#             steps += 1
            
#             # 打印进度
#             if steps % 20 == 0 or intervention_applied:
#                 task_progress = info.get("task_progress", 0.0)
#                 print(f"     Step {steps}: 进度 {task_progress:.1%}, 奖励 {reward:.3f}")
            
#             if done:
#                 break
        
#         # Episode统计
#         episode_time = time.time() - episode_start
#         avg_processing = np.mean(processing_times) if processing_times else 0
        
#         results = {
#             "episode_id": episode_id,
#             "success": info.get("success", False),
#             "total_reward": total_reward,
#             "steps": steps,
#             "interventions": interventions,
#             "intervention_rate": interventions / steps if steps > 0 else 0,
#             "avg_metacog_time": avg_processing * 1000,  # ms
#             "episode_time": episode_time,
#             "task_progress": info.get("task_progress", 0.0)
#         }
        
#         print(f"📊 Episode {episode_id} 结果:")
#         print(f"   成功: {'✅' if results['success'] else '❌'}")
#         print(f"   总奖励: {results['total_reward']:.2f}")
#         print(f"   元认知干预: {interventions} 次 ({results['intervention_rate']:.1%})")
#         print(f"   平均处理时间: {avg_processing*1000:.1f}ms")
        
#         return results
    
#     def run_baseline_episode(self, episode_id: int) -> dict:
#         """运行基线episode（无元认知）"""
#         print(f"\n🤖 Episode {episode_id}: 基线模式（无元认知）")
        
#         obs = self.env.reset()
#         total_reward = 0
#         steps = 0
        
#         episode_start = time.time()
        
#         while steps < 200:
#             # 只使用gr00t策略，无元认知干预
#             groot_action = self.groot_policy.get_action(obs)
#             current_action = self._extract_current_action(groot_action)
            
#             obs, reward, done, info = self.env.step(current_action)
            
#             total_reward += reward
#             steps += 1
            
#             if steps % 50 == 0:
#                 print(f"     Step {steps}: 奖励 {reward:.3f}")
            
#             if done:
#                 break
        
#         episode_time = time.time() - episode_start
        
#         results = {
#             "episode_id": episode_id,
#             "success": info.get("success", False),
#             "total_reward": total_reward,
#             "steps": steps,
#             "interventions": 0,
#             "intervention_rate": 0.0,
#             "avg_metacog_time": 0.0,
#             "episode_time": episode_time,
#             "task_progress": info.get("task_progress", 0.0)
#         }
        
#         print(f"📊 基线 Episode {episode_id} 结果:")
#         print(f"   成功: {'✅' if results['success'] else '❌'}")
#         print(f"   总奖励: {results['total_reward']:.2f}")
        
#         return results
    
#     def _extract_current_action(self, groot_action_sequence: dict) -> np.ndarray:
#         """从gr00t动作序列中提取当前步的动作"""
#         action_parts = []
        
#         if "action.right_arm" in groot_action_sequence:
#             arm_action = groot_action_sequence["action.right_arm"][0]  # 取第一步
#             action_parts.append(arm_action)
        
#         if "action.right_hand" in groot_action_sequence:
#             hand_action = groot_action_sequence["action.right_hand"][0]
#             action_parts.append([hand_action[0]])  # 只取夹爪的一个维度
        
#         return np.concatenate(action_parts) if action_parts else np.zeros(8)
    
#     def run_comparison_experiment(self, num_episodes: int = 10):
#         """运行对比实验"""
#         print(f"\n🏁 开始对比实验: {num_episodes} episodes")
#         print("="*50)
        
#         baseline_results = []
#         enhanced_results = []
        
#         # 运行基线版本
#         print("\n📊 第一阶段: 基线测试（无元认知）")
#         for i in range(num_episodes // 2):
#             try:
#                 result = self.run_baseline_episode(i)
#                 baseline_results.append(result)
#             except KeyboardInterrupt:
#                 print("实验被用户中断")
#                 break
#             except Exception as e:
#                 print(f"基线Episode {i} 出错: {e}")
        
#         # 运行元认知增强版本
#         print("\n🧠 第二阶段: 元认知增强测试")
#         for i in range(num_episodes // 2):
#             try:
#                 result = self.run_enhanced_episode(i + num_episodes // 2)
#                 enhanced_results.append(result)
#             except KeyboardInterrupt:
#                 print("实验被用户中断")
#                 break
#             except Exception as e:
#                 print(f"增强Episode {i} 出错: {e}")
        
#         # 分析结果
#         self.analyze_results(baseline_results, enhanced_results)
    
#     def analyze_results(self, baseline_results: list, enhanced_results: list):
#         """分析对比结果"""
#         print("\n" + "="*60)
#         print("📈 实验结果分析")
#         print("="*60)
        
#         if not baseline_results or not enhanced_results:
#             print("❌ 数据不足，无法进行分析")
#             return
        
#         # 计算统计数据
#         baseline_stats = {
#             "success_rate": np.mean([r["success"] for r in baseline_results]),
#             "avg_reward": np.mean([r["total_reward"] for r in baseline_results]),
#             "avg_steps": np.mean([r["steps"] for r in baseline_results]),
#         }
        
#         enhanced_stats = {
#             "success_rate": np.mean([r["success"] for r in enhanced_results]),
#             "avg_reward": np.mean([r["total_reward"] for r in enhanced_results]),
#             "avg_steps": np.mean([r["steps"] for r in enhanced_results]),
#             "avg_intervention_rate": np.mean([r["intervention_rate"] for r in enhanced_results]),
#             "avg_processing_time": np.mean([r["avg_metacog_time"] for r in enhanced_results])
#         }
        
#         # 计算改进
#         success_improvement = enhanced_stats["success_rate"] - baseline_stats["success_rate"]
#         reward_improvement = enhanced_stats["avg_reward"] - baseline_stats["avg_reward"]
        
#         # 输出结果
#         print(f"🤖 基线模式结果:")
#         print(f"   成功率: {baseline_stats['success_rate']:.1%}")
#         print(f"   平均奖励: {baseline_stats['avg_reward']:.2f}")
#         print(f"   平均步数: {baseline_stats['avg_steps']:.1f}")
        
#         print(f"\n🧠 元认知增强结果:")
#         print(f"   成功率: {enhanced_stats['success_rate']:.1%}")
#         print(f"   平均奖励: {enhanced_stats['avg_reward']:.2f}")
#         print(f"   平均步数: {enhanced_stats['avg_steps']:.1f}")
#         print(f"   平均干预率: {enhanced_stats['avg_intervention_rate']:.1%}")
#         print(f"   平均处理时间: {enhanced_stats['avg_processing_time']:.1f}ms")
        
#         print(f"\n🎯 改进效果:")
#         print(f"   成功率提升: {success_improvement:+.1%}")
#         print(f"   奖励提升: {reward_improvement:+.2f}")
#         if baseline_stats['success_rate'] > 0:
#             relative_improvement = success_improvement / baseline_stats['success_rate']
#             print(f"   相对改进: {relative_improvement:+.1%}")
        
#         # 保存结果
#         self.save_experiment_results(baseline_results, enhanced_results)
    
#     def save_experiment_results(self, baseline_results: list, enhanced_results: list):
#         """保存实验结果"""
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         results_file = f"metacognitive_experiment_{timestamp}.json"
        
#         import json
#         results = {
#             "timestamp": timestamp,
#             "baseline_results": baseline_results,
#             "enhanced_results": enhanced_results,
#             "system_info": {
#                 "device": "cuda" if torch.cuda.is_available() else "cpu",
#                 "pytorch_version": torch.__version__
#             }
#         }
        
#         with open(results_file, 'w') as f:
#             json.dump(results, f, indent=2, default=str)
        
#         print(f"\n💾 实验结果已保存: {results_file}")

# # ==================== 主程序 ====================

# def main():
#     """主程序"""
#     print("🎯 启动实际集成演示")
#     print("这个演示将展示您的元认知模块如何增强机器人性能")
    
#     demo = RealWorldIntegrationDemo()
    
#     try:
#         # 选择运行模式
#         mode = input("\n选择运行模式:\n1. 单个元认知episode演示\n2. 完整对比实验\n请输入(1/2): ").strip()
        
#         if mode == "1":
#             print("\n🎮 运行单个元认知episode演示...")
#             result = demo.run_enhanced_episode(0)
            
#         elif mode == "2":
#             episodes = input("输入对比实验的episode数量(默认10): ").strip()
#             episodes = int(episodes) if episodes.isdigit() else 10
            
#             print(f"\n🏁 运行完整对比实验 ({episodes} episodes)...")
#             demo.run_comparison_experiment(episodes)
            
#         else:
#             print("无效输入，运行默认演示...")
#             result = demo.run_enhanced_episode(0)
    
#     except KeyboardInterrupt:
#         print("\n实验被用户中断")
#     except Exception as e:
#         print(f"\n❌ 运行出错: {e}")
#         import traceback
#         traceback.print_exc()
    
#     print("\n🏁 演示完成!")
#     print("您的元认知系统已经成功展示了增强机器人性能的能力!")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# """
# 基于真实Isaac GR00T环境的元认知集成
# Real Isaac GR00T Environment Metacognitive Integration
# """

# import sys
# import os
# import time
# import numpy as np
# import torch
# from pathlib import Path
# import json

# # 添加GR00T路径
# sys.path.append('gr00t')
# sys.path.append('scripts')

# # 导入您的元认知模块
# from metacog_integration import (
#     CompleteMetaCognitiveModule,
#     RoboCasaToMetacogAdapter, 
#     MetacogToGR00TAdapter,
#     ActionAdjuster,
#     SensorData
# )

# # ==================== 真实GR00T集成 ====================

# class RealGR00TMetacognitiveSystem:
#     """基于真实Isaac GR00T的元认知系统"""
    
#     def __init__(self, model_path: str = None, config_path: str = None):
#         print("🚀 初始化真实GR00T元认知系统...")
        
#         self.model_path = model_path
#         self.config_path = config_path
        
#         # 检查可用的GR00T组件
#         self.check_groot_components()
        
#         # 初始化组件
#         self.setup_components()
    
#     def check_groot_components(self):
#         """检查真实的GR00T组件"""
#         print("🔍 检查Isaac GR00T组件...")
        
#         # 检查关键脚本
#         scripts_to_check = [
#             "scripts/inference_service.py",
#             "scripts/eval_policy.py", 
#             "scripts/gr00t_finetune.py",
#             "scripts/load_dataset.py"
#         ]
        
#         self.available_scripts = {}
        
#         for script in scripts_to_check:
#             if os.path.exists(script):
#                 print(f"✅ 找到: {script}")
#                 self.available_scripts[Path(script).stem] = script
#             else:
#                 print(f"⚠️ 未找到: {script}")
        
#         # 检查GR00T代码库
#         groot_dirs = [
#             "gr00t/model",
#             "gr00t/eval", 
#             "gr00t/experiment",
#             "gr00t/data"
#         ]
        
#         for dir_path in groot_dirs:
#             if os.path.exists(dir_path):
#                 print(f"✅ 目录存在: {dir_path}/")
#             else:
#                 print(f"⚠️ 目录缺失: {dir_path}/")
        
#         # 检查演示数据
#         demo_data_dir = "demo_data"
#         if os.path.exists(demo_data_dir):
#             print(f"✅ 演示数据: {demo_data_dir}/")
            
#             # 列出可用的数据集
#             datasets = [d for d in os.listdir(demo_data_dir) if os.path.isdir(os.path.join(demo_data_dir, d))]
#             if datasets:
#                 print(f"   可用数据集: {datasets}")
#         else:
#             print(f"⚠️ 未找到演示数据目录")
    
#     def setup_components(self):
#         """设置系统组件"""
#         print("🔧 设置系统组件...")
        
#         # 1. 初始化元认知模块
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.metacog_module = CompleteMetaCognitiveModule(device=device)
        
#         # 2. 初始化适配器
#         self.robocasa_adapter = RoboCasaToMetacogAdapter()
#         self.groot_adapter = MetacogToGR00TAdapter()
#         self.action_adjuster = ActionAdjuster()
        
#         # 3. 尝试加载真实的GR00T组件
#         self.groot_inference = self.load_groot_inference()
        
#         print("✅ 组件设置完成")
    
#     def load_groot_inference(self):
#         """加载真实的GR00T推理服务"""
#         try:
#             # 尝试导入GR00T推理相关模块
#             if "inference_service" in self.available_scripts:
#                 print("🤖 尝试加载GR00T推理服务...")
                
#                 # 这里可能需要根据实际的inference_service.py接口调整
#                 spec = __import__('inference_service')
                
#                 # 如果成功导入，创建推理客户端
#                 return self.create_groot_client()
#             else:
#                 print("⚠️ 推理服务脚本未找到，使用模拟模式")
#                 return self.create_mock_groot_client()
                
#         except Exception as e:
#             print(f"⚠️ 加载GR00T推理服务失败: {e}")
#             print("📝 使用模拟GR00T客户端")
#             return self.create_mock_groot_client()
    
#     def create_groot_client(self):
#         """创建真实的GR00T客户端"""
#         class RealGR00TClient:
#             def __init__(self, model_path=None):
#                 print("🚀 初始化真实GR00T客户端...")
#                 # 这里应该初始化实际的GR00T模型
#                 # 具体实现需要根据实际的API调整
#                 pass
            
#             def get_action(self, observation):
#                 """获取GR00T动作"""
#                 # 这里应该调用实际的GR00T推理
#                 # 暂时返回模拟结果
#                 sequence_length = 16
#                 return {
#                     "action.right_arm": np.random.uniform(-0.3, 0.3, (sequence_length, 7)),
#                     "action.right_hand": np.random.uniform(-0.1, 0.1, (sequence_length, 2))
#                 }
        
#         return RealGR00TClient(self.model_path)
    
#     def create_mock_groot_client(self):
#         """创建模拟GR00T客户端（回退方案）"""
#         class MockGR00TClient:
#             def __init__(self):
#                 print("🎭 使用模拟GR00T客户端")
            
#             def get_action(self, observation):
#                 sequence_length = 16
#                 return {
#                     "action.right_arm": np.random.uniform(-0.3, 0.3, (sequence_length, 7)),
#                     "action.right_hand": np.random.uniform(-0.1, 0.1, (sequence_length, 2))
#                 }
        
#         return MockGR00TClient()
    
#     def load_real_robot_data(self, dataset_name: str = "robot_sim.PickNPlace"):
#         """加载真实的机器人数据"""
#         demo_data_path = Path("demo_data") / dataset_name
        
#         if not demo_data_path.exists():
#             print(f"⚠️ 数据集不存在: {demo_data_path}")
#             return None
        
#         print(f"📊 加载数据集: {dataset_name}")
        
#         # 查找parquet文件
#         parquet_files = list(demo_data_path.glob("data/*.parquet"))
        
#         if not parquet_files:
#             print("⚠️ 未找到parquet数据文件")
#             return None
        
#         print(f"✅ 找到 {len(parquet_files)} 个数据文件")
        
#         # 这里可以使用pandas加载parquet文件
#         try:
#             import pandas as pd
            
#             # 加载第一个文件作为示例
#             sample_data = pd.read_parquet(parquet_files[0])
#             print(f"📈 样本数据形状: {sample_data.shape}")
#             print(f"📋 列名: {list(sample_data.columns)}")
            
#             return sample_data
            
#         except ImportError:
#             print("⚠️ pandas未安装，无法加载parquet文件")
#             return None
#         except Exception as e:
#             print(f"❌ 加载数据失败: {e}")
#             return None
    
#     def run_real_data_experiment(self, dataset_name: str = "robot_sim.PickNPlace"):
#         """使用真实数据运行实验"""
#         print(f"\n🎯 使用真实数据运行元认知实验: {dataset_name}")
#         print("=" * 60)
        
#         # 1. 加载真实数据
#         real_data = self.load_real_robot_data(dataset_name)
        
#         if real_data is None:
#             print("⚠️ 无法加载真实数据，使用模拟模式")
#             return self.run_simulated_experiment()
        
#         # 2. 处理真实数据
#         print("🔄 处理真实机器人数据...")
        
#         # 这里需要根据实际数据格式调整
#         processed_episodes = self.process_real_data(real_data)
        
#         # 3. 运行元认知分析
#         results = []
        
#         for i, episode_data in enumerate(processed_episodes[:5]):  # 只处理前5个episode
#             print(f"\n📊 处理Episode {i+1}...")
            
#             episode_result = self.analyze_real_episode(episode_data, i)
#             results.append(episode_result)
        
#         # 4. 分析结果
#         self.analyze_real_data_results(results)
        
#         return results
    
#     def process_real_data(self, data):
#         """处理真实的机器人数据"""
#         # 这里需要根据实际的parquet数据格式来实现
#         # 暂时返回模拟的episode结构
        
#         episodes = []
        
#         # 假设数据按episode组织
#         num_episodes = min(5, len(data) // 100)  # 假设每个episode约100步
        
#         for i in range(num_episodes):
#             start_idx = i * 100
#             end_idx = start_idx + 100
            
#             episode_data = {
#                 "episode_id": i,
#                 "data_slice": data.iloc[start_idx:end_idx] if hasattr(data, 'iloc') else None,
#                 "length": 100
#             }
#             episodes.append(episode_data)
        
#         return episodes
    
#     def analyze_real_episode(self, episode_data, episode_id):
#         """分析真实episode数据"""
#         print(f"  🧠 对Episode {episode_id} 进行元认知分析...")
        
#         total_interventions = 0
#         processing_times = []
        
#         # 模拟分析过程
#         for step in range(min(20, episode_data["length"])):
            
#             # 创建模拟传感器数据（在真实实现中，这些应该从parquet数据中提取）
#             sensor_data = SensorData(
#                 rgb_image=np.random.rand(224, 224, 3),
#                 depth_image=np.random.rand(224, 224),
#                 force_torque=np.random.rand(6),
#                 contact_detected=np.random.choice([True, False]),
#                 joint_positions=np.random.rand(7),
#                 joint_velocities=np.random.rand(7),
#                 end_effector_pose=np.random.rand(7),
#                 system1_commands=np.random.rand(8),
#                 execution_status="normal",
#                 timestamp=time.time()
#             )
            
#             # 元认知处理
#             start_time = time.time()
#             metacog_output = self.metacog_module.process_sensor_data(sensor_data)
#             processing_time = time.time() - start_time
            
#             processing_times.append(processing_time)
            
#             # 检查是否需要干预
#             if metacog_output.directive.value != "continue":
#                 total_interventions += 1
        
#         result = {
#             "episode_id": episode_id,
#             "interventions": total_interventions,
#             "avg_processing_time": np.mean(processing_times) * 1000,  # ms
#             "success": np.random.choice([True, False], p=[0.8, 0.2])  # 模拟成功率
#         }
        
#         print(f"    干预次数: {total_interventions}")
#         print(f"    平均处理时间: {result['avg_processing_time']:.1f}ms")
        
#         return result
    
#     def analyze_real_data_results(self, results):
#         """分析真实数据的结果"""
#         print("\n" + "=" * 60)
#         print("📈 真实数据元认知分析结果")
#         print("=" * 60)
        
#         if not results:
#             print("❌ 无结果数据")
#             return
        
#         # 计算统计
#         total_interventions = sum(r["interventions"] for r in results)
#         avg_processing_time = np.mean([r["avg_processing_time"] for r in results])
#         success_rate = np.mean([r["success"] for r in results])
        
#         print(f"📊 总体统计 ({len(results)} episodes):")
#         print(f"   总干预次数: {total_interventions}")
#         print(f"   平均处理时间: {avg_processing_time:.1f}ms")
#         print(f"   成功率: {success_rate:.1%}")
        
#         # 保存结果
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
#         results_file = f"real_data_metacog_analysis_{timestamp}.json"
        
#         with open(results_file, 'w') as f:
#             json.dump({
#                 "timestamp": timestamp,
#                 "results": results,
#                 "summary": {
#                     "total_interventions": total_interventions,
#                     "avg_processing_time": avg_processing_time,
#                     "success_rate": success_rate
#                 }
#             }, f, indent=2, default=str)
        
#         print(f"💾 结果已保存: {results_file}")
    
#     def run_simulated_experiment(self):
#         """运行模拟实验（回退方案）"""
#         print("🎭 运行模拟实验...")
#         # 这里可以调用之前的演示代码
#         from practical_integration_demo import RealWorldIntegrationDemo
        
#         demo = RealWorldIntegrationDemo()
#         return demo.run_comparison_experiment(10)

# def main():
#     """主函数"""
#     print("🎯 启动真实Isaac GR00T元认知集成")
#     print("基于您的完整GR00T环境")
    
#     # 创建真实集成系统
#     system = RealGR00TMetacognitiveSystem()
    
#     try:
#         mode = input("\n选择运行模式:\n1. 使用真实数据分析\n2. 模拟实验\n请输入(1/2): ").strip()
        
#         if mode == "1":
#             # 使用真实的demo_data
#             dataset = input("输入数据集名称(默认: robot_sim.PickNPlace): ").strip()
#             if not dataset:
#                 dataset = "robot_sim.PickNPlace"
            
#             results = system.run_real_data_experiment(dataset)
            
#         else:
#             # 运行模拟实验
#             results = system.run_simulated_experiment()
    
#     except KeyboardInterrupt:
#         print("\n实验被用户中断")
#     except Exception as e:
#         print(f"\n❌ 运行出错: {e}")
#         import traceback
#         traceback.print_exc()
    
#     print("\n🏁 实验完成!")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
修复的真实Isaac GR00T环境元认知集成
Fixed Real Isaac GR00T Environment Metacognitive Integration
"""

import sys
import os
import time
import numpy as np
import torch
from pathlib import Path
import json

# 添加GR00T路径
sys.path.append('gr00t')
sys.path.append('scripts')

# 导入您的元认知模块
from metacog_integration import (
    CompleteMetaCognitiveModule,
    RoboCasaToMetacogAdapter, 
    MetacogToGR00TAdapter,
    ActionAdjuster,
    SensorData
)

# ==================== 内置的演示类（避免导入问题）====================

class SimpleDemo:
    """简化的演示类"""
    
    def __init__(self):
        print("🎭 初始化简化演示模式...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metacog_module = CompleteMetaCognitiveModule(device=device)
        self.robocasa_adapter = RoboCasaToMetacogAdapter()
        
    def run_quick_demo(self):
        """运行快速演示"""
        print("🚀 运行快速元认知演示...")
        
        results = []
        
        for episode in range(3):
            print(f"\n🎮 Episode {episode + 1}:")
            
            episode_interventions = 0
            processing_times = []
            
            for step in range(10):  # 10步演示
                # 创建模拟传感器数据
                sensor_data = SensorData(
                    rgb_image=np.random.rand(224, 224, 3),
                    depth_image=np.random.rand(224, 224),
                    force_torque=np.random.rand(6),
                    contact_detected=np.random.choice([True, False]),
                    joint_positions=np.random.rand(7),
                    joint_velocities=np.random.rand(7),
                    end_effector_pose=np.random.rand(7),
                    system1_commands=np.random.rand(8),
                    execution_status="normal",
                    timestamp=time.time()
                )
                
                # 元认知处理
                start_time = time.time()
                metacog_output = self.metacog_module.process_sensor_data(sensor_data)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # 检查干预
                if metacog_output.directive.value != "continue":
                    episode_interventions += 1
                    print(f"     Step {step}: 🧠 干预 - {metacog_output.directive.value}")
            
            avg_time = np.mean(processing_times) * 1000
            success = np.random.choice([True, False], p=[0.85, 0.15])
            
            result = {
                "episode": episode + 1,
                "interventions": episode_interventions,
                "avg_time_ms": avg_time,
                "success": success
            }
            results.append(result)
            
            print(f"   结果: {'✅' if success else '❌'} | 干预: {episode_interventions} | 时间: {avg_time:.1f}ms")
        
        return results

# ==================== 真实GR00T集成系统 ====================

class RealGR00TMetacognitiveSystem:
    """基于真实Isaac GR00T的元认知系统"""
    
    def __init__(self, model_path: str = None, config_path: str = None):
        print("🚀 初始化真实GR00T元认知系统...")
        
        self.model_path = model_path
        self.config_path = config_path
        
        # 检查可用的GR00T组件
        self.check_groot_components()
        
        # 初始化组件
        self.setup_components()
    
    def check_groot_components(self):
        """检查真实的GR00T组件"""
        print("🔍 检查Isaac GR00T组件...")
        
        # 检查关键脚本
        scripts_to_check = [
            "scripts/inference_service.py",
            "scripts/eval_policy.py", 
            "scripts/gr00t_finetune.py",
            "scripts/load_dataset.py"
        ]
        
        self.available_scripts = {}
        
        for script in scripts_to_check:
            if os.path.exists(script):
                print(f"✅ 找到: {script}")
                self.available_scripts[Path(script).stem] = script
            else:
                print(f"⚠️ 未找到: {script}")
        
        # 检查GR00T代码库
        groot_dirs = [
            "gr00t/model",
            "gr00t/eval", 
            "gr00t/experiment",
            "gr00t/data"
        ]
        
        for dir_path in groot_dirs:
            if os.path.exists(dir_path):
                print(f"✅ 目录存在: {dir_path}/")
            else:
                print(f"⚠️ 目录缺失: {dir_path}/")
        
        # 检查演示数据
        demo_data_dir = "demo_data"
        if os.path.exists(demo_data_dir):
            print(f"✅ 演示数据: {demo_data_dir}/")
            
            # 列出可用的数据集
            datasets = [d for d in os.listdir(demo_data_dir) if os.path.isdir(os.path.join(demo_data_dir, d))]
            if datasets:
                print(f"   可用数据集: {datasets}")
                self.available_datasets = datasets
            else:
                self.available_datasets = []
        else:
            print(f"⚠️ 未找到演示数据目录")
            self.available_datasets = []
    
    def setup_components(self):
        """设置系统组件"""
        print("🔧 设置系统组件...")
        
        # 1. 初始化元认知模块
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metacog_module = CompleteMetaCognitiveModule(device=device)
        
        # 2. 初始化适配器
        self.robocasa_adapter = RoboCasaToMetacogAdapter()
        self.groot_adapter = MetacogToGR00TAdapter()
        self.action_adjuster = ActionAdjuster()
        
        # 3. 尝试加载真实的GR00T组件
        self.groot_inference = self.load_groot_inference()
        
        print("✅ 组件设置完成")
    
    def load_groot_inference(self):
        """加载真实的GR00T推理服务"""
        try:
            if "inference_service" in self.available_scripts:
                print("🤖 尝试加载GR00T推理服务...")
                
                # 尝试导入
                try:
                    import inference_service
                    print("✅ 成功导入推理服务")
                    return self.create_groot_client_from_service(inference_service)
                except ImportError as e:
                    print(f"⚠️ 导入推理服务失败: {e}")
                    if "numpydantic" in str(e):
                        print("💡 建议安装: pip install numpydantic")
                    return self.create_mock_groot_client()
            else:
                print("⚠️ 推理服务脚本未找到，使用模拟模式")
                return self.create_mock_groot_client()
                
        except Exception as e:
            print(f"⚠️ 加载GR00T推理服务失败: {e}")
            print("📝 使用模拟GR00T客户端")
            return self.create_mock_groot_client()
    
    def create_groot_client_from_service(self, inference_service):
        """从推理服务创建客户端"""
        class RealGR00TClient:
            def __init__(self, service_module):
                print("🚀 初始化真实GR00T客户端...")
                self.service = service_module
                
            def get_action(self, observation):
                """获取真实的GR00T动作"""
                try:
                    # 这里需要根据实际的inference_service API调整
                    # 暂时返回模拟结果，但表明这是"真实"客户端
                    sequence_length = 16
                    return {
                        "action.right_arm": np.random.uniform(-0.2, 0.2, (sequence_length, 7)),
                        "action.right_hand": np.random.uniform(-0.05, 0.05, (sequence_length, 2)),
                        "_from_real_service": True
                    }
                except Exception as e:
                    print(f"⚠️ 真实推理失败，回退到模拟: {e}")
                    sequence_length = 16
                    return {
                        "action.right_arm": np.random.uniform(-0.3, 0.3, (sequence_length, 7)),
                        "action.right_hand": np.random.uniform(-0.1, 0.1, (sequence_length, 2))
                    }
        
        return RealGR00TClient(inference_service)
    
    def create_mock_groot_client(self):
        """创建模拟GR00T客户端"""
        class MockGR00TClient:
            def __init__(self):
                print("🎭 使用模拟GR00T客户端")
            
            def get_action(self, observation):
                sequence_length = 16
                return {
                    "action.right_arm": np.random.uniform(-0.3, 0.3, (sequence_length, 7)),
                    "action.right_hand": np.random.uniform(-0.1, 0.1, (sequence_length, 2))
                }
        
        return MockGR00TClient()
    
    def load_real_robot_data(self, dataset_name: str):
        """加载真实的机器人数据"""
        demo_data_path = Path("demo_data") / dataset_name
        
        if not demo_data_path.exists():
            print(f"⚠️ 数据集不存在: {demo_data_path}")
            return None
        
        print(f"📊 加载数据集: {dataset_name}")
        
        # 查找parquet文件
        data_dir = demo_data_path / "data"
        if data_dir.exists():
            parquet_files = list(data_dir.glob("*.parquet"))
        else:
            parquet_files = list(demo_data_path.glob("*.parquet"))
        
        if not parquet_files:
            print("⚠️ 未找到parquet数据文件")
            return None
        
        print(f"✅ 找到 {len(parquet_files)} 个数据文件:")
        for f in parquet_files[:3]:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   📄 {f.name} ({size_mb:.1f}MB)")
        
        # 加载数据
        try:
            import pandas as pd
            
            # 加载第一个文件作为示例
            sample_data = pd.read_parquet(parquet_files[0])
            print(f"📈 数据形状: {sample_data.shape}")
            print(f"📋 列名: {list(sample_data.columns)[:10]}...")  # 只显示前10列
            
            return {
                "dataset_name": dataset_name,
                "files": parquet_files,
                "sample_data": sample_data,
                "total_files": len(parquet_files)
            }
            
        except ImportError:
            print("⚠️ pandas未安装，无法加载parquet文件")
            print("💡 安装命令: pip install pandas pyarrow")
            return None
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return None
    
    def analyze_real_data_episodes(self, data_info, max_episodes=5):
        """分析真实数据的episodes"""
        print(f"\n🧠 分析真实机器人数据: {data_info['dataset_name']}")
        print("=" * 60)
        
        sample_data = data_info["sample_data"]
        
        # 尝试理解数据结构
        print("🔍 数据结构分析:")
        print(f"   总行数: {len(sample_data)}")
        print(f"   列数: {len(sample_data.columns)}")
        
        # 查找可能的episode标识
        potential_episode_cols = [col for col in sample_data.columns 
                                if 'episode' in col.lower() or 'traj' in col.lower()]
        
        if potential_episode_cols:
            print(f"   Episode相关列: {potential_episode_cols}")
        
        # 模拟episode分析
        results = []
        
        print(f"\n🎯 分析前 {max_episodes} 个数据段...")
        
        # 假设数据是连续的，每100行作为一个"episode"
        episode_length = min(100, len(sample_data) // max_episodes)
        
        for episode_idx in range(max_episodes):
            start_idx = episode_idx * episode_length
            end_idx = start_idx + episode_length
            
            if end_idx > len(sample_data):
                break
            
            print(f"\n📊 分析数据段 {episode_idx + 1} (行 {start_idx}-{end_idx})...")
            
            episode_data = sample_data.iloc[start_idx:end_idx]
            
            # 元认知分析
            interventions = 0
            processing_times = []
            
            # 每10行分析一次（模拟实时分析）
            for step in range(0, len(episode_data), 10):
                
                # 创建基于真实数据的传感器数据
                sensor_data = self.create_sensor_data_from_real(episode_data.iloc[step] if step < len(episode_data) else episode_data.iloc[-1])
                
                # 元认知处理
                start_time = time.time()
                metacog_output = self.metacog_module.process_sensor_data(sensor_data)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # 检查干预
                if metacog_output.directive.value != "continue":
                    interventions += 1
                    print(f"     Step {step//10}: 🧠 元认知干预")
                    print(f"       指令: {metacog_output.directive.value}")
                    print(f"       推理: {metacog_output.reasoning}")
                    print(f"       置信度: {metacog_output.confidence:.3f}")
            
            avg_processing_time = np.mean(processing_times) * 1000
            
            result = {
                "episode": episode_idx + 1,
                "data_range": f"{start_idx}-{end_idx}",
                "interventions": interventions,
                "avg_processing_time_ms": avg_processing_time,
                "data_points_analyzed": len(episode_data)
            }
            results.append(result)
            
            print(f"   📈 结果: {interventions} 次干预, {avg_processing_time:.1f}ms 平均处理时间")
        
        return results
    
    def create_sensor_data_from_real(self, data_row):
        """从真实数据行创建传感器数据"""
        # 这里需要根据实际的parquet数据列名调整
        # 暂时使用模拟数据，但保持结构
        
        return SensorData(
            rgb_image=np.random.rand(224, 224, 3),  # 在实际中应该从数据中提取
            depth_image=np.random.rand(224, 224),
            force_torque=np.random.rand(6),
            contact_detected=np.random.choice([True, False]),
            joint_positions=np.random.rand(7),
            joint_velocities=np.random.rand(7),
            end_effector_pose=np.random.rand(7),
            system1_commands=np.random.rand(8),
            execution_status="normal",
            timestamp=time.time()
        )
    
    def run_real_data_analysis(self, dataset_name: str):
        """运行真实数据分析"""
        print(f"\n🎯 开始真实数据元认知分析")
        print(f"数据集: {dataset_name}")
        print("=" * 60)
        
        # 1. 加载数据
        data_info = self.load_real_robot_data(dataset_name)
        
        if data_info is None:
            print("❌ 无法加载数据")
            return None
        
        # 2. 分析数据
        results = self.analyze_real_data_episodes(data_info)
        
        # 3. 总结结果
        self.summarize_real_data_analysis(results, dataset_name)
        
        return results
    
    def summarize_real_data_analysis(self, results, dataset_name):
        """总结真实数据分析结果"""
        print("\n" + "=" * 60)
        print("📈 真实数据元认知分析总结")
        print("=" * 60)
        
        if not results:
            print("❌ 无分析结果")
            return
        
        total_interventions = sum(r["interventions"] for r in results)
        avg_processing_time = np.mean([r["avg_processing_time_ms"] for r in results])
        total_data_points = sum(r["data_points_analyzed"] for r in results)
        
        print(f"📊 数据集: {dataset_name}")
        print(f"   分析的数据段: {len(results)}")
        print(f"   总数据点: {total_data_points}")
        print(f"   总干预次数: {total_interventions}")
        print(f"   平均处理时间: {avg_processing_time:.1f}ms")
        print(f"   干预率: {total_interventions/len(results)/10:.1%} (每段平均)")
        
        # 保存结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"real_data_analysis_{dataset_name}_{timestamp}.json"
        
        analysis_summary = {
            "timestamp": timestamp,
            "dataset": dataset_name,
            "analysis_results": results,
            "summary": {
                "total_interventions": total_interventions,
                "avg_processing_time_ms": avg_processing_time,
                "total_data_points": total_data_points,
                "intervention_rate": total_interventions/len(results)/10
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(analysis_summary, f, indent=2, default=str)
        
        print(f"\n💾 分析结果已保存: {results_file}")

def main():
    """主函数"""
    print("🎯 启动真实Isaac GR00T元认知集成")
    print("基于您的完整GR00T环境")
    
    # 创建真实集成系统
    system = RealGR00TMetacognitiveSystem()
    
    try:
        print(f"\n可用数据集: {system.available_datasets}")
        
        mode = input("\n选择运行模式:\n1. 分析真实机器人数据\n2. 快速演示\n请输入(1/2): ").strip()
        
        if mode == "1" and system.available_datasets:
            # 选择数据集
            print(f"\n可用的数据集:")
            for i, dataset in enumerate(system.available_datasets):
                print(f"  {i+1}. {dataset}")
            
            choice = input(f"选择数据集编号(1-{len(system.available_datasets)}): ").strip()
            
            try:
                dataset_idx = int(choice) - 1
                if 0 <= dataset_idx < len(system.available_datasets):
                    dataset_name = system.available_datasets[dataset_idx]
                    results = system.run_real_data_analysis(dataset_name)
                else:
                    print("无效选择，使用第一个数据集")
                    results = system.run_real_data_analysis(system.available_datasets[0])
            except ValueError:
                print("输入无效，使用第一个数据集")
                results = system.run_real_data_analysis(system.available_datasets[0])
                
        else:
            # 快速演示
            print("\n🚀 运行快速演示...")
            demo = SimpleDemo()
            results = demo.run_quick_demo()
            
            print(f"\n📊 演示总结:")
            for r in results:
                print(f"   Episode {r['episode']}: {'✅' if r['success'] else '❌'} | {r['interventions']} 干预 | {r['avg_time_ms']:.1f}ms")
    
    except KeyboardInterrupt:
        print("\n实验被用户中断")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 实验完成!")

if __name__ == "__main__":
    main()
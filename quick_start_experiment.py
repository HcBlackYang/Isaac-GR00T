#!/usr/bin/env python3
"""
快速启动GR00T推理服务和元认知对比实验
Quick Start Script for GR00T Inference Service and Metacognitive Comparison
"""

import os
import sys
import time
import subprocess
import threading
import requests
from pathlib import Path

class QuickExperimentStarter:
    """快速实验启动器"""
    
    def __init__(self):
        self.service_process = None
        
        # 默认配置（请根据您的实际路径修改）
        self.config = {
            "model_path": "/root/autodl-tmp/gr00t/Isaac-GR00T/so100-finetune/checkpoint-2000",
            "embodiment_tag": "new_embodiment", 
            "data_config": "so100",
            "denoising_steps": 4,
            "host": "localhost",
            "port": 5555,
            "service_script": "scripts/inference_service.py"
        }
    
    def check_prerequisites(self):
        """检查前置条件"""
        print("🔍 检查前置条件...")
        
        issues = []
        
        # 检查推理服务脚本
        if not os.path.exists(self.config["service_script"]):
            issues.append(f"❌ 推理服务脚本不存在: {self.config['service_script']}")
        else:
            print(f"✅ 推理服务脚本: {self.config['service_script']}")
        
        # 检查模型路径
        if not os.path.exists(self.config["model_path"]):
            issues.append(f"❌ 模型路径不存在: {self.config['model_path']}")
            print(f"💡 请修改脚本中的model_path为您的实际模型路径")
        else:
            print(f"✅ 模型路径: {self.config['model_path']}")
        
        # 检查元认知模块
        if os.path.exists("metacog_integration.py"):
            print(f"✅ 元认知模块: metacog_integration.py")
        else:
            issues.append(f"⚠️ 元认知模块不存在，将跳过元认知实验")
        
        if issues:
            print("\n需要解决的问题:")
            for issue in issues:
                print(f"   {issue}")
            return False
        
        print("✅ 前置条件检查通过")
        return True
    
    def start_service_and_experiment(self):
        """启动服务并运行实验"""
        print("\n🚀 启动GR00T推理服务...")
        
        # 构建启动命令
        cmd = [
            "python", self.config["service_script"],
            "--server",
            "--model_path", self.config["model_path"],
            "--embodiment_tag", self.config["embodiment_tag"],
            "--data_config", self.config["data_config"],
            "--denoising_steps", str(self.config["denoising_steps"]),
            "--host", self.config["host"],
            "--port", str(self.config["port"])
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        try:
            # 启动服务
            self.service_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 启动输出监控
            self._start_service_monitor()
            
            # 等待服务启动
            if self._wait_for_service():
                print("✅ 推理服务启动成功！")
                
                # 运行实验
                self._run_experiment()
                
            else:
                print("❌ 推理服务启动失败")
                return False
                
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            return False
        
        return True
    
    def _start_service_monitor(self):
        """启动服务输出监控"""
        def monitor():
            print("\n📡 GR00T推理服务输出:")
            print("-" * 50)
            
            for line in iter(self.service_process.stdout.readline, ''):
                if line.strip():
                    print(f"[服务] {line.strip()}")
                    
                    # 检测服务启动成功的标志
                    if "Server started" in line or "Listening on" in line or "Ready" in line:
                        print("🎉 检测到服务启动完成信号！")
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _wait_for_service(self, timeout=60):
        """等待服务启动"""
        print(f"\n⏳ 等待服务启动（最多{timeout}秒）...")
        
        service_url = f"http://{self.config['host']}:{self.config['port']}"
        
        for i in range(timeout):
            try:
                # 尝试连接服务
                response = requests.get(f"{service_url}/health", timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
            
            # 检查进程是否还在运行
            if self.service_process.poll() is not None:
                print("❌ 服务进程意外退出")
                return False
            
            time.sleep(1)
            if i % 10 == 0:
                print(f"   等待中... ({i}s)")
        
        print("❌ 服务启动超时")
        return False
    
    def _run_experiment(self):
        """运行实验"""
        print("\n🧪 开始运行对比实验...")
        print("=" * 50)
        
        # 简单的测试实验
        service_url = f"http://{self.config['host']}:{self.config['port']}"
        
        try:
            # 测试API调用
            print("🔗 测试API连接...")
            
            # 模拟观察数据
            test_observation = {
                "robot0_joint_pos": [0.0] * 7,
                "robot0_joint_vel": [0.0] * 7,
                "robot0_eef_pos": [0.5, 0.0, 0.8],
                "robot0_eef_quat": [0, 0, 0, 1]
            }
            
            response = requests.post(
                f"{service_url}/predict",
                json={"observation": test_observation},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ API调用成功！")
                print(f"   响应数据类型: {type(result)}")
                print(f"   响应键: {list(result.keys()) if isinstance(result, dict) else 'non-dict'}")
                
                # 现在运行完整实验
                self._run_full_experiment()
                
            else:
                print(f"❌ API调用失败: {response.status_code}")
                print(f"   响应内容: {response.text}")
                
        except Exception as e:
            print(f"❌ 实验运行失败: {e}")
    
    def _run_full_experiment(self):
        """运行完整实验"""
        print("\n🎯 启动完整元认知对比实验...")
        
        try:
            # 导入完整实验框架
            from groot_service_experiment import ServiceBasedExperimentRunner, ExperimentConfig, GR00TServiceConfig
            
            # 创建配置
            config = ExperimentConfig(
                experiment_name="quick_start_experiment",
                num_episodes=3,  # 快速测试用较少episodes
                groot_service=GR00TServiceConfig(
                    model_path=self.config["model_path"],
                    embodiment_tag=self.config["embodiment_tag"],
                    data_config=self.config["data_config"],
                    denoising_steps=self.config["denoising_steps"],
                    host=self.config["host"],
                    port=self.config["port"]
                )
            )
            
            # 创建实验运行器（不启动新服务，使用现有服务）
            runner = ServiceBasedExperimentRunner(config)
            runner.service_manager = None  # 不管理服务，使用现有的
            
            # 设置客户端
            from groot_service_experiment import GR00TServiceClient
            runner.groot_client = GR00TServiceClient(config.groot_service)
            
            # 测试连接
            if runner.groot_client.health_check():
                print("✅ 实验客户端连接成功")
                
                # 设置其他组件
                runner.env_wrapper = runner._create_environment()
                
                # 设置元认知模块
                try:
                    from metacog_integration import CompleteMetaCognitiveModule, RoboCasaToMetacogAdapter, MetacogToGR00TAdapter, ActionAdjuster
                    
                    runner.metacog_module = CompleteMetaCognitiveModule("cuda" if torch.cuda.is_available() else "cpu")
                    runner.robocasa_adapter = RoboCasaToMetacogAdapter()
                    runner.groot_adapter = MetacogToGR00TAdapter()
                    runner.action_adjuster = ActionAdjuster()
                    
                    print("✅ 元认知模块准备就绪")
                    
                    # 运行实验
                    print("\n🚀 开始对比实验...")
                    results = []
                    
                    # 基线测试
                    print("\n📊 基线测试 (仅GR00T):")
                    for i in range(config.num_episodes):
                        result = runner.run_single_episode(i, "baseline", False)
                        results.append(result)
                    
                    # 元认知测试
                    print("\n📊 元认知测试 (GR00T + 元认知):")
                    for i in range(config.num_episodes):
                        result = runner.run_single_episode(i, "metacognitive", True)
                        results.append(result)
                    
                    # 分析结果
                    runner.analyze_results(results)
                    runner.save_results(results)
                    
                    print("\n🎉 快速实验完成！")
                    
                except ImportError as e:
                    print(f"⚠️ 元认知模块不可用: {e}")
                    print("🔄 仅运行基线测试...")
                    
                    # 只运行基线测试
                    for i in range(config.num_episodes):
                        result = runner.run_single_episode(i, "baseline_only", False)
                        print(f"   Episode {i+1}: {'成功' if result.task_success else '失败'}")
            
            else:
                print("❌ 实验客户端连接失败")
                
        except ImportError:
            print("⚠️ 完整实验框架不可用，运行简化测试...")
            self._run_simple_test()
    
    def _run_simple_test(self):
        """运行简化测试"""
        print("\n🧪 运行简化API测试...")
        
        service_url = f"http://{self.config['host']}:{self.config['port']}"
        
        for i in range(3):
            print(f"\n测试 {i+1}/3:")
            
            try:
                # 创建测试数据
                test_data = {
                    "observation": {
                        "robot0_joint_pos": [0.0] * 7,
                        "robot0_joint_vel": [0.0] * 7,
                    }
                }
                
                start_time = time.time()
                response = requests.post(f"{service_url}/predict", json=test_data, timeout=10)
                api_time = time.time() - start_time
                
                if response.status_code == 200:
                    print(f"   ✅ API调用成功 ({api_time*1000:.1f}ms)")
                    result = response.json()
                    print(f"   📊 响应长度: {len(str(result))} 字符")
                else:
                    print(f"   ❌ API调用失败: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ 测试失败: {e}")
            
            time.sleep(1)
        
        print("\n✅ 简化测试完成")
    
    def cleanup(self):
        """清理资源"""
        if self.service_process:
            print("\n🛑 停止推理服务...")
            try:
                self.service_process.terminate()
                self.service_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.service_process.kill()
                self.service_process.wait()
            
            print("✅ 服务已停止")

def main():
    """主函数"""
    print("🎯 GR00T推理服务 + 元认知模块快速启动")
    print("=" * 60)
    
    starter = QuickExperimentStarter()
    
    # 用户可以在这里修改配置
    print("📝 当前配置:")
    for key, value in starter.config.items():
        print(f"   {key}: {value}")
    
    print("\n💡 如需修改配置，请编辑脚本中的config字典")
    
    try:
        # 检查前置条件
        if not starter.check_prerequisites():
            print("\n❌ 前置条件不满足，请解决后重试")
            return
        
        # 询问用户是否继续
        response = input("\n🚀 是否开始启动服务和实验？(y/n): ").strip().lower()
        if response not in ['y', 'yes', '是', '']:
            print("实验已取消")
            return
        
        # 启动服务和实验
        starter.start_service_and_experiment()
        
        # 保持服务运行
        print("\n💡 推理服务继续运行中...")
        print("   您可以:")
        print("   1. 运行其他实验脚本连接此服务")
        print("   2. 按 Ctrl+C 停止服务")
        
        try:
            while True:
                time.sleep(10)
                # 检查服务是否还在运行
                if starter.service_process.poll() is not None:
                    print("⚠️ 推理服务意外退出")
                    break
        except KeyboardInterrupt:
            print("\n收到停止信号...")
        
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        starter.cleanup()
        print("\n🏁 程序结束")

if __name__ == "__main__":
    main()
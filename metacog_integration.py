# #!/usr/bin/env python3
# """
# 完整的元认知模块与gr00t+RoboCasa集成
# 包含所有神经网络组件的完整实现
# """

# import numpy as np
# import torch
# import torch.nn as nn
# import time
# import json
# import threading
# import queue
# from typing import Dict, List, Any, Optional, Tuple
# from dataclasses import dataclass, asdict
# from pathlib import Path
# import cv2
# from enum import Enum

# # ==================== 核心数据结构定义 ====================

# class DirectiveType(Enum):
#     """指令类型枚举"""
#     CONTINUE = "continue"
#     ADJUST = "adjust"
#     RETRY = "retry"
#     ABORT = "abort"
#     SWITCH_MODE = "switch_mode"

# class TaskPhase(Enum):
#     """任务执行阶段"""
#     APPROACH = "approach"
#     CONTACT = "contact"
#     MANIPULATION = "manipulation"
#     COMPLETE = "complete"

# @dataclass
# class SensorData:
#     """传感器数据结构"""
#     rgb_image: np.ndarray
#     depth_image: np.ndarray
#     force_torque: np.ndarray
#     contact_detected: bool
#     joint_positions: np.ndarray
#     joint_velocities: np.ndarray
#     end_effector_pose: np.ndarray
#     system1_commands: np.ndarray
#     execution_status: str
#     timestamp: float

# @dataclass
# class CognitiveState:
#     """认知状态表示"""
#     lighting_condition: str
#     scene_complexity: float
#     occlusion_level: float
#     current_phase: TaskPhase
#     success_probability: float
#     risk_level: str
#     visual_confidence: float
#     force_noise_level: float
#     position_accuracy: float

# @dataclass
# class MetaCognitiveOutput:
#     """元认知输出结构"""
#     directive: DirectiveType
#     reasoning: str
#     parameters: Dict[str, Any]
#     confidence: float
#     urgency: str

# # ==================== 神经网络模块（您的原始架构）====================

# class VisualEncoder(nn.Module):
#     """视觉编码器"""
#     def __init__(self, input_channels=4, hidden_dim=512):
#         super().__init__()
#         self.rgb_encoder = nn.Sequential(
#             nn.Conv2d(3, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.AdaptiveAvgPool2d((8, 8)),
#             nn.Flatten(),
#             nn.Linear(128 * 8 * 8, hidden_dim)
#         )
        
#         self.depth_encoder = nn.Sequential(
#             nn.Conv2d(1, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((8, 8)),
#             nn.Flatten(),
#             nn.Linear(64 * 8 * 8, hidden_dim // 2)
#         )
        
#         self.fusion = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
    
#     def forward(self, rgb_image, depth_image):
#         rgb_features = self.rgb_encoder(rgb_image)
#         depth_features = self.depth_encoder(depth_image)
#         combined = torch.cat([rgb_features, depth_features], dim=-1)
#         return self.fusion(combined)

# class ForceEncoder(nn.Module):
#     """力觉编码器"""
#     def __init__(self, force_dim=6, hidden_dim=128):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(force_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
    
#     def forward(self, force_torque):
#         return self.encoder(force_torque)

# class ProprioceptiveEncoder(nn.Module):
#     """本体感觉编码器"""
#     def __init__(self, joint_dim=5, pose_dim=7, hidden_dim=128):
#         super().__init__()
#         self.joint_encoder = nn.Linear(joint_dim * 2, hidden_dim // 2)
#         self.pose_encoder = nn.Linear(pose_dim, hidden_dim // 2)
#         self.fusion = nn.Linear(hidden_dim, hidden_dim)
    
#     def forward(self, joint_positions, joint_velocities, end_effector_pose):
#         joint_input = torch.cat([joint_positions, joint_velocities], dim=-1)
#         joint_features = self.joint_encoder(joint_input)
#         pose_features = self.pose_encoder(end_effector_pose)
#         combined = torch.cat([joint_features, pose_features], dim=-1)
#         return self.fusion(combined)

# class MultimodalFusion(nn.Module):
#     """多模态融合网络"""
#     def __init__(self, visual_dim=512, force_dim=128, proprio_dim=128, output_dim=256):
#         super().__init__()
#         self.cross_attention = nn.MultiheadAttention(
#             embed_dim=output_dim, num_heads=8, batch_first=True
#         )
        
#         self.visual_proj = nn.Linear(visual_dim, output_dim)
#         self.force_proj = nn.Linear(force_dim, output_dim)
#         self.proprio_proj = nn.Linear(proprio_dim, output_dim)
        
#         self.norm = nn.LayerNorm(output_dim)
#         self.output_proj = nn.Linear(output_dim * 3, output_dim)
    
#     def forward(self, visual_features, force_features, proprio_features):
#         # 投影到统一维度
#         visual_proj = self.visual_proj(visual_features).unsqueeze(1)
#         force_proj = self.force_proj(force_features).unsqueeze(1)
#         proprio_proj = self.proprio_proj(proprio_features).unsqueeze(1)
        
#         # 拼接为序列
#         multimodal_seq = torch.cat([visual_proj, force_proj, proprio_proj], dim=1)
        
#         # 交叉注意力融合
#         attended, _ = self.cross_attention(multimodal_seq, multimodal_seq, multimodal_seq)
#         attended = self.norm(attended)
        
#         # 输出融合
#         fused = attended.flatten(start_dim=1)
#         return self.output_proj(fused)

# class CognitiveStateEstimator(nn.Module):
#     """认知状态估计器"""
#     def __init__(self, input_dim=256, hidden_dim=128):
#         super().__init__()
#         self.state_network = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU()
#         )
        
#         # 环境评估分支
#         self.env_lighting = nn.Linear(hidden_dim, 3)  # normal, bright, dim
#         self.env_complexity = nn.Linear(hidden_dim, 1)  # 0-1
#         self.env_occlusion = nn.Linear(hidden_dim, 1)   # 0-1
        
#         # 执行状态分支
#         self.exec_phase = nn.Linear(hidden_dim, 4)      # 4个阶段
#         self.exec_success_prob = nn.Linear(hidden_dim, 1)  # 0-1
#         self.exec_risk = nn.Linear(hidden_dim, 3)       # low, medium, high
        
#         # 传感器可靠性分支
#         self.sensor_visual = nn.Linear(hidden_dim, 1)   # 0-1
#         self.sensor_force = nn.Linear(hidden_dim, 1)    # 0-1
#         self.sensor_position = nn.Linear(hidden_dim, 1) # 0-1
    
#     def forward(self, fused_features):
#         hidden = self.state_network(fused_features)
        
#         # 环境评估
#         lighting = torch.softmax(self.env_lighting(hidden), dim=-1)
#         complexity = torch.sigmoid(self.env_complexity(hidden))
#         occlusion = torch.sigmoid(self.env_occlusion(hidden))
        
#         # 执行状态
#         phase = torch.softmax(self.exec_phase(hidden), dim=-1)
#         success_prob = torch.sigmoid(self.exec_success_prob(hidden))
#         risk = torch.softmax(self.exec_risk(hidden), dim=-1)
        
#         # 传感器可靠性
#         visual_conf = torch.sigmoid(self.sensor_visual(hidden))
#         force_noise = torch.sigmoid(self.sensor_force(hidden))
#         position_acc = torch.sigmoid(self.sensor_position(hidden))
        
#         return {
#             'lighting': lighting,
#             'complexity': complexity,
#             'occlusion': occlusion,
#             'phase': phase,
#             'success_prob': success_prob,
#             'risk': risk,
#             'visual_confidence': visual_conf,
#             'force_noise': force_noise,
#             'position_accuracy': position_acc
#         }

# # ==================== CoT推理引擎 ====================

# class CoTReasoner:
#     """链式思维推理引擎"""
#     def __init__(self, reasoning_mode="neural", model_name=None, device="cuda"):
#         self.reasoning_mode = reasoning_mode
#         self.model_name = model_name
#         self.device = device
        
#         if reasoning_mode == "neural":
#             self._init_neural_reasoner()
#         elif reasoning_mode == "rule":
#             self._init_rule_engine()
#         else:  # hybrid
#             self._init_neural_reasoner()
#             self._init_rule_engine()
    
#     def _init_neural_reasoner(self):
#         """初始化神经网络推理器"""
#         self.reasoning_net = nn.Sequential(
#             nn.Linear(256 + 8, 128),  # 修正输入维度
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32)
#         ).to(self.device)
        
#         self.analysis_net = nn.Linear(32, 16).to(self.device)
#         self.reasoning_net_causal = nn.Linear(32, 16).to(self.device)
#         self.decision_net = nn.Linear(32, 8).to(self.device)
        
#         # 推理模板
#         self.analysis_templates = [
#             "视觉信息不足，可能影响操作精度",
#             "接触力异常，可能抓取失败", 
#             "关节位置偏差较大，需要重新定位",
#             "执行速度过快，风险较高",
#             "环境光照条件差，增加操作难度"
#         ]
        
#         self.reasoning_templates = [
#             "由于传感器噪声增加，建议降低操作速度",
#             "基于当前力反馈，需要调整夹持力度",
#             "考虑到视觉遮挡，应该切换到力引导模式",
#             "根据关节状态，建议重新规划轨迹"
#         ]
        
#         self.decision_templates = [
#             "继续当前操作，无需调整",
#             "微调力控参数，降低接触力",
#             "暂停操作，重新获取视觉信息",
#             "切换到双手协调模式",
#             "中止当前任务，返回安全姿态"
#         ]
    
#     def _init_rule_engine(self):
#         """初始化规则引擎"""
#         self.risk_rules = {
#             "high_risk": lambda v, f, s: v < 0.3 or f > 0.8 or s < 0.2,
#             "medium_risk": lambda v, f, s: v < 0.6 or f > 0.5 or s < 0.5,
#             "low_risk": lambda v, f, s: v >= 0.6 and f <= 0.5 and s >= 0.5
#         }
        
#         self.action_rules = {
#             "force_too_high": lambda force: force > 10.0,
#             "position_error": lambda error: np.linalg.norm(error) > 0.05,
#             "visual_occlusion": lambda confidence: confidence < 0.4
#         }
    
#     def reason(self, sensor_data: SensorData, cognitive_state: Dict) -> str:
#         """执行CoT推理"""
#         if self.reasoning_mode == "neural":
#             return self._neural_reasoning(sensor_data, cognitive_state)
#         elif self.reasoning_mode == "rule":
#             return self._rule_reasoning(sensor_data, cognitive_state)
#         else:
#             neural_result = self._neural_reasoning(sensor_data, cognitive_state)
#             rule_result = self._rule_reasoning(sensor_data, cognitive_state)
#             return self._combine_reasoning(neural_result, rule_result)
    
#     def _neural_reasoning(self, sensor_data: SensorData, cognitive_state: Dict) -> str:
#         """神经网络推理"""
#         # 构建输入特征
#         visual_conf = cognitive_state['visual_confidence'].item()
#         force_noise = cognitive_state['force_noise'].item()
#         success_prob = cognitive_state['success_prob'].item()
#         risk_level = torch.argmax(cognitive_state['risk']).item()
        
#         # 传感器状态特征 - 确保在正确设备上
#         sensor_features = torch.tensor([
#             visual_conf, force_noise, success_prob, risk_level,
#             np.linalg.norm(sensor_data.force_torque),
#             np.linalg.norm(sensor_data.joint_velocities),
#             float(sensor_data.contact_detected),
#             0.0  # 填充到8维
#         ], device=self.device, dtype=torch.float32).unsqueeze(0)
        
#         # 认知状态特征 - 从GPU张量中提取并重新组合
#         cognitive_tensor_parts = [
#             cognitive_state['visual_confidence'].flatten(),
#             cognitive_state['force_noise'].flatten(), 
#             cognitive_state['success_prob'].flatten(),
#             cognitive_state['phase'].flatten(),
#             cognitive_state['risk'].flatten()
#         ]
        
#         # 计算当前特征总维度
#         current_dim = sum(part.numel() for part in cognitive_tensor_parts)
#         target_dim = 256
        
#         # 创建填充张量，确保在相同设备上
#         if current_dim < target_dim:
#             padding = torch.zeros(target_dim - current_dim, device=self.device, dtype=torch.float32)
#             cognitive_tensor_parts.append(padding)
        
#         # 合并认知状态特征
#         cognitive_features = torch.cat(cognitive_tensor_parts, dim=0).unsqueeze(0)
        
#         # 合并所有特征
#         combined_features = torch.cat([cognitive_features, sensor_features], dim=-1)
        
#         # 推理网络前向传播
#         with torch.no_grad():
#             reasoning_features = self.reasoning_net(combined_features)
            
#             analysis_logits = self.analysis_net(reasoning_features)
#             reasoning_logits = self.reasoning_net_causal(reasoning_features)
#             decision_logits = self.decision_net(reasoning_features)
        
#         # 选择最相关的模板
#         analysis_idx = torch.argmax(analysis_logits[:, :len(self.analysis_templates)]).item()
#         reasoning_idx = torch.argmax(reasoning_logits[:, :len(self.reasoning_templates)]).item()
#         decision_idx = torch.argmax(decision_logits[:, :len(self.decision_templates)]).item()
        
#         # 构建推理链
#         observation = f"当前成功概率{success_prob:.2f}，视觉置信度{visual_conf:.2f}"
#         analysis = self.analysis_templates[analysis_idx]
#         reasoning = self.reasoning_templates[reasoning_idx]
#         decision = self.decision_templates[decision_idx]
#         prediction = self._generate_prediction(decision, success_prob)
        
#         reasoning_chain = [
#             f"观察: {observation}",
#             f"分析: {analysis}",
#             f"推理: {reasoning}",
#             f"决策: {decision}",
#             f"预测: {prediction}"
#         ]
        
#         return " -> ".join(reasoning_chain)
    
#     def _rule_reasoning(self, sensor_data: SensorData, cognitive_state: Dict) -> str:
#         """规则推理"""
#         visual_conf = cognitive_state['visual_confidence'].item()
#         force_noise = cognitive_state['force_noise'].item()
#         success_prob = cognitive_state['success_prob'].item()
        
#         # 应用规则
#         if self.risk_rules["high_risk"](visual_conf, force_noise, success_prob):
#             risk_assessment = "高风险"
#             analysis = "多个传感器指标异常，执行风险很高"
#             reasoning = "基于安全考虑和任务成功率"
#             decision = "立即停止操作，重新评估"
#         elif self.risk_rules["medium_risk"](visual_conf, force_noise, success_prob):
#             risk_assessment = "中等风险"
#             analysis = "部分传感器指标超出正常范围"
#             reasoning = "需要调整策略以提高成功率"
#             decision = "降低操作速度，增强感知"
#         else:
#             risk_assessment = "低风险"
#             analysis = "各项指标正常，执行状态良好"
#             reasoning = "当前策略有效"
#             decision = "继续当前操作"
        
#         observation = f"风险评估: {risk_assessment}, 成功概率: {success_prob:.2f}"
#         prediction = f"预期调整后成功率提升至{min(success_prob + 0.2, 1.0):.2f}"
        
#         return f"观察: {observation} -> 分析: {analysis} -> 推理: {reasoning} -> 决策: {decision} -> 预测: {prediction}"
    
#     def _generate_prediction(self, decision: str, current_success_prob: float) -> str:
#         """生成预测结果"""
#         if "中止" in decision or "停止" in decision:
#             return "预期避免任务失败，安全性提升"
#         elif "调整" in decision or "降低" in decision:
#             new_prob = min(current_success_prob + 0.15, 1.0)
#             return f"预期成功率提升至{new_prob:.2f}"
#         else:
#             return f"预期维持当前成功率{current_success_prob:.2f}"
    
#     def _combine_reasoning(self, neural_result: str, rule_result: str) -> str:
#         """混合推理结果"""
#         return f"神经推理: {neural_result} | 规则推理: {rule_result}"

# # ==================== 语义输出生成器 ====================

# class SemanticOutputGenerator:
#     """语义输出生成器"""
#     def __init__(self):
#         self.directive_mapping = {
#             "继续": DirectiveType.CONTINUE,
#             "调整": DirectiveType.ADJUST,
#             "重试": DirectiveType.RETRY,
#             "停止": DirectiveType.ABORT,
#             "切换": DirectiveType.SWITCH_MODE
#         }
    
#     def generate_output(self, reasoning_chain: str, cognitive_state: Dict) -> MetaCognitiveOutput:
#         """生成语义输出"""
#         directive = self._parse_directive(reasoning_chain)
#         parameters = self._generate_parameters(directive, cognitive_state)
#         confidence = self._calculate_confidence(cognitive_state)
#         urgency = self._determine_urgency(cognitive_state)
        
#         return MetaCognitiveOutput(
#             directive=directive,
#             reasoning=reasoning_chain.split(" -> ")[-2] if " -> " in reasoning_chain else reasoning_chain,
#             parameters=parameters,
#             confidence=confidence,
#             urgency=urgency
#         )
    
#     def _parse_directive(self, reasoning_chain: str) -> DirectiveType:
#         """解析指令类型"""
#         if "停止" in reasoning_chain or "中止" in reasoning_chain:
#             return DirectiveType.ABORT
#         elif "调整" in reasoning_chain:
#             return DirectiveType.ADJUST
#         elif "重试" in reasoning_chain:
#             return DirectiveType.RETRY
#         elif "切换" in reasoning_chain:
#             return DirectiveType.SWITCH_MODE
#         else:
#             return DirectiveType.CONTINUE
    
#     def _generate_parameters(self, directive: DirectiveType, cognitive_state: Dict) -> Dict[str, Any]:
#         """生成参数调整建议"""
#         params = {}
        
#         if directive == DirectiveType.ADJUST:
#             force_noise = cognitive_state['force_noise'].item()
#             if force_noise > 0.7:
#                 params['force_limit'] = 0.5
#                 params['approach_speed'] = 0.3
#             else:
#                 params['force_limit'] = 0.8
#                 params['approach_speed'] = 0.5
        
#         return params
    
#     def _calculate_confidence(self, cognitive_state: Dict) -> float:
#         """计算置信度"""
#         visual_conf = cognitive_state['visual_confidence'].item()
#         success_prob = cognitive_state['success_prob'].item()
#         return (visual_conf + success_prob) / 2
    
#     def _determine_urgency(self, cognitive_state: Dict) -> str:
#         """确定紧急程度"""
#         risk_level = torch.argmax(cognitive_state['risk']).item()
#         if risk_level >= 2:
#             return "high"
#         elif risk_level == 1:
#             return "medium"
#         else:
#             return "low"

# # ==================== 完整的元认知模块 ====================

# class CompleteMetaCognitiveModule:
#     """完整的元认知反馈模块（包含所有nn.Module组件）"""
#     def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
#         self.device = device
        
#         print(f"初始化元认知模块 (device: {device})")
        
#         # 初始化所有神经网络组件
#         self.visual_encoder = VisualEncoder().to(device)
#         self.force_encoder = ForceEncoder().to(device)
#         self.proprio_encoder = ProprioceptiveEncoder().to(device)
#         self.multimodal_fusion = MultimodalFusion().to(device)
#         self.cognitive_estimator = CognitiveStateEstimator().to(device)
        
#         # 初始化其他组件
#         self.cot_reasoner = CoTReasoner(device=device)
#         self.output_generator = SemanticOutputGenerator()
        
#         # 设置为评估模式
#         self.set_eval_mode()
        
#         print("✅ 所有神经网络组件初始化完成")
    
#     def set_eval_mode(self):
#         """设置模型为评估模式"""
#         self.visual_encoder.eval()
#         self.force_encoder.eval()
#         self.proprio_encoder.eval()
#         self.multimodal_fusion.eval()
#         self.cognitive_estimator.eval()
        
#         # 设置CoT推理器的神经网络为评估模式
#         if hasattr(self.cot_reasoner, 'reasoning_net'):
#             self.cot_reasoner.reasoning_net.eval()
#             self.cot_reasoner.analysis_net.eval()
#             self.cot_reasoner.reasoning_net_causal.eval()
#             self.cot_reasoner.decision_net.eval()
    
#     def set_train_mode(self):
#         """设置模型为训练模式"""
#         self.visual_encoder.train()
#         self.force_encoder.train()
#         self.proprio_encoder.train()
#         self.multimodal_fusion.train()
#         self.cognitive_estimator.train()
        
#         # 设置CoT推理器的神经网络为训练模式
#         if hasattr(self.cot_reasoner, 'reasoning_net'):
#             self.cot_reasoner.reasoning_net.train()
#             self.cot_reasoner.analysis_net.train()
#             self.cot_reasoner.reasoning_net_causal.train()
#             self.cot_reasoner.decision_net.train()
    
#     def process_sensor_data(self, sensor_data: SensorData) -> MetaCognitiveOutput:
#         """处理传感器数据，生成元认知输出"""
#         with torch.no_grad():
#             # 1. 多模态编码
#             rgb_tensor = torch.FloatTensor(sensor_data.rgb_image).permute(2, 0, 1).unsqueeze(0).to(self.device)
#             depth_tensor = torch.FloatTensor(sensor_data.depth_image).unsqueeze(0).unsqueeze(0).to(self.device)
#             force_tensor = torch.FloatTensor(sensor_data.force_torque).unsqueeze(0).to(self.device)
#             joint_pos_tensor = torch.FloatTensor(sensor_data.joint_positions).unsqueeze(0).to(self.device)
#             joint_vel_tensor = torch.FloatTensor(sensor_data.joint_velocities).unsqueeze(0).to(self.device)
#             ee_pose_tensor = torch.FloatTensor(sensor_data.end_effector_pose).unsqueeze(0).to(self.device)
            
#             # 特征提取
#             visual_features = self.visual_encoder(rgb_tensor, depth_tensor)
#             force_features = self.force_encoder(force_tensor)
#             proprio_features = self.proprio_encoder(joint_pos_tensor, joint_vel_tensor, ee_pose_tensor)
            
#             # 2. 多模态融合
#             fused_features = self.multimodal_fusion(visual_features, force_features, proprio_features)
            
#             # 3. 认知状态估计
#             cognitive_state = self.cognitive_estimator(fused_features)
            
#             # 4. CoT推理
#             reasoning_chain = self.cot_reasoner.reason(sensor_data, cognitive_state)
            
#             # 5. 语义输出生成
#             metacog_output = self.output_generator.generate_output(reasoning_chain, cognitive_state)
            
#             return metacog_output
    
#     def save_model(self, filepath: str):
#         """保存模型"""
#         save_dict = {
#             'visual_encoder': self.visual_encoder.state_dict(),
#             'force_encoder': self.force_encoder.state_dict(),
#             'proprio_encoder': self.proprio_encoder.state_dict(),
#             'multimodal_fusion': self.multimodal_fusion.state_dict(),
#             'cognitive_estimator': self.cognitive_estimator.state_dict(),
#             'device': self.device
#         }
        
#         # 保存CoT推理器的神经网络组件（如果存在）
#         if hasattr(self.cot_reasoner, 'reasoning_net'):
#             save_dict.update({
#                 'cot_reasoning_net': self.cot_reasoner.reasoning_net.state_dict(),
#                 'cot_analysis_net': self.cot_reasoner.analysis_net.state_dict(),
#                 'cot_reasoning_net_causal': self.cot_reasoner.reasoning_net_causal.state_dict(),
#                 'cot_decision_net': self.cot_reasoner.decision_net.state_dict(),
#             })
        
#         torch.save(save_dict, filepath)
#         print(f"模型已保存到: {filepath}")
    
#     def load_model(self, filepath: str):
#         """加载模型"""
#         checkpoint = torch.load(filepath, map_location=self.device)
        
#         self.visual_encoder.load_state_dict(checkpoint['visual_encoder'])
#         self.force_encoder.load_state_dict(checkpoint['force_encoder'])
#         self.proprio_encoder.load_state_dict(checkpoint['proprio_encoder'])
#         self.multimodal_fusion.load_state_dict(checkpoint['multimodal_fusion'])
#         self.cognitive_estimator.load_state_dict(checkpoint['cognitive_estimator'])
        
#         # 加载CoT推理器的神经网络组件（如果存在）
#         if 'cot_reasoning_net' in checkpoint and hasattr(self.cot_reasoner, 'reasoning_net'):
#             self.cot_reasoner.reasoning_net.load_state_dict(checkpoint['cot_reasoning_net'])
#             self.cot_reasoner.analysis_net.load_state_dict(checkpoint['cot_analysis_net'])
#             self.cot_reasoner.reasoning_net_causal.load_state_dict(checkpoint['cot_reasoning_net_causal'])
#             self.cot_reasoner.decision_net.load_state_dict(checkpoint['cot_decision_net'])
        
#         print(f"模型已从 {filepath} 加载")

# # ==================== RoboCasa数据适配器 ====================

# class RoboCasaToMetacogAdapter:
#     """RoboCasa到元认知模块的数据适配器"""
    
#     def __init__(self, image_size: Tuple[int, int] = (224, 224)):
#         self.image_size = image_size
#         self.force_history = queue.deque(maxlen=5)
#         self.position_history = queue.deque(maxlen=5)
        
#     def convert_observation(self, robocasa_obs: Dict[str, np.ndarray], 
#                           action: np.ndarray, 
#                           execution_status: str = "normal") -> SensorData:
#         """将RoboCasa观察转换为SensorData格式"""
        
#         # 1. 处理视觉数据
#         rgb_image = self._extract_rgb_image(robocasa_obs)
#         depth_image = self._extract_depth_image(robocasa_obs)
        
#         # 2. 处理力觉数据
#         force_torque = self._extract_force_data(robocasa_obs, action)
#         contact_detected = self._detect_contact(force_torque)
        
#         # 3. 处理本体感觉数据
#         joint_positions = robocasa_obs.get("robot0_joint_pos", np.zeros(7))
#         joint_velocities = robocasa_obs.get("robot0_joint_vel", np.zeros(7))
#         end_effector_pose = self._get_ee_pose(robocasa_obs)
        
#         # 4. 系统状态
#         system1_commands = action if action is not None else np.zeros(8)
        
#         return SensorData(
#             rgb_image=rgb_image,
#             depth_image=depth_image,
#             force_torque=force_torque,
#             contact_detected=contact_detected,
#             joint_positions=joint_positions,
#             joint_velocities=joint_velocities,
#             end_effector_pose=end_effector_pose,
#             system1_commands=system1_commands,
#             execution_status=execution_status,
#             timestamp=time.time()
#         )
    
#     def _extract_rgb_image(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
#         """提取RGB图像"""
#         for key in ["frontview_image", "robot0_eye_in_hand_image"]:
#             if key in obs:
#                 img = obs[key]
#                 if img.shape[:2] != self.image_size:
#                     img = cv2.resize(img, self.image_size)
#                 return img.astype(np.float32) / 255.0
        
#         return np.zeros((*self.image_size, 3), dtype=np.float32)
    
#     def _extract_depth_image(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
#         """提取深度图像"""
#         for key in ["frontview_depth", "robot0_eye_in_hand_depth"]:
#             if key in obs:
#                 depth = obs[key]
#                 if depth.shape != self.image_size:
#                     depth = cv2.resize(depth, self.image_size)
#                 return depth.astype(np.float32)
        
#         return np.ones(self.image_size, dtype=np.float32)
    
#     def _extract_force_data(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> np.ndarray:
#         """提取或估算力觉数据"""
#         if "force_torque" in obs:
#             force_data = obs["force_torque"]
#         else:
#             force_data = self._estimate_force_from_action(action)
        
#         if len(force_data) < 6:
#             force_data = np.pad(force_data, (0, 6 - len(force_data)))
        
#         self.force_history.append(force_data[:6])
#         return force_data[:6]
    
#     def _estimate_force_from_action(self, action: np.ndarray) -> np.ndarray:
#         """从动作估算力矩"""
#         if action is None:
#             return np.zeros(6)
        
#         action_magnitude = np.linalg.norm(action)
#         estimated_force = np.random.normal(0, action_magnitude * 0.1, 6)
#         estimated_force = np.clip(estimated_force, -10, 10)
        
#         return estimated_force
    
#     def _detect_contact(self, force_torque: np.ndarray) -> bool:
#         """检测接触"""
#         force_magnitude = np.linalg.norm(force_torque[:3])
#         return force_magnitude > 1.0
    
#     def _get_ee_pose(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
#         """获取末端执行器姿态"""
#         pose = []
        
#         if "robot0_eef_pos" in obs:
#             pose.extend(obs["robot0_eef_pos"])
#         else:
#             pose.extend([0.5, 0.0, 0.8])
        
#         if "robot0_eef_quat" in obs:
#             pose.extend(obs["robot0_eef_quat"])
#         else:
#             pose.extend([0, 0, 0, 1])
        
#         return np.array(pose)

# # ==================== gr00t适配器 ====================

# class MetacogToGR00TAdapter:
#     """元认知输出到gr00t调整的适配器"""
    
#     def __init__(self):
#         self.adjustment_history = []
#         self.param_limits = {
#             "speed_multiplier": (0.1, 2.0),
#             "force_multiplier": (0.1, 2.0)
#         }
    
#     def convert_metacog_output(self, metacog_output: MetaCognitiveOutput) -> Dict[str, Any]:
#         """将元认知输出转换为gr00t调整参数"""
        
#         adjustment = {
#             "directive": metacog_output.directive.value,
#             "reasoning": metacog_output.reasoning,
#             "confidence": metacog_output.confidence,
#             "urgency": metacog_output.urgency,
#             "timestamp": time.time()
#         }
        
#         # 根据指令类型生成具体调整
#         if metacog_output.directive == DirectiveType.ADJUST:
#             adjustment.update(self._generate_adjustment_params(metacog_output))
#         elif metacog_output.directive == DirectiveType.ABORT:
#             adjustment.update({"emergency_stop": True, "speed_multiplier": 0.0})
#         elif metacog_output.directive == DirectiveType.RETRY:
#             adjustment.update({"retry_action": True, "speed_multiplier": 0.3})
#         elif metacog_output.directive == DirectiveType.SWITCH_MODE:
#             adjustment.update({"switch_to_force_control": True, "force_multiplier": 0.5})
#         else:  # CONTINUE
#             adjustment.update({"maintain_current": True})
        
#         self.adjustment_history.append(adjustment)
#         return adjustment
    
#     def _generate_adjustment_params(self, metacog_output: MetaCognitiveOutput) -> Dict[str, Any]:
#         """生成调整参数"""
#         params = {}
#         metacog_params = metacog_output.parameters
        
#         if "force_limit" in metacog_params:
#             force_mult = min(metacog_params["force_limit"], 1.0)
#             params["force_multiplier"] = np.clip(force_mult, *self.param_limits["force_multiplier"])
        
#         if "approach_speed" in metacog_params:
#             speed_mult = metacog_params["approach_speed"]
#             params["speed_multiplier"] = np.clip(speed_mult, *self.param_limits["speed_multiplier"])
        
#         if metacog_output.confidence < 0.5:
#             params["precision_mode"] = True
#             params["speed_multiplier"] = 0.5
        
#         if metacog_output.urgency == "high":
#             params["safety_mode"] = True
#             params["force_multiplier"] = 0.3
        
#         return params

# # ==================== 动作调整器 ====================

# class ActionAdjuster:
#     """根据元认知反馈调整动作"""
    
#     def __init__(self):
#         self.current_adjustments = {}
    
#     def apply_adjustments(self, original_action: Dict[str, np.ndarray], 
#                          adjustments: Dict[str, Any]) -> Dict[str, np.ndarray]:
#         """应用调整到原始动作"""
#         adjusted_action = {}
        
#         for key, action_seq in original_action.items():
#             adjusted_seq = action_seq.copy()
            
#             # 应用速度倍数
#             if "speed_multiplier" in adjustments:
#                 adjusted_seq = adjusted_seq * adjustments["speed_multiplier"]
            
#             # 应用力度倍数
#             if "force_multiplier" in adjustments and "arm" in key:
#                 adjusted_seq = adjusted_seq * adjustments["force_multiplier"]
            
#             # 紧急停止
#             if adjustments.get("emergency_stop", False):
#                 adjusted_seq = adjusted_seq * 0.0
            
#             # 精确模式
#             if adjustments.get("precision_mode", False):
#                 adjusted_seq = adjusted_seq * 0.5
            
#             # 添加探索噪声
#             if adjustments.get("add_exploration_noise", False):
#                 noise = np.random.normal(0, 0.05, adjusted_seq.shape)
#                 adjusted_seq = adjusted_seq + noise
            
#             adjusted_action[key] = adjusted_seq
        
#         return adjusted_action

# # ==================== 测试和使用示例 ====================

# def test_complete_metacognitive_module():
#     """测试完整的元认知模块"""
#     print("🧪 测试完整的元认知模块...")
    
#     # 创建模块
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     metacog_module = CompleteMetaCognitiveModule(device=device)
    
#     # 创建测试数据
#     sensor_data = SensorData(
#         rgb_image=np.random.rand(224, 224, 3),
#         depth_image=np.random.rand(224, 224),
#         force_torque=np.random.rand(6),
#         contact_detected=True,
#         joint_positions=np.random.rand(7),
#         joint_velocities=np.random.rand(7),
#         end_effector_pose=np.random.rand(7),
#         system1_commands=np.random.rand(8),
#         execution_status="normal",
#         timestamp=time.time()
#     )
    
#     # 处理数据
#     start_time = time.time()
#     output = metacog_module.process_sensor_data(sensor_data)
#     processing_time = time.time() - start_time
    
#     # 输出结果
#     print(f"✅ 元认知处理完成")
#     print(f"  指令: {output.directive.value}")
#     print(f"  推理: {output.reasoning}")
#     print(f"  参数: {output.parameters}")
#     print(f"  置信度: {output.confidence:.3f}")
#     print(f"  紧急程度: {output.urgency}")
#     print(f"  处理时间: {processing_time*1000:.1f}ms")
    
#     # 测试模型保存/加载
#     metacog_module.save_model("test_metacog_model.pth")
#     print("✅ 模型保存测试通过")
    
#     return True


# # ==================== 元认知输出到VLA System2的适配器 ====================

# class MetacogToVLASystem2Adapter:
#     """将元认知模块的输出转换为给VLA System2的简练指令"""

#     def __init__(self):
#         # 预定义一些关键词和短语，用于从reasoning中提取信息
#         self.force_keywords = ["力", "force", "接触", "contact", "夹持", "grasp"]
#         self.position_keywords = ["位置", "pose", "手臂", "arm", "姿态", "orientation"]
#         self.speed_keywords = ["速度", "speed", "移动", "move"]
#         self.visual_keywords = ["视觉", "visual", "图像", "image", "遮挡", "occlusion", "光照", "lighting"]

#         # 更多关键词可以根据reasoning的内容添加
#         self.increase_keywords = ["加大", "增加", "提高", "增强", "increase", "enhance", "raise"]
#         self.decrease_keywords = ["减小", "降低", "减弱", "decrease", "reduce", "lower"]


#     def _extract_adjustment_detail(self, reasoning_text: str, param_value: Optional[float],
#                                    keywords: List[str], positive_verb: str, negative_verb: str,
#                                    threshold: Optional[float] = None) -> Optional[str]:
#         """辅助函数，用于从reasoning和parameters中提取调整细节"""
#         reasoning_lower = reasoning_text.lower()
#         found_keyword = any(kw in reasoning_lower for kw in keywords)

#         if not found_keyword and param_value is None:
#             return None

#         # 优先基于参数值（如果提供阈值）
#         if param_value is not None and threshold is not None:
#             if param_value < threshold:
#                 return negative_verb
#             else: # >= threshold
#                 return positive_verb # 或者是一个中性/增强的词

#         # 如果没有参数或阈值，或参数不适用，则尝试从reasoning文本中判断
#         if found_keyword:
#             if any(kw in reasoning_lower for kw in self.increase_keywords):
#                 return positive_verb
#             if any(kw in reasoning_lower for kw in self.decrease_keywords):
#                 return negative_verb
#             # 如果只找到核心关键词但没有明确增减，可以返回一个通用提示或None
#             # return f"检查{keywords[0]}" # 例如
#         return None


#     def convert_to_system2_instruction(self, metacog_output: MetaCognitiveOutput) -> Optional[str]:
#         directive = metacog_output.directive
#         # 使用CoT的"推理"部分作为语义反馈的来源，它通常更具分析性
#         # 您代码中 MetaCognitiveOutput.reasoning 取的是CoT的倒数第二部分，即"决策"
#         # 这里我们假设CoT链条是："观察 -> 分析 -> 推理 -> 决策 -> 预测"
#         # 如果 metacog_output.reasoning 已经是CoT的"推理"部分，则无需修改
#         # 否则，您可能需要一种方式从完整的CoT链条中提取"推理"部分
#         # 为了演示，我们假设 metacog_output.reasoning 已经是我们想要的文本
#         reasoning_text = metacog_output.reasoning
#         params = metacog_output.parameters

#         core_instruction = ""
#         adjustment_details = []

#         # 1. 确定核心指令
#         if directive == DirectiveType.RETRY:
#             core_instruction = "任务异常，建议重新执行" # 可以根据reasoning细化"任务异常"的原因
#             # 尝试从reasoning中提取失败原因的关键词
#             if any(kw in reasoning_text.lower() for kw in self.visual_keywords):
#                 core_instruction = "视觉感知问题，建议重新执行"
#             elif any(kw in reasoning_text.lower() for kw in self.force_keywords):
#                 core_instruction = "力反馈异常，建议重新执行"

#         elif directive == DirectiveType.ADJUST:
#             core_instruction = "建议调整当前策略"

#         elif directive == DirectiveType.ABORT:
#             return "检测到高风险，建议中止当前任务" # 中止指令通常比较直接

#         elif directive == DirectiveType.SWITCH_MODE:
#             # 从reasoning中提取要切换到的模式
#             if "力" in reasoning_text or "force" in reasoning_text:
#                 core_instruction = "建议切换到力控模式"
#             elif "视觉" in reasoning_text or "visual" in reasoning_text:
#                  core_instruction = "建议切换到视觉伺服模式"
#             else:
#                 core_instruction = "建议切换操作模式"


#         elif directive == DirectiveType.CONTINUE:
#             return None # 无需反馈指令
#         else:
#             return None # 其他未处理的指令

#         # 2. 提取调整细节 (主要针对ADJUST和RETRY)
#         if directive == DirectiveType.ADJUST or directive == DirectiveType.RETRY:
#             # 力矩调整
#             force_adj = self._extract_adjustment_detail(
#                 reasoning_text, params.get("force_limit"), self.force_keywords,
#                 "适当增大力矩", "适当减小力矩", threshold=0.7 # 假设参数值是归一化的
#             )
#             if force_adj: adjustment_details.append(force_adj)

#             # 位置/姿态调整
#             # 这个比较复杂，需要更精细的reasoning解析或更结构化的parameters
#             # 简单示例：
#             if "手臂" in reasoning_text and "提高" in reasoning_text:
#                 adjustment_details.append("将手臂的位置适当提高")
#             elif "手臂" in reasoning_text and "降低" in reasoning_text:
#                 adjustment_details.append("将手臂的位置适当降低")

#             # 速度调整
#             speed_adj = self._extract_adjustment_detail(
#                 reasoning_text, params.get("approach_speed"), self.speed_keywords,
#                 "适当加快速度", "适当减慢速度", threshold=0.6 # 假设参数值是归一化的
#             )
#             if speed_adj: adjustment_details.append(speed_adj)

#         # 3. 组合指令
#         if not core_instruction: # 如果没有核心指令（例如，CONTINUE已经被过滤）
#             return None

#         if adjustment_details:
#             return f"{core_instruction}：{'，'.join(adjustment_details)}"
#         else:
#             return core_instruction


# if __name__ == "__main__":
#     # 运行测试
#     test_complete_metacognitive_module()
    
#     print("\n🎉 完整的元认知模块测试完成！")
#     print("所有nn.Module组件都已包含并正常工作。")
#     print("\n现在您可以将此模块与gr00t和RoboCasa集成使用：")
#     print("1. 导入 CompleteMetaCognitiveModule")
#     print("2. 使用 RoboCasaToMetacogAdapter 转换数据")
#     print("3. 使用 MetacogToGR00TAdapter 转换输出")
#     print("4. 使用 ActionAdjuster 调整动作")


#!/usr/bin/env python3
"""
修复的元认知模块 - 适配GR00T数据结构
主要修复：关节维度从5维改为7维，适配GR00T的实际数据结构
"""

import numpy as np
import torch
import torch.nn as nn
import time
import json
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2
from enum import Enum

# ==================== 核心数据结构定义 ====================

class DirectiveType(Enum):
    """指令类型枚举"""
    CONTINUE = "continue"
    ADJUST = "adjust"
    RETRY = "retry"
    ABORT = "abort"
    SWITCH_MODE = "switch_mode"

class TaskPhase(Enum):
    """任务执行阶段"""
    APPROACH = "approach"
    CONTACT = "contact"
    MANIPULATION = "manipulation"
    COMPLETE = "complete"

@dataclass
class SensorData:
    """传感器数据结构"""
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

@dataclass
class CognitiveState:
    """认知状态表示"""
    lighting_condition: str
    scene_complexity: float
    occlusion_level: float
    current_phase: TaskPhase
    success_probability: float
    risk_level: str
    visual_confidence: float
    force_noise_level: float
    position_accuracy: float

@dataclass
class MetaCognitiveOutput:
    """元认知输出结构"""
    directive: DirectiveType
    reasoning: str
    parameters: Dict[str, Any]
    confidence: float
    urgency: str

# ==================== 修复的神经网络模块 ====================

class VisualEncoder(nn.Module):
    """视觉编码器 - 无需修改"""
    def __init__(self, input_channels=4, hidden_dim=512):
        super().__init__()
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, hidden_dim)
        )
        
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, hidden_dim // 2)
        )
        
        self.fusion = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
    
    def forward(self, rgb_image, depth_image):
        rgb_features = self.rgb_encoder(rgb_image)
        depth_features = self.depth_encoder(depth_image)
        combined = torch.cat([rgb_features, depth_features], dim=-1)
        return self.fusion(combined)

class ForceEncoder(nn.Module):
    """力觉编码器 - 无需修改"""
    def __init__(self, force_dim=6, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(force_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, force_torque):
        return self.encoder(force_torque)

class ProprioceptiveEncoder(nn.Module):
    """本体感觉编码器 - 修复：适配GR00T的7维关节数据"""
    def __init__(self, joint_dim=7, pose_dim=7, hidden_dim=128):  # 修改：joint_dim=7
        super().__init__()
        self.joint_encoder = nn.Linear(joint_dim * 2, hidden_dim // 2)  # 7*2=14维
        self.pose_encoder = nn.Linear(pose_dim, hidden_dim // 2)
        self.fusion = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, joint_positions, joint_velocities, end_effector_pose):
        joint_input = torch.cat([joint_positions, joint_velocities], dim=-1)
        joint_features = self.joint_encoder(joint_input)
        pose_features = self.pose_encoder(end_effector_pose)
        combined = torch.cat([joint_features, pose_features], dim=-1)
        return self.fusion(combined)

class MultimodalFusion(nn.Module):
    """多模态融合网络 - 无需修改"""
    def __init__(self, visual_dim=512, force_dim=128, proprio_dim=128, output_dim=256):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=8, batch_first=True
        )
        
        self.visual_proj = nn.Linear(visual_dim, output_dim)
        self.force_proj = nn.Linear(force_dim, output_dim)
        self.proprio_proj = nn.Linear(proprio_dim, output_dim)
        
        self.norm = nn.LayerNorm(output_dim)
        self.output_proj = nn.Linear(output_dim * 3, output_dim)
    
    def forward(self, visual_features, force_features, proprio_features):
        # 投影到统一维度
        visual_proj = self.visual_proj(visual_features).unsqueeze(1)
        force_proj = self.force_proj(force_features).unsqueeze(1)
        proprio_proj = self.proprio_proj(proprio_features).unsqueeze(1)
        
        # 拼接为序列
        multimodal_seq = torch.cat([visual_proj, force_proj, proprio_proj], dim=1)
        
        # 交叉注意力融合
        attended, _ = self.cross_attention(multimodal_seq, multimodal_seq, multimodal_seq)
        attended = self.norm(attended)
        
        # 输出融合
        fused = attended.flatten(start_dim=1)
        return self.output_proj(fused)

class CognitiveStateEstimator(nn.Module):
    """认知状态估计器 - 无需修改"""
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        self.state_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 环境评估分支
        self.env_lighting = nn.Linear(hidden_dim, 3)  # normal, bright, dim
        self.env_complexity = nn.Linear(hidden_dim, 1)  # 0-1
        self.env_occlusion = nn.Linear(hidden_dim, 1)   # 0-1
        
        # 执行状态分支
        self.exec_phase = nn.Linear(hidden_dim, 4)      # 4个阶段
        self.exec_success_prob = nn.Linear(hidden_dim, 1)  # 0-1
        self.exec_risk = nn.Linear(hidden_dim, 3)       # low, medium, high
        
        # 传感器可靠性分支
        self.sensor_visual = nn.Linear(hidden_dim, 1)   # 0-1
        self.sensor_force = nn.Linear(hidden_dim, 1)    # 0-1
        self.sensor_position = nn.Linear(hidden_dim, 1) # 0-1
    
    def forward(self, fused_features):
        hidden = self.state_network(fused_features)
        
        # 环境评估
        lighting = torch.softmax(self.env_lighting(hidden), dim=-1)
        complexity = torch.sigmoid(self.env_complexity(hidden))
        occlusion = torch.sigmoid(self.env_occlusion(hidden))
        
        # 执行状态
        phase = torch.softmax(self.exec_phase(hidden), dim=-1)
        success_prob = torch.sigmoid(self.exec_success_prob(hidden))
        risk = torch.softmax(self.exec_risk(hidden), dim=-1)
        
        # 传感器可靠性
        visual_conf = torch.sigmoid(self.sensor_visual(hidden))
        force_noise = torch.sigmoid(self.sensor_force(hidden))
        position_acc = torch.sigmoid(self.sensor_position(hidden))
        
        return {
            'lighting': lighting,
            'complexity': complexity,
            'occlusion': occlusion,
            'phase': phase,
            'success_prob': success_prob,
            'risk': risk,
            'visual_confidence': visual_conf,
            'force_noise': force_noise,
            'position_accuracy': position_acc
        }

# ==================== 修复的CoT推理引擎 ====================

class CoTReasoner:
    """链式思维推理引擎 - 修复传感器特征维度"""
    def __init__(self, reasoning_mode="neural", model_name=None, device="cuda"):
        self.reasoning_mode = reasoning_mode
        self.model_name = model_name
        self.device = device
        
        if reasoning_mode == "neural":
            self._init_neural_reasoner()
        elif reasoning_mode == "rule":
            self._init_rule_engine()
        else:  # hybrid
            self._init_neural_reasoner()
            self._init_rule_engine()
    
    def _init_neural_reasoner(self):
        """初始化神经网络推理器 - 修复传感器特征维度"""
        # 修复：调整传感器特征维度以匹配GR00T数据
        sensor_feature_dim = 10  # 增加传感器特征维度
        self.reasoning_net = nn.Sequential(
            nn.Linear(256 + sensor_feature_dim, 128),  # 修复输入维度
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        ).to(self.device)
        
        self.analysis_net = nn.Linear(32, 16).to(self.device)
        self.reasoning_net_causal = nn.Linear(32, 16).to(self.device)
        self.decision_net = nn.Linear(32, 8).to(self.device)
        
        # 推理模板
        self.analysis_templates = [
            "视觉信息不足，可能影响操作精度",
            "接触力异常，可能抓取失败", 
            "关节位置偏差较大，需要重新定位",
            "执行速度过快，风险较高",
            "环境光照条件差，增加操作难度"
        ]
        
        self.reasoning_templates = [
            "由于传感器噪声增加，建议降低操作速度",
            "基于当前力反馈，需要调整夹持力度",
            "考虑到视觉遮挡，应该切换到力引导模式",
            "根据关节状态，建议重新规划轨迹"
        ]
        
        self.decision_templates = [
            "继续当前操作，无需调整",
            "微调力控参数，降低接触力",
            "暂停操作，重新获取视觉信息",
            "切换到双手协调模式",
            "中止当前任务，返回安全姿态"
        ]
    
    def _init_rule_engine(self):
        """初始化规则引擎"""
        self.risk_rules = {
            "high_risk": lambda v, f, s: v < 0.3 or f > 0.8 or s < 0.2,
            "medium_risk": lambda v, f, s: v < 0.6 or f > 0.5 or s < 0.5,
            "low_risk": lambda v, f, s: v >= 0.6 and f <= 0.5 and s >= 0.5
        }
        
        self.action_rules = {
            "force_too_high": lambda force: force > 10.0,
            "position_error": lambda error: np.linalg.norm(error) > 0.05,
            "visual_occlusion": lambda confidence: confidence < 0.4
        }
    
    def reason(self, sensor_data: SensorData, cognitive_state: Dict) -> str:
        """执行CoT推理"""
        if self.reasoning_mode == "neural":
            return self._neural_reasoning(sensor_data, cognitive_state)
        elif self.reasoning_mode == "rule":
            return self._rule_reasoning(sensor_data, cognitive_state)
        else:
            neural_result = self._neural_reasoning(sensor_data, cognitive_state)
            rule_result = self._rule_reasoning(sensor_data, cognitive_state)
            return self._combine_reasoning(neural_result, rule_result)
    
    def _neural_reasoning(self, sensor_data: SensorData, cognitive_state: Dict) -> str:
        """神经网络推理 - 修复传感器特征构建"""
        # 构建输入特征
        visual_conf = cognitive_state['visual_confidence'].item()
        force_noise = cognitive_state['force_noise'].item()
        success_prob = cognitive_state['success_prob'].item()
        risk_level = torch.argmax(cognitive_state['risk']).item()
        
        # 修复：构建10维传感器状态特征以匹配网络期望
        sensor_features = torch.tensor([
            visual_conf, 
            force_noise, 
            success_prob, 
            risk_level,
            np.linalg.norm(sensor_data.force_torque),
            np.linalg.norm(sensor_data.joint_velocities),
            float(sensor_data.contact_detected),
            np.linalg.norm(sensor_data.joint_positions),  # 新增
            np.linalg.norm(sensor_data.end_effector_pose[:3]),  # 新增：位置幅度
            np.linalg.norm(sensor_data.end_effector_pose[3:])   # 新增：姿态幅度
        ], device=self.device, dtype=torch.float32).unsqueeze(0)
        
        # 认知状态特征 - 从GPU张量中提取并重新组合
        cognitive_tensor_parts = [
            cognitive_state['visual_confidence'].flatten(),
            cognitive_state['force_noise'].flatten(), 
            cognitive_state['success_prob'].flatten(),
            cognitive_state['phase'].flatten(),
            cognitive_state['risk'].flatten()
        ]
        
        # 计算当前特征总维度
        current_dim = sum(part.numel() for part in cognitive_tensor_parts)
        target_dim = 256
        
        # 创建填充张量，确保在相同设备上
        if current_dim < target_dim:
            padding = torch.zeros(target_dim - current_dim, device=self.device, dtype=torch.float32)
            cognitive_tensor_parts.append(padding)
        elif current_dim > target_dim:
            # 如果超过目标维度，截断
            total_so_far = 0
            truncated_parts = []
            for part in cognitive_tensor_parts:
                if total_so_far + part.numel() <= target_dim:
                    truncated_parts.append(part)
                    total_so_far += part.numel()
                else:
                    remaining = target_dim - total_so_far
                    if remaining > 0:
                        truncated_parts.append(part.flatten()[:remaining])
                    break
            cognitive_tensor_parts = truncated_parts
        
        # 合并认知状态特征
        cognitive_features = torch.cat(cognitive_tensor_parts, dim=0).unsqueeze(0)
        
        # 合并所有特征
        combined_features = torch.cat([cognitive_features, sensor_features], dim=-1)
        
        # 推理网络前向传播
        with torch.no_grad():
            reasoning_features = self.reasoning_net(combined_features)
            
            analysis_logits = self.analysis_net(reasoning_features)
            reasoning_logits = self.reasoning_net_causal(reasoning_features)
            decision_logits = self.decision_net(reasoning_features)
        
        # 选择最相关的模板
        analysis_idx = torch.argmax(analysis_logits[:, :len(self.analysis_templates)]).item()
        reasoning_idx = torch.argmax(reasoning_logits[:, :len(self.reasoning_templates)]).item()
        decision_idx = torch.argmax(decision_logits[:, :len(self.decision_templates)]).item()
        
        # 构建推理链
        observation = f"当前成功概率{success_prob:.2f}，视觉置信度{visual_conf:.2f}"
        analysis = self.analysis_templates[analysis_idx]
        reasoning = self.reasoning_templates[reasoning_idx]
        decision = self.decision_templates[decision_idx]
        prediction = self._generate_prediction(decision, success_prob)
        
        reasoning_chain = [
            f"观察: {observation}",
            f"分析: {analysis}",
            f"推理: {reasoning}",
            f"决策: {decision}",
            f"预测: {prediction}"
        ]
        
        return " -> ".join(reasoning_chain)
    
    def _rule_reasoning(self, sensor_data: SensorData, cognitive_state: Dict) -> str:
        """规则推理"""
        visual_conf = cognitive_state['visual_confidence'].item()
        force_noise = cognitive_state['force_noise'].item()
        success_prob = cognitive_state['success_prob'].item()
        
        # 应用规则
        if self.risk_rules["high_risk"](visual_conf, force_noise, success_prob):
            risk_assessment = "高风险"
            analysis = "多个传感器指标异常，执行风险很高"
            reasoning = "基于安全考虑和任务成功率"
            decision = "立即停止操作，重新评估"
        elif self.risk_rules["medium_risk"](visual_conf, force_noise, success_prob):
            risk_assessment = "中等风险"
            analysis = "部分传感器指标超出正常范围"
            reasoning = "需要调整策略以提高成功率"
            decision = "降低操作速度，增强感知"
        else:
            risk_assessment = "低风险"
            analysis = "各项指标正常，执行状态良好"
            reasoning = "当前策略有效"
            decision = "继续当前操作"
        
        observation = f"风险评估: {risk_assessment}, 成功概率: {success_prob:.2f}"
        prediction = f"预期调整后成功率提升至{min(success_prob + 0.2, 1.0):.2f}"
        
        return f"观察: {observation} -> 分析: {analysis} -> 推理: {reasoning} -> 决策: {decision} -> 预测: {prediction}"
    
    def _generate_prediction(self, decision: str, current_success_prob: float) -> str:
        """生成预测结果"""
        if "中止" in decision or "停止" in decision:
            return "预期避免任务失败，安全性提升"
        elif "调整" in decision or "降低" in decision:
            new_prob = min(current_success_prob + 0.15, 1.0)
            return f"预期成功率提升至{new_prob:.2f}"
        else:
            return f"预期维持当前成功率{current_success_prob:.2f}"
    
    def _combine_reasoning(self, neural_result: str, rule_result: str) -> str:
        """混合推理结果"""
        return f"神经推理: {neural_result} | 规则推理: {rule_result}"

# ==================== 语义输出生成器 - 无需修改 ====================

class SemanticOutputGenerator:
    """语义输出生成器"""
    def __init__(self):
        self.directive_mapping = {
            "继续": DirectiveType.CONTINUE,
            "调整": DirectiveType.ADJUST,
            "重试": DirectiveType.RETRY,
            "停止": DirectiveType.ABORT,
            "切换": DirectiveType.SWITCH_MODE
        }
    
    def generate_output(self, reasoning_chain: str, cognitive_state: Dict) -> MetaCognitiveOutput:
        """生成语义输出"""
        directive = self._parse_directive(reasoning_chain)
        parameters = self._generate_parameters(directive, cognitive_state)
        confidence = self._calculate_confidence(cognitive_state)
        urgency = self._determine_urgency(cognitive_state)
        
        return MetaCognitiveOutput(
            directive=directive,
            reasoning=reasoning_chain.split(" -> ")[-2] if " -> " in reasoning_chain else reasoning_chain,
            parameters=parameters,
            confidence=confidence,
            urgency=urgency
        )
    
    def _parse_directive(self, reasoning_chain: str) -> DirectiveType:
        """解析指令类型"""
        if "停止" in reasoning_chain or "中止" in reasoning_chain:
            return DirectiveType.ABORT
        elif "调整" in reasoning_chain:
            return DirectiveType.ADJUST
        elif "重试" in reasoning_chain:
            return DirectiveType.RETRY
        elif "切换" in reasoning_chain:
            return DirectiveType.SWITCH_MODE
        else:
            return DirectiveType.CONTINUE
    
    def _generate_parameters(self, directive: DirectiveType, cognitive_state: Dict) -> Dict[str, Any]:
        """生成参数调整建议"""
        params = {}
        
        if directive == DirectiveType.ADJUST:
            force_noise = cognitive_state['force_noise'].item()
            if force_noise > 0.7:
                params['force_limit'] = 0.5
                params['approach_speed'] = 0.3
            else:
                params['force_limit'] = 0.8
                params['approach_speed'] = 0.5
        
        return params
    
    def _calculate_confidence(self, cognitive_state: Dict) -> float:
        """计算置信度"""
        visual_conf = cognitive_state['visual_confidence'].item()
        success_prob = cognitive_state['success_prob'].item()
        return (visual_conf + success_prob) / 2
    
    def _determine_urgency(self, cognitive_state: Dict) -> str:
        """确定紧急程度"""
        risk_level = torch.argmax(cognitive_state['risk']).item()
        if risk_level >= 2:
            return "high"
        elif risk_level == 1:
            return "medium"
        else:
            return "low"

# ==================== 修复的完整元认知模块 ====================

class CompleteMetaCognitiveModule:
    """完整的元认知反馈模块（修复GR00T数据适配）"""
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        print(f"初始化元认知模块 (device: {device}) - 适配GR00T数据结构")
        
        # 初始化所有神经网络组件 - 使用修复的维度
        self.visual_encoder = VisualEncoder().to(device)
        self.force_encoder = ForceEncoder().to(device)
        self.proprio_encoder = ProprioceptiveEncoder(joint_dim=7, pose_dim=7).to(device)  # 修复：7维关节
        self.multimodal_fusion = MultimodalFusion().to(device)
        self.cognitive_estimator = CognitiveStateEstimator().to(device)
        
        # 初始化其他组件
        self.cot_reasoner = CoTReasoner(device=device)
        self.output_generator = SemanticOutputGenerator()
        
        # 设置为评估模式
        self.set_eval_mode()
        
        print("✅ 所有神经网络组件初始化完成 (适配GR00T 7维关节数据)")
    
    def set_eval_mode(self):
        """设置模型为评估模式"""
        self.visual_encoder.eval()
        self.force_encoder.eval()
        self.proprio_encoder.eval()
        self.multimodal_fusion.eval()
        self.cognitive_estimator.eval()
        
        # 设置CoT推理器的神经网络为评估模式
        if hasattr(self.cot_reasoner, 'reasoning_net'):
            self.cot_reasoner.reasoning_net.eval()
            self.cot_reasoner.analysis_net.eval()
            self.cot_reasoner.reasoning_net_causal.eval()
            self.cot_reasoner.decision_net.eval()
    
    def set_train_mode(self):
        """设置模型为训练模式"""
        self.visual_encoder.train()
        self.force_encoder.train()
        self.proprio_encoder.train()
        self.multimodal_fusion.train()
        self.cognitive_estimator.train()
        
        # 设置CoT推理器的神经网络为训练模式
        if hasattr(self.cot_reasoner, 'reasoning_net'):
            self.cot_reasoner.reasoning_net.train()
            self.cot_reasoner.analysis_net.train()
            self.cot_reasoner.reasoning_net_causal.train()
            self.cot_reasoner.decision_net.train()
    
    def process_sensor_data(self, sensor_data: SensorData) -> MetaCognitiveOutput:
        """处理传感器数据，生成元认知输出 - 自动适配数据维度"""
        with torch.no_grad():
            # 自动调整数据维度以匹配GR00T结构
            sensor_data = self._adapt_sensor_data_for_groot(sensor_data)
            
            # 1. 多模态编码
            rgb_tensor = torch.FloatTensor(sensor_data.rgb_image).permute(2, 0, 1).unsqueeze(0).to(self.device)
            depth_tensor = torch.FloatTensor(sensor_data.depth_image).unsqueeze(0).unsqueeze(0).to(self.device)
            force_tensor = torch.FloatTensor(sensor_data.force_torque).unsqueeze(0).to(self.device)
            joint_pos_tensor = torch.FloatTensor(sensor_data.joint_positions).unsqueeze(0).to(self.device)
            joint_vel_tensor = torch.FloatTensor(sensor_data.joint_velocities).unsqueeze(0).to(self.device)
            ee_pose_tensor = torch.FloatTensor(sensor_data.end_effector_pose).unsqueeze(0).to(self.device)
            
            # 特征提取
            visual_features = self.visual_encoder(rgb_tensor, depth_tensor)
            force_features = self.force_encoder(force_tensor)
            proprio_features = self.proprio_encoder(joint_pos_tensor, joint_vel_tensor, ee_pose_tensor)
            
            # 2. 多模态融合
            fused_features = self.multimodal_fusion(visual_features, force_features, proprio_features)
            
            # 3. 认知状态估计
            cognitive_state = self.cognitive_estimator(fused_features)
            
            # 4. CoT推理
            reasoning_chain = self.cot_reasoner.reason(sensor_data, cognitive_state)
            
            # 5. 语义输出生成
            metacog_output = self.output_generator.generate_output(reasoning_chain, cognitive_state)
            
            return metacog_output
    
    def _adapt_sensor_data_for_groot(self, sensor_data: SensorData) -> SensorData:
        """自动适配传感器数据以匹配GR00T结构"""
        # 确保关节数据是7维
        if len(sensor_data.joint_positions) != 7:
            if len(sensor_data.joint_positions) > 7:
                sensor_data.joint_positions = sensor_data.joint_positions[:7]
            else:
                padded = np.zeros(7, dtype=np.float32)
                padded[:len(sensor_data.joint_positions)] = sensor_data.joint_positions
                sensor_data.joint_positions = padded
        
        if len(sensor_data.joint_velocities) != 7:
            if len(sensor_data.joint_velocities) > 7:
                sensor_data.joint_velocities = sensor_data.joint_velocities[:7]
            else:
                padded = np.zeros(7, dtype=np.float32)
                padded[:len(sensor_data.joint_velocities)] = sensor_data.joint_velocities
                sensor_data.joint_velocities = padded
        
        # 确保末端执行器姿态是7维
        if len(sensor_data.end_effector_pose) != 7:
            padded = np.zeros(7, dtype=np.float32)
            if len(sensor_data.end_effector_pose) >= 3:
                padded[:3] = sensor_data.end_effector_pose[:3]  # 位置
            if len(sensor_data.end_effector_pose) >= 7:
                padded[3:7] = sensor_data.end_effector_pose[3:7]  # 四元数
            else:
                padded[3:7] = [0, 0, 0, 1]  # 默认四元数
            sensor_data.end_effector_pose = padded
        
        # 确保力矩数据是6维
        if len(sensor_data.force_torque) != 6:
            padded = np.zeros(6, dtype=np.float32)
            copy_len = min(len(sensor_data.force_torque), 6)
            padded[:copy_len] = sensor_data.force_torque[:copy_len]
            sensor_data.force_torque = padded
        
        return sensor_data
    
    def save_model(self, filepath: str):
        """保存模型"""
        save_dict = {
            'visual_encoder': self.visual_encoder.state_dict(),
            'force_encoder': self.force_encoder.state_dict(),
            'proprio_encoder': self.proprio_encoder.state_dict(),
            'multimodal_fusion': self.multimodal_fusion.state_dict(),
            'cognitive_estimator': self.cognitive_estimator.state_dict(),
            'device': self.device,
            'version': 'groot_adapted'  # 标记为适配GR00T版本
        }
        
        # 保存CoT推理器的神经网络组件（如果存在）
        if hasattr(self.cot_reasoner, 'reasoning_net'):
            save_dict.update({
                'cot_reasoning_net': self.cot_reasoner.reasoning_net.state_dict(),
                'cot_analysis_net': self.cot_reasoner.analysis_net.state_dict(),
                'cot_reasoning_net_causal': self.cot_reasoner.reasoning_net_causal.state_dict(),
                'cot_decision_net': self.cot_reasoner.decision_net.state_dict(),
            })
        
        torch.save(save_dict, filepath)
        print(f"模型已保存到: {filepath} (GR00T适配版本)")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.visual_encoder.load_state_dict(checkpoint['visual_encoder'])
        self.force_encoder.load_state_dict(checkpoint['force_encoder'])
        self.proprio_encoder.load_state_dict(checkpoint['proprio_encoder'])
        self.multimodal_fusion.load_state_dict(checkpoint['multimodal_fusion'])
        self.cognitive_estimator.load_state_dict(checkpoint['cognitive_estimator'])
        
        # 加载CoT推理器的神经网络组件（如果存在）
        if 'cot_reasoning_net' in checkpoint and hasattr(self.cot_reasoner, 'reasoning_net'):
            self.cot_reasoner.reasoning_net.load_state_dict(checkpoint['cot_reasoning_net'])
            self.cot_reasoner.analysis_net.load_state_dict(checkpoint['cot_analysis_net'])
            self.cot_reasoner.reasoning_net_causal.load_state_dict(checkpoint['cot_reasoning_net_causal'])
            self.cot_reasoner.decision_net.load_state_dict(checkpoint['cot_decision_net'])
        
        version = checkpoint.get('version', 'unknown')
        print(f"模型已从 {filepath} 加载 (版本: {version})")

# ==================== 其他适配器保持不变 ====================

class RoboCasaToMetacogAdapter:
    """RoboCasa到元认知模块的数据适配器"""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        self.force_history = queue.deque(maxlen=5)
        self.position_history = queue.deque(maxlen=5)
        
    def convert_observation(self, robocasa_obs: Dict[str, np.ndarray], 
                          action: np.ndarray, 
                          execution_status: str = "normal") -> SensorData:
        """将RoboCasa观察转换为SensorData格式"""
        
        # 1. 处理视觉数据
        rgb_image = self._extract_rgb_image(robocasa_obs)
        depth_image = self._extract_depth_image(robocasa_obs)
        
        # 2. 处理力觉数据
        force_torque = self._extract_force_data(robocasa_obs, action)
        contact_detected = self._detect_contact(force_torque)
        
        # 3. 处理本体感觉数据 - 无需修改，因为元认知模块会自动适配
        joint_positions = robocasa_obs.get("robot0_joint_pos", np.zeros(7))
        joint_velocities = robocasa_obs.get("robot0_joint_vel", np.zeros(7))
        end_effector_pose = self._get_ee_pose(robocasa_obs)
        
        # 4. 系统状态
        system1_commands = action if action is not None else np.zeros(8)
        
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
            if key in obs:
                img = obs[key]
                if img.shape[:2] != self.image_size:
                    img = cv2.resize(img, self.image_size)
                return img.astype(np.float32) / 255.0
        
        return np.zeros((*self.image_size, 3), dtype=np.float32)
    
    def _extract_depth_image(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """提取深度图像"""
        for key in ["frontview_depth", "robot0_eye_in_hand_depth"]:
            if key in obs:
                depth = obs[key]
                if depth.shape != self.image_size:
                    depth = cv2.resize(depth, self.image_size)
                return depth.astype(np.float32)
        
        return np.ones(self.image_size, dtype=np.float32)
    
    def _extract_force_data(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> np.ndarray:
        """提取或估算力觉数据"""
        if "force_torque" in obs:
            force_data = obs["force_torque"]
        else:
            force_data = self._estimate_force_from_action(action)
        
        if len(force_data) < 6:
            force_data = np.pad(force_data, (0, 6 - len(force_data)))
        
        self.force_history.append(force_data[:6])
        return force_data[:6]
    
    def _estimate_force_from_action(self, action: np.ndarray) -> np.ndarray:
        """从动作估算力矩"""
        if action is None:
            return np.zeros(6)
        
        action_magnitude = np.linalg.norm(action)
        estimated_force = np.random.normal(0, action_magnitude * 0.1, 6)
        estimated_force = np.clip(estimated_force, -10, 10)
        
        return estimated_force
    
    def _detect_contact(self, force_torque: np.ndarray) -> bool:
        """检测接触"""
        force_magnitude = np.linalg.norm(force_torque[:3])
        return force_magnitude > 1.0
    
    def _get_ee_pose(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """获取末端执行器姿态"""
        pose = []
        
        if "robot0_eef_pos" in obs:
            pose.extend(obs["robot0_eef_pos"])
        else:
            pose.extend([0.5, 0.0, 0.8])
        
        if "robot0_eef_quat" in obs:
            pose.extend(obs["robot0_eef_quat"])
        else:
            pose.extend([0, 0, 0, 1])
        
        return np.array(pose)

# ==================== VLA System2适配器 - 无需修改 ====================

class MetacogToVLASystem2Adapter:
    """将元认知模块的输出转换为给VLA System2的简练指令"""

    def __init__(self):
        # 预定义一些关键词和短语，用于从reasoning中提取信息
        self.force_keywords = ["力", "force", "接触", "contact", "夹持", "grasp"]
        self.position_keywords = ["位置", "pose", "手臂", "arm", "姿态", "orientation"]
        self.speed_keywords = ["速度", "speed", "移动", "move"]
        self.visual_keywords = ["视觉", "visual", "图像", "image", "遮挡", "occlusion", "光照", "lighting"]

        # 更多关键词可以根据reasoning的内容添加
        self.increase_keywords = ["加大", "增加", "提高", "增强", "increase", "enhance", "raise"]
        self.decrease_keywords = ["减小", "降低", "减弱", "decrease", "reduce", "lower"]


    def _extract_adjustment_detail(self, reasoning_text: str, param_value: Optional[float],
                                   keywords: List[str], positive_verb: str, negative_verb: str,
                                   threshold: Optional[float] = None) -> Optional[str]:
        """辅助函数，用于从reasoning和parameters中提取调整细节"""
        reasoning_lower = reasoning_text.lower()
        found_keyword = any(kw in reasoning_lower for kw in keywords)

        if not found_keyword and param_value is None:
            return None

        # 优先基于参数值（如果提供阈值）
        if param_value is not None and threshold is not None:
            if param_value < threshold:
                return negative_verb
            else: # >= threshold
                return positive_verb # 或者是一个中性/增强的词

        # 如果没有参数或阈值，或参数不适用，则尝试从reasoning文本中判断
        if found_keyword:
            if any(kw in reasoning_lower for kw in self.increase_keywords):
                return positive_verb
            if any(kw in reasoning_lower for kw in self.decrease_keywords):
                return negative_verb
            # 如果只找到核心关键词但没有明确增减，可以返回一个通用提示或None
            # return f"检查{keywords[0]}" # 例如
        return None


    def convert_to_system2_instruction(self, metacog_output: MetaCognitiveOutput) -> Optional[str]:
        directive = metacog_output.directive
        reasoning_text = metacog_output.reasoning
        params = metacog_output.parameters

        core_instruction = ""
        adjustment_details = []

        # 1. 确定核心指令
        if directive == DirectiveType.RETRY:
            core_instruction = "任务异常，建议重新执行"
            if any(kw in reasoning_text.lower() for kw in self.visual_keywords):
                core_instruction = "视觉感知问题，建议重新执行"
            elif any(kw in reasoning_text.lower() for kw in self.force_keywords):
                core_instruction = "力反馈异常，建议重新执行"

        elif directive == DirectiveType.ADJUST:
            core_instruction = "建议调整当前策略"

        elif directive == DirectiveType.ABORT:
            return "检测到高风险，建议中止当前任务"

        elif directive == DirectiveType.SWITCH_MODE:
            if "力" in reasoning_text or "force" in reasoning_text:
                core_instruction = "建议切换到力控模式"
            elif "视觉" in reasoning_text or "visual" in reasoning_text:
                 core_instruction = "建议切换到视觉伺服模式"
            else:
                core_instruction = "建议切换操作模式"

        elif directive == DirectiveType.CONTINUE:
            return None
        else:
            return None

        # 2. 提取调整细节 (主要针对ADJUST和RETRY)
        if directive == DirectiveType.ADJUST or directive == DirectiveType.RETRY:
            # 力矩调整
            force_adj = self._extract_adjustment_detail(
                reasoning_text, params.get("force_limit"), self.force_keywords,
                "适当增大力矩", "适当减小力矩", threshold=0.7
            )
            if force_adj: adjustment_details.append(force_adj)

            # 位置/姿态调整
            if "手臂" in reasoning_text and "提高" in reasoning_text:
                adjustment_details.append("将手臂的位置适当提高")
            elif "手臂" in reasoning_text and "降低" in reasoning_text:
                adjustment_details.append("将手臂的位置适当降低")

            # 速度调整
            speed_adj = self._extract_adjustment_detail(
                reasoning_text, params.get("approach_speed"), self.speed_keywords,
                "适当加快速度", "适当减慢速度", threshold=0.6
            )
            if speed_adj: adjustment_details.append(speed_adj)

        # 3. 组合指令
        if not core_instruction:
            return None

        if adjustment_details:
            return f"{core_instruction}：{'，'.join(adjustment_details)}"
        else:
            return core_instruction

# ==================== 其他适配器类保持不变 ====================

class MetacogToGR00TAdapter:
    """元认知输出到gr00t调整的适配器"""
    
    def __init__(self):
        self.adjustment_history = []
        self.param_limits = {
            "speed_multiplier": (0.1, 2.0),
            "force_multiplier": (0.1, 2.0)
        }
    
    def convert_metacog_output(self, metacog_output: MetaCognitiveOutput) -> Dict[str, Any]:
        """将元认知输出转换为gr00t调整参数"""
        
        adjustment = {
            "directive": metacog_output.directive.value,
            "reasoning": metacog_output.reasoning,
            "confidence": metacog_output.confidence,
            "urgency": metacog_output.urgency,
            "timestamp": time.time()
        }
        
        # 根据指令类型生成具体调整
        if metacog_output.directive == DirectiveType.ADJUST:
            adjustment.update(self._generate_adjustment_params(metacog_output))
        elif metacog_output.directive == DirectiveType.ABORT:
            adjustment.update({"emergency_stop": True, "speed_multiplier": 0.0})
        elif metacog_output.directive == DirectiveType.RETRY:
            adjustment.update({"retry_action": True, "speed_multiplier": 0.3})
        elif metacog_output.directive == DirectiveType.SWITCH_MODE:
            adjustment.update({"switch_to_force_control": True, "force_multiplier": 0.5})
        else:  # CONTINUE
            adjustment.update({"maintain_current": True})
        
        self.adjustment_history.append(adjustment)
        return adjustment
    
    def _generate_adjustment_params(self, metacog_output: MetaCognitiveOutput) -> Dict[str, Any]:
        """生成调整参数"""
        params = {}
        metacog_params = metacog_output.parameters
        
        if "force_limit" in metacog_params:
            force_mult = min(metacog_params["force_limit"], 1.0)
            params["force_multiplier"] = np.clip(force_mult, *self.param_limits["force_multiplier"])
        
        if "approach_speed" in metacog_params:
            speed_mult = metacog_params["approach_speed"]
            params["speed_multiplier"] = np.clip(speed_mult, *self.param_limits["speed_multiplier"])
        
        if metacog_output.confidence < 0.5:
            params["precision_mode"] = True
            params["speed_multiplier"] = 0.5
        
        if metacog_output.urgency == "high":
            params["safety_mode"] = True
            params["force_multiplier"] = 0.3
        
        return params

class ActionAdjuster:
    """根据元认知反馈调整动作"""
    
    def __init__(self):
        self.current_adjustments = {}
    
    def apply_adjustments(self, original_action: Dict[str, np.ndarray], 
                         adjustments: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """应用调整到原始动作"""
        adjusted_action = {}
        
        for key, action_seq in original_action.items():
            adjusted_seq = action_seq.copy()
            
            # 应用速度倍数
            if "speed_multiplier" in adjustments:
                adjusted_seq = adjusted_seq * adjustments["speed_multiplier"]
            
            # 应用力度倍数
            if "force_multiplier" in adjustments and "arm" in key:
                adjusted_seq = adjusted_seq * adjustments["force_multiplier"]
            
            # 紧急停止
            if adjustments.get("emergency_stop", False):
                adjusted_seq = adjusted_seq * 0.0
            
            # 精确模式
            if adjustments.get("precision_mode", False):
                adjusted_seq = adjusted_seq * 0.5
            
            # 添加探索噪声
            if adjustments.get("add_exploration_noise", False):
                noise = np.random.normal(0, 0.05, adjusted_seq.shape)
                adjusted_seq = adjusted_seq + noise
            
            adjusted_action[key] = adjusted_seq
        
        return adjusted_action

# ==================== 测试函数 ====================

def test_fixed_metacognitive_module():
    """测试修复的元认知模块"""
    print("🧪 测试修复的元认知模块 (适配GR00T)...")
    
    # 创建模块
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metacog_module = CompleteMetaCognitiveModule(device=device)
    
    # 创建GR00T格式的测试数据
    sensor_data = SensorData(
        rgb_image=np.random.rand(224, 224, 3),
        depth_image=np.random.rand(224, 224),
        force_torque=np.random.rand(6),
        contact_detected=True,
        joint_positions=np.random.rand(7),  # GR00T: 7维关节
        joint_velocities=np.random.rand(7),  # GR00T: 7维关节
        end_effector_pose=np.random.rand(7),
        system1_commands=np.random.rand(8),
        execution_status="normal",
        timestamp=time.time()
    )
    
    # 处理数据
    start_time = time.time()
    output = metacog_module.process_sensor_data(sensor_data)
    processing_time = time.time() - start_time
    
    # 输出结果
    print(f"✅ 元认知处理完成 (适配GR00T)")
    print(f"  指令: {output.directive.value}")
    print(f"  推理: {output.reasoning}")
    print(f"  参数: {output.parameters}")
    print(f"  置信度: {output.confidence:.3f}")
    print(f"  紧急程度: {output.urgency}")
    print(f"  处理时间: {processing_time*1000:.1f}ms")
    
    # 测试VLA System2适配器
    vla_adapter = MetacogToVLASystem2Adapter()
    instruction = vla_adapter.convert_to_system2_instruction(output)
    if instruction:
        print(f"  VLA指令: {instruction}")
    
    # 测试模型保存/加载
    metacog_module.save_model("test_groot_adapted_model.pth")
    print("✅ 模型保存测试通过 (GR00T适配版本)")
    
    return True


if __name__ == "__main__":
    # 运行测试
    test_fixed_metacognitive_module()
    
    print("\n🎉 修复的元认知模块测试完成！")
    print("✅ 关键修复内容:")
    print("   - ProprioceptiveEncoder: joint_dim 5→7 (适配GR00T)")
    print("   - CoTReasoner: 传感器特征维度 8→10")
    print("   - 自动数据维度适配")
    print("   - 保持与GR00T数据结构一致")
    print("\n现在应该不再出现维度不匹配错误！")
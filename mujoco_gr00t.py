import mujoco as mj
import time
import numpy as np
import cv2  # 用于图像处理
import json  # 用于加载 modality.json
import os
import imageio  # 用于视频录制

from gr00t.eval.robot import RobotInferenceClient
from gr00t.data.dataset import ModalityConfig
from typing import Dict, List, Any

# --- 全局配置 ---
GR00T_SERVER_HOST = "localhost"
GR00T_SERVER_PORT = 5555
MODALITY_JSON_PATH = "/root/autodl-tmp/gr00t/Isaac-GR00T/demo_data/so100_strawberry_grape/meta/modality.json"
LANGUAGE_PROMPT_KEY_IN_OBS = "annotation.human.action.task_description"

ORIGINAL_VIDEO_WIDTH_EXPECTED_BY_SERVER = 640  # 根据错误信息修改
ORIGINAL_VIDEO_HEIGHT_EXPECTED_BY_SERVER = 480  # 根据错误信息修改

# MuJoCo模型路径 - 需要根据你的实际模型文件调整
SO100_XML_PATH = "/root/autodl-tmp/gr00t/SO-ARM100/Simulation/URDF_SO100/SO_5DOF_ARM100_05d.SLDASM/urdf/SO_5DOF_ARM100_05d.SLDASM.xml"

# --- 输出和录制配置 ---
OUTPUT_DATA_DIR = "./mujoco_gr00t_output"  # 数据和视频保存的根目录
VIDEO_FILENAME = "so100_gr00t_mujoco_sim.mp4"
VIDEO_FPS = 15  # 录制视频的帧率 (可以低于控制频率)
RECORD_VIDEO = True  # 是否录制视频

# --- 仿真参数 ---
CONTROL_FREQUENCY = 10  # Hz, GR00T 推理频率
MAX_SIMULATION_DURATION_SECONDS = 30  # 秒，最大仿真时长

# --- MuJoCo 初始化 ---
print("正在加载 MuJoCo 模型...")
try:
    model = mj.MjModel.from_xml_path(SO100_XML_PATH)
    data = mj.MjData(model)
    print(f"成功加载MuJoCo模型: {SO100_XML_PATH}")
except Exception as e:
    print(f"加载 MuJoCo 模型失败: {e}")
    exit()

# 获取可控制的关节信息
actuated_joint_indices = []
actuated_joint_names = []
for i in range(model.nu):  # model.nu是执行器数量
    actuator_id = i
    joint_id = model.actuator_trnid[actuator_id, 0]  # 执行器对应的关节ID
    if joint_id >= 0:  # 有效的关节ID
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        actuated_joint_indices.append(joint_id)
        actuated_joint_names.append(joint_name if joint_name else f"joint_{joint_id}")

num_dof = len(actuated_joint_indices)
print(f"So100 机器人可动自由度数量 (DoF): {num_dof}")
print(f"So100 可动关节名称: {actuated_joint_names}")
print(f"执行器数量: {model.nu}")

# 初始化渲染器
renderer = mj.Renderer(model, height=ORIGINAL_VIDEO_HEIGHT_EXPECTED_BY_SERVER, width=ORIGINAL_VIDEO_WIDTH_EXPECTED_BY_SERVER)

# --- 摄像头设置 ---
# 在MuJoCo中，我们使用scene来设置摄像头视角
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# 设置摄像头参数
cam = mj.MjvCamera()
cam.type = mj.mjtCamera.mjCAMERA_FREE
cam.lookat = np.array([0, 0, 0.25])  # 看向的目标点
cam.distance = 1.5  # 距离
cam.azimuth = 90  # 方位角
cam.elevation = -20  # 俯仰角


# --- MuJoCoGR00TBridge 类定义 ---
class MuJoCoGR00TBridge:
    def __init__(self, modality_json_path: str, mj_model, mj_data, 
                 num_dof_robot: int, robot_joint_indices: List[int], robot_joint_names: List[str]):
        self.model = mj_model
        self.data = mj_data
        self.num_dof = num_dof_robot
        self.robot_joint_indices = robot_joint_indices
        self.robot_joint_names = robot_joint_names

        self.current_prompt = ["拿起那个物体"]

        # 1. 加载 modality.json
        try:
            with open(modality_json_path, 'r') as f:
                self.modality_spec = json.load(f)
            print(f"成功从以下路径加载 modality spec: {modality_json_path}")
        except FileNotFoundError:
            print(f"错误: Modality JSON 文件未找到: {modality_json_path}")
            raise
        except json.JSONDecodeError:
            print(f"错误: 解析 Modality JSON 文件失败: {modality_json_path}")
            raise

        # 2. 初始化 GR00T 策略客户端
        print(f"正在连接到 GR00T 策略服务器 {GR00T_SERVER_HOST}:{GR00T_SERVER_PORT}...")
        self.policy_client = RobotInferenceClient(host=GR00T_SERVER_HOST, port=GR00T_SERVER_PORT)
        print("已连接到 GR00T 策略服务器。")

        # 3. 从服务器获取顶层 ModalityConfig
        try:
            print("正在从服务器获取 ModalityConfig...")
            server_modality_configs_raw: Dict[str, Any] = self.policy_client.get_modality_config()
            self.server_modality_configs: Dict[str, ModalityConfig] = {}
            print("成功从服务器获取 ModalityConfig:")
            for m_name, m_config_dict in server_modality_configs_raw.items():
                if isinstance(m_config_dict, dict):
                     m_config = ModalityConfig.model_validate(m_config_dict)
                elif isinstance(m_config_dict, ModalityConfig):
                     m_config = m_config_dict
                else:
                    raise TypeError(f"从服务器获取的 ModalityConfig[{m_name}] 类型未知: {type(m_config_dict)}")
                print(f"  {m_name}: keys={m_config.modality_keys}, delta_indices={m_config.delta_indices}")
                self.server_modality_configs[m_name] = m_config
        except Exception as e:
            print(f"错误: 从服务器获取 ModalityConfig 失败: {e}")
            raise

        # 4. 根据服务器的 ModalityConfig 和本地的 modality.json 确定键名
        self.video_obs_key = self._determine_obs_key("video")
        self.language_prompt_key_in_obs = self._determine_obs_key("language", LANGUAGE_PROMPT_KEY_IN_OBS)

        # 5. 预处理 modality_spec (获取状态和动作的有序键列表)
        self.ordered_state_keys_from_json = self._get_ordered_keys_from_json(self.modality_spec.get("state", {}))
        self.ordered_action_keys_from_json = self._get_ordered_keys_from_json(self.modality_spec.get("action", {}))
        print(f"从 modality.json 解析的状态键顺序: {self.ordered_state_keys_from_json}")
        print(f"从 modality.json 解析的动作键顺序: {self.ordered_action_keys_from_json}")

        # 6. 构建DOF映射
        self.state_key_to_mujoco_indices: Dict[str, List[int]] = {}
        self.action_key_to_mujoco_indices: Dict[str, List[int]] = {}
        self._build_dof_mappings()

    def _validate_observation_types(self, obs_dict):
        """验证观测数据的类型是否符合GR00T的期望"""
        for key, value in obs_dict.items():
            if key == self.video_obs_key:
                if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
                    print(f"警告: 图像数据类型不正确: {type(value)}, dtype={getattr(value, 'dtype', None)}")
                    # 确保是uint8类型的ndarray
                    obs_dict[key] = np.array(value, dtype=np.uint8)
            elif key == "observation.state":
                if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.floating):
                    print(f"警告: 状态数据类型不正确: {type(value)}, dtype={getattr(value, 'dtype', None)}")
                    # 确保是float32类型的ndarray
                    obs_dict[key] = np.array(value, dtype=np.float32)
            elif key == self.language_prompt_key_in_obs:
                if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    print(f"警告: 语言提示数据类型不正确: {type(value)}")
                    # 确保是字符串列表
                    if isinstance(value, str):
                        obs_dict[key] = [value]
                    else:
                        obs_dict[key] = [str(item) for item in value] if hasattr(value, '__iter__') else [str(value)]
        return obs_dict

    def _determine_obs_key(self, modality_name: str, default_key: str = None) -> str:
        if modality_name in self.server_modality_configs and self.server_modality_configs[modality_name].modality_keys:
            key = self.server_modality_configs[modality_name].modality_keys[0]
            return key
        elif default_key:
            print(f"警告: 服务器未提供 '{modality_name}' 键名，使用默认值: '{default_key}'")
            return default_key
        else:
            if modality_name == "video" and "video" in self.modality_spec and self.modality_spec["video"]:
                first_video_entry_key = list(self.modality_spec["video"].keys())[0]
                key = self.modality_spec["video"][first_video_entry_key].get("original_key", f"observation.images.{first_video_entry_key}")
                print(f"警告: 服务器未提供 '{modality_name}' 键名，从 modality.json 推断为: '{key}'")
                return key
            raise ValueError(f"无法确定 '{modality_name}' 的观测键名。")

    def _get_ordered_keys_from_json(self, spec_dict: Dict) -> List[str]:
        if not spec_dict: return []
        if not all("start" in v for v in spec_dict.values()):
            print(f"警告: modality.json 的 spec 字典 {list(spec_dict.keys())} 中的某些条目缺少 'start' 键。")
            return list(spec_dict.keys())
        return sorted(spec_dict.keys(), key=lambda k: spec_dict[k]["start"])

    def _build_dof_mappings(self):
        # 构建状态映射
        mujoco_dof_counter = 0
        for state_subkey in self.ordered_state_keys_from_json:
            spec = self.modality_spec["state"][state_subkey]
            dim_of_part_in_json = spec["end"] - spec["start"]
            indices_for_this_part = self.robot_joint_indices[mujoco_dof_counter : mujoco_dof_counter + dim_of_part_in_json]
            self.state_key_to_mujoco_indices[state_subkey] = indices_for_this_part
            mujoco_dof_counter += dim_of_part_in_json
        if mujoco_dof_counter != self.num_dof and self.ordered_state_keys_from_json:
            print(f"警告: 状态映射后，MuJoCo DOF数量 ({mujoco_dof_counter}) 与机器人总DOF ({self.num_dof}) 不匹配。")

        # 构建动作映射
        mujoco_dof_counter = 0
        for action_subkey in self.ordered_action_keys_from_json:
            spec = self.modality_spec["action"][action_subkey]
            dim_of_part_in_json = spec["end"] - spec["start"]
            indices_for_this_part = self.robot_joint_indices[mujoco_dof_counter : mujoco_dof_counter + dim_of_part_in_json]
            self.action_key_to_mujoco_indices[action_subkey] = indices_for_this_part
            mujoco_dof_counter += dim_of_part_in_json
        if mujoco_dof_counter != self.num_dof and self.ordered_action_keys_from_json:
            print(f"警告: 动作映射后，MuJoCo DOF数量 ({mujoco_dof_counter}) 与机器人总DOF ({self.num_dof}) 不匹配。")

    def get_current_image_observation(self) -> np.ndarray:
        """使用MuJoCo渲染器获取当前图像观测"""
        # 更新场景
        mj.mjv_updateScene(self.model, self.data, mj.MjvOption(), None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        
        # 渲染图像
        renderer.update_scene(self.data, camera=cam)
        rgb_array = renderer.render()
        
        # 确保uint8类型和正确的维度 [H, W, C]
        rgb_array = rgb_array.astype(np.uint8)
        
        return rgb_array

    def _get_mujoco_observations_dict(self) -> Dict[str, Any]:
        """根据官方示例格式获取观测数据"""
        # 获取图像并添加批次维度
        rgb_array_rgb = self.get_current_image_observation()
        rgb_array_rgb_4d = np.expand_dims(rgb_array_rgb, axis=0)  # [1, H, W, C]
        
        # 获取关节位置 - 从qpos中提取对应关节的位置
        joint_positions = np.zeros(self.num_dof, dtype=np.float64)
        for i, joint_idx in enumerate(self.robot_joint_indices):
            if joint_idx < len(self.data.qpos):
                joint_positions[i] = self.data.qpos[joint_idx]
        
        # 分离关节状态为arm和gripper部分，参考官方示例
        # 假设最后一个关节是夹爪
        arm_state = joint_positions[:-1]
        gripper_state = joint_positions[-1:]
        
        # 添加批次维度
        arm_state_2d = np.expand_dims(arm_state, axis=0)  # [1, arm_dim]
        gripper_state_2d = np.expand_dims(gripper_state, axis=0)  # [1, 1]
        
        # 确保语言提示是简单字符串列表，不是嵌套列表
        if isinstance(self.current_prompt, list) and len(self.current_prompt) > 0 and isinstance(self.current_prompt[0], list):
            # 扁平化嵌套列表
            lang_prompt = self.current_prompt[0]
        else:
            lang_prompt = self.current_prompt
        
        # 构建观测字典，完全匹配官方示例格式
        raw_obs_dict = {
            "video.webcam": rgb_array_rgb_4d,
            "state.single_arm": arm_state_2d,
            "state.gripper": gripper_state_2d,
            "annotation.human.task_description": lang_prompt  # 使用正确的格式
        }
        
        return raw_obs_dict

    def _preprocess_for_gr00t(self, observation_dict: Dict[str, Any]) -> Dict[str, Any]:
        """确保所有数据格式符合GR00T期望"""
        processed_dict = {}
        
        # 处理图像
        if self.video_obs_key in observation_dict:
            img = observation_dict[self.video_obs_key]
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            if len(img.shape) == 3 and img.shape[2] >= 3:
                img = img[:, :, :3]
            processed_dict[self.video_obs_key] = img
        
        # 处理状态 - 关键是确保"state"字段是3D格式
        if "observation.state" in observation_dict:
            obs_state = observation_dict["observation.state"]
            if not isinstance(obs_state, np.ndarray):
                obs_state = np.array(obs_state, dtype=np.float32)
            if obs_state.dtype != np.float32:
                obs_state = obs_state.astype(np.float32)
            
            # 原始observation.state保持不变，可能是1D数组
            processed_dict["observation.state"] = obs_state
            
            # 为state字段创建3D格式 [batch, seq_len, features]
            state_3d = obs_state
            if len(state_3d.shape) == 1:  # [features]
                state_3d = np.expand_dims(np.expand_dims(state_3d, axis=0), axis=0)  # [1, 1, features]
            elif len(state_3d.shape) == 2:  # [batch, features]
                state_3d = np.expand_dims(state_3d, axis=1)  # [batch, 1, features]
            
            # 确保是3D
            if len(state_3d.shape) != 3:
                raise ValueError(f"无法将状态转换为3D格式: {state_3d.shape}")
            
            processed_dict["state"] = state_3d
        
        # 处理语言提示
        if self.language_prompt_key_in_obs in observation_dict:
            prompt = observation_dict[self.language_prompt_key_in_obs]
            if isinstance(prompt, str):
                prompt = [prompt]
            elif not isinstance(prompt, list):
                prompt = [str(prompt)]
            processed_dict[self.language_prompt_key_in_obs] = prompt
        
        # 处理embodiment_id
        processed_dict["embodiment_id"] = np.array([0], dtype=np.int64)
        
        return processed_dict

    def get_gr00t_action(self) -> Dict[str, Any]:
        try:
            # 获取按照官方格式构建的观测
            obs_dict = self._get_mujoco_observations_dict()
            
            # 打印观测信息进行调试
            print("\n--- 发送给GR00T的观测数据 ---")
            for key, value in obs_dict.items():
                if isinstance(value, np.ndarray):
                    print(f"键: {key}, 类型: {type(value)}, 形状: {value.shape}, dtype: {value.dtype}")
                else:
                    print(f"键: {key}, 类型: {type(value)}, 值: {value}")
            
            # 调用GR00T API
            raw_action_chunk = self.policy_client.get_action(obs_dict)
            
            return raw_action_chunk
        except Exception as e:
            print(f"从GR00T获取动作时出错: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def apply_action_to_robot(self, gr00t_action_chunk: Dict[str, Any]) -> np.ndarray:
        """应用GR00T返回的动作到机器人"""
        # 初始化目标关节位置数组
        target_dof_positions_np = np.zeros(self.num_dof, dtype=np.float32)
        
        # 提取动作数据
        if "action.single_arm" in gr00t_action_chunk and "action.gripper" in gr00t_action_chunk:
            # 检查数据类型和维度
            arm_action = gr00t_action_chunk["action.single_arm"][0]
            gripper_action = gr00t_action_chunk["action.gripper"][0]
            
            # 打印详细信息以便调试
            print(f"arm_action 类型: {type(arm_action)}, 形状: {getattr(arm_action, 'shape', '标量')}")
            print(f"gripper_action 类型: {type(gripper_action)}, 形状: {getattr(gripper_action, 'shape', '标量')}")
            
            # 确保arm_action是一维数组
            if not isinstance(arm_action, np.ndarray):
                arm_action = np.array([arm_action])
            elif len(arm_action.shape) == 0:  # 标量
                arm_action = np.array([arm_action])
            
            # 确保gripper_action是一维数组
            if not isinstance(gripper_action, np.ndarray):
                gripper_action = np.array([gripper_action])
            elif len(getattr(gripper_action, 'shape', ())) == 0:  # 标量
                gripper_action = np.array([gripper_action])
            
            # 拼接动作数据
            print(f"处理后: arm_action形状: {arm_action.shape}, gripper_action形状: {gripper_action.shape}")
            full_action = np.concatenate([arm_action, gripper_action])
            
            # 检查尺寸是否匹配
            if len(full_action) == self.num_dof:
                target_dof_positions_np = full_action
            else:
                print(f"警告: 动作维度不匹配，期望{self.num_dof}，实际{len(full_action)}")
                print(f"full_action: {full_action}")
        else:
            # 打印可用的键，用于调试
            print(f"警告: 未找到预期的动作键，可用键: {list(gr00t_action_chunk.keys())}")
        
        # 应用目标位置到机器人关节 - 使用MuJoCo的控制接口
        # 在MuJoCo中，我们直接设置ctrl数组来控制执行器
        for i in range(min(len(target_dof_positions_np), self.model.nu)):
            self.data.ctrl[i] = target_dof_positions_np[i]
        
        return target_dof_positions_np

    def set_prompt(self, prompt_text: str):
        self.current_prompt = [prompt_text]  # 始终以列表形式存储
        print(f"提示已设置为: '{prompt_text}'")


# --- 主仿真循环 ---
if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    video_path = os.path.join(OUTPUT_DATA_DIR, VIDEO_FILENAME)
    observed_states_path = os.path.join(OUTPUT_DATA_DIR, "observed_joint_states.npy")
    groot_actions_path = os.path.join(OUTPUT_DATA_DIR, "groot_raw_actions.jsonl") # JSON Lines for dicts
    applied_targets_path = os.path.join(OUTPUT_DATA_DIR, "applied_joint_targets.npy")
    prompt_path = os.path.join(OUTPUT_DATA_DIR, "prompt.txt")

    # 实例化桥接器
    bridge = MuJoCoGR00TBridge(MODALITY_JSON_PATH, model, data, num_dof, actuated_joint_indices, actuated_joint_names)
    
    # 用户输入提示
    user_prompt = input("请输入您对机器人的指令 (例如 '拿起红色的方块'): ")
    bridge.set_prompt(user_prompt)
    with open(prompt_path, 'w') as f:
        f.write(user_prompt + "\n")
    print(f"用户提示已保存到: {prompt_path}")

    # 初始化视频录制器 (如果需要)
    video_writer = None

    if RECORD_VIDEO:
        try:
            print(f"尝试初始化视频写入器: path='{video_path}', fps={VIDEO_FPS}")
            video_writer = imageio.get_writer(
                video_path,
                fps=VIDEO_FPS,
                format='FFMPEG',  # 明确指定使用 FFmpeg 格式容器
                codec='libx264',  # 明确指定使用 H.264 编解码器
                pixelformat='yuv420p', # 一个常见的H.264兼容像素格式
                output_params=['-pix_fmt', 'yuv420p']
            )
            print(f"视频将保存到: {video_path}")
        except Exception as e:
            print(f"无法初始化视频录制器: {e}. 将不录制视频。")
            RECORD_VIDEO = False

    # 初始化数据保存列表
    all_observed_states = []
    all_applied_targets = []

    # 仿真循环
    frame_count = 0
    sim_step_count = 0
    control_interval_steps = int((1.0 / CONTROL_FREQUENCY) / model.opt.timestep)
    video_render_interval_steps = int((1.0 / VIDEO_FPS) / model.opt.timestep)
    
    print(f"仿真将运行约 {MAX_SIMULATION_DURATION_SECONDS} 秒。")
    print(f"MuJoCo时间步长: {model.opt.timestep}")
    print(f"控制频率: {CONTROL_FREQUENCY} Hz (每 {control_interval_steps} 仿真步一次控制)")
    if RECORD_VIDEO:
        print(f"视频录制频率: {VIDEO_FPS} Hz (每 {video_render_interval_steps} 仿真步一次渲染)")

    try:
        start_sim_real_time = time.time()
        while (sim_step_count * model.opt.timestep) < MAX_SIMULATION_DURATION_SECONDS:
            # 控制逻辑
            if sim_step_count % control_interval_steps == 0:
                current_sim_time_s = sim_step_count * model.opt.timestep
                print(f"\n仿真时间: {current_sim_time_s:.2f}s / {MAX_SIMULATION_DURATION_SECONDS}s, 控制步 {frame_count+1}")
                
                # 1. 获取当前观测状态并保存
                observed_js = np.zeros(num_dof)
                for i, joint_idx in enumerate(actuated_joint_indices):
                    if joint_idx < len(data.qpos):
                        observed_js[i] = data.qpos[joint_idx]
                all_observed_states.append(observed_js)

                # 2. 从GR00T获取动作
                try:
                    action_chunk = bridge.get_gr00t_action()
                    
                    # 保存GR00T返回的原始动作字典
                    action_to_save = {k: v.tolist() for k, v in action_chunk.items()} # ndarray to list for json
                    with open(groot_actions_path, 'a') as f_actions: # 追加模式
                        json.dump({"sim_time": current_sim_time_s, "step": frame_count, "action": action_to_save}, f_actions)
                        f_actions.write('\n')

                    # 3. 应用动作到机器人并获取目标位置用于保存
                    if action_chunk:
                        applied_targets_np = bridge.apply_action_to_robot(action_chunk)
                        all_applied_targets.append(applied_targets_np)
                    else:
                        print("警告: 未从 GR00T 收到有效动作。维持上一目标或零目标。")
                        all_applied_targets.append(np.zeros_like(observed_js))

                except Exception as e:
                    print(f"与 GR00T 服务器交互或应用动作时发生错误: {e}")
                    import traceback
                    traceback.print_exc()
                    # 发生错误时，也记录一个全零的目标
                    all_applied_targets.append(np.zeros_like(observed_js))
                
                frame_count += 1

            # 视频帧录制逻辑
            if RECORD_VIDEO and video_writer and (sim_step_count % video_render_interval_steps == 0):
                try:
                    rgb_frame = bridge.get_current_image_observation()
                    video_writer.append_data(rgb_frame)
                except Exception as e:
                    print(f"录制视频帧时出错: {e}")
            
            # 步进MuJoCo仿真
            mj.mj_step(model, data)

            if sim_step_count % 10 == 0:  # 每10步打印一次
                # 打印机器人状态信息
                print(f"仿真时间: {data.time:.3f}s")
                
                # 可选：打印关节状态
                joint_positions = [round(data.qpos[idx], 4) for idx in actuated_joint_indices]
                print(f"机器人关节位置: {joint_positions}")

            sim_step_count += 1

    except KeyboardInterrupt:
        print("用户中断仿真。")
    finally:
        # 清理和保存
        if RECORD_VIDEO and video_writer:
            try:
                video_writer.close()
                print(f"视频成功保存到: {video_path}")
            except Exception as e:
                print(f"关闭视频写入器时出错: {e}")
        
        np.save(observed_states_path, np.array(all_observed_states))
        print(f"观测到的关节状态已保存到: {observed_states_path} (形状: {np.array(all_observed_states).shape})")
        
        np.save(applied_targets_path, np.array(all_applied_targets))
        print(f"应用的关节目标已保存到: {applied_targets_path} (形状: {np.array(all_applied_targets).shape})")

        print(f"GR00T原始动作已保存到: {groot_actions_path}")

        print("MuJoCo仿真已结束。")
        end_sim_real_time = time.time()
        print(f"总实际运行时间: {end_sim_real_time - start_sim_real_time:.2f} 秒")
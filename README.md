<div align="center">
<img src="./assets/RoboBrain-X0-logo.png" width="500"/>
</div>

# RoboBrain-X0: A Unified Cross-Embodiment Vision-Language-Action Model for Token Reasoning and Action Generation. 


<p align="center">
        </a>&nbsp&nbsp⭐️ <a href="https://superrobobrain.github.io/">Project</a></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/cmyopu/RoboBrain-X0-Preview">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://www.modelscope.cn/models/BAAI/RoboBrain2.0-7B/files/">ModelScope</a>&nbsp&nbsp 

<!-- </p>
<p align="center">
        </a>&nbsp&nbsp📖 <a href="https://wisemodel.cn/models/BAAI/RoboBrain2.0-7B">Wisemodel</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2507.02029 ">Technical Report</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="./assets/wechat.jpeg">WeChat & RedNote</a>
</p> -->

</p>

<p align="center">
</a>&nbsp&nbsp🚀 <a href="https://github.com/FlagOpen/RoboBrain2.0">RoboBrain 2.0</a>: See Better. Think Harder. Do Smarter. 
</p>
<p align="center">
        </a>&nbsp&nbsp🎯 <a href="https://flagopen.github.io/RoboOS/">RoboOS</a>: An Efficient Open-Source Multi-Robot Coordination System for RoboBrain.
</p>
<p align="center">
</a>&nbsp&nbsp⭐️ <a href="https://github.com/tanhuajie/Reason-RFT">Reason-RFT</a>: Core Post-Training Strategy for Embodied Visual Reasoning in RoboBrain2.0.
</p>
<p align="center">
</a>&nbsp&nbsp🌍 <a href="https://github.com/FlagOpen/RoboBrain">RoboBrain 1.0</a>: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete.
</p>

💬 ***If you have any questions, feel free to contact us via WeChat or RedNote.***

<div align="center">
<img src="./assets/wechat.jpeg", width=750 />
</div>

## 🔥 Overview
We are thrilled to introduce **RoboBrain-X0**, a groundbreaking cross-ontology foundation model designed to overcome the limitations of single-robot systems in heterogeneous ontology transfer. By leveraging End-Effector pose representation in SE(3) task space, coupled with a Unified Action Vocabulary (UAV) and action tokenizer, RoboBrain-X0 achieves efficient **zero-shot** generalization and **complex task decomposition**. Its Grouped Residual Quantizer (GRVQ) maps continuous control sequences from diverse degrees of freedom and mechanical structures to a shared discrete action primitive space, ensuring semantic consistency and transferability across ontologies such as **AgileX**, **R1-Lite dual-arm wheeled robots**, and **Franka single-arm systems**. Through ontology-conditioned diverse prompting, the model supports flexible decoding from multi-view RGB-D inputs to specific executions, significantly reducing training and inference overhead. RoboBrain-X0 delivers state-of-the-art performance in embodied reasoning tasks, laying a robust foundation for developing versatile, real-world robotic agents and advancing embodied intelligence research.


<div style="display: flex; justify-content: center;">
  <img src="./assets/Eval_Result.png" style="width: 120%; height: 150%" />
</div>



## 🗞️ News
- **`2025-09-29`**: 🔥 [**RoboBrain-X0-preview**](https://huggingface.co/BAAI/RoboBrain-X0-Preview) was released.


## 📆 Todo
- [x] Release model checkpoint for RoboBrain-X0-Preview
- [x] Release quick inference example for RoboBrain-X0
- [x] Release training and evaluation codes for RoboBrain-X0


## 🚀 Features
**RoboBrain-X0** supports the unified modeling of heterogeneous ontologies and offers **zero-shot** generalization and complex task decomposition capabilities. Building on RoboBrain's multimodal foundation, RoboBrain-X0 further integrates real-world robot motion data based on **RoboBrain-2.0** data. By unifying vision, language, and motion modeling, it achieves cross-ontology generalization and adaptation, providing integrated capabilities from perception to execution.

<div align="center">
<img src="./assets/demo_01.png", width=750 />
</div>
<div align="center">
<img src="./assets/sat.png", width=750 />
</div>

## ⭐️ Architecture
This model includes **RoboBrain 2.0** and **OmniSAT** (action tokenizer). Based on RoboBrain 2.0, the model is trained on a large amount of real-world robotics data and embodied reasoning data, enabling it to possess general robotic manipulation capabilities. The action token sequences output by the model are converted into underlying robot control signals through our proprietary action tokenizer. Model details are as follows:
- **Multimodal Input**: The model accepts single-image, multi-image, and text input (covering pointing task scenarios, object maneuverability scenarios, trajectory scenarios, and subtask execution scenarios), and produces outputs of varying dimensions based on the input scenarios.
- **Action Generation and Execution**: After model processing, OmniSAT converts these into multi-degree-of-freedom (DoF) action sequences, ultimately driving the robot to complete the operation.

<div align="center">
<img src="./assets/RoboBrain-X0.jpg" width="750" />      
</div>


## 🤗 Model Zoo


| Models               | Checkpoint                                                     | Description                                                | 
|----------------------|----------------------------------------------------------------|------------------------------------------------------------|
| RoboBrain-X0-preview     | [🤗 BAAI/RoboBrain-X0-Preview(9.30)](https://huggingface.co/cmyopu/RoboBrain-X0-Preview)  | preview version of the RoboBrain-X0                  | 
| RoboBrain-X0-FlagOS     | [🤗 FlagRelease/RoboBrain-X0-FlagOS](https://huggingface.co/FlagRelease/RoboBrain-X0-FlagOS)  | multi-chip version of the RoboBrain-X0                  | 
| RoboBrain-X0-FlagOS     | [🤗 BAAI/RoboBrain-X0-Dataset(9.30)](https://huggingface.co/datasets/BAAI/RoboBrain-X0-Dataset)  | training dataset of RoboBrain-X0                  | 




## 🛠️ Setup

```bash
# Pull Docker Image.
docker pull ghcr.io/robobrain-roboos-robotic/robotics_pretrain_flagscale:cuda12.4.1-cudnn9.5.0-python3.12-torch2.6.0-time250928-ssh

# Run Container.
docker run -itd \
  --name robotics_pretrain \
  --privileged \
  --gpus all \
  --net=host \
  --ipc=host \
  --device=/dev/infiniband \
  --shm-size 512g \
  --ulimit memlock=-1 \
  -v /nfs/hcr/models/:/models \
  ghcr.io/robobrain-roboos-robotic/robotics_pretrain_flagscale:cuda12.4.1-cudnn9.5.0-python3.12-torch2.6.0-time250928-ssh
```

## 🤖 Training
```bash
cd /root/robotics_pretrain/flag-scale
conda activate flagscale-train

python run.py \
  --config-path ./examples/qwen2_5_vl/conf \
  --config-name train_3b_action_S6_subtask_agilex_eval5_demo \
  action=run
```

## 🔍 Evaluation
The evaluation on real robots adopts a "server-client" architecture, with a PC acting as a data transfer hub: the model is deployed on a remote server (responsible for inference), while the client (running on the robot side) sends the robot's state to the server. After the model performs inference, it returns the generated action results to the client, which then controls the robot to execute these actions.

### Real Robots Evaluation Architecture Overview

The system operates on a simple yet effective server-client model:

1.  **Server (Inference Side)**: A remote machine (usually with a powerful GPU) where the trained model is deployed.
    *   It receives the robot's state and images.
    *   It performs inference to decide on the next sequence of actions.
    *   It sends the generated actions back to the client in the HTTP response.
2.  **Client (Robot Side)**: The Python script running on the robot's control computer.
    *   It continuously captures the robot's current state (e.g., end-effector pose, joint angles) and images from its cameras.
    *   It packages this observation data into a JSON payload.
    *   It sends the data to the remote inference server via an HTTP POST request.

This decoupling allows for running complex, resource-intensive models without needing high-end computing hardware directly connected to the robot.

```
┌──────────────────┐      ┌───────────────────────────┐      ┌────────────────────┐
│                  │      │      Control PC (Client)  │      │                    │
│   Real Robot     ├─────►│ 1. Get State & Images     ├─────►│  Inference Server  │
│                  │      │ 2. Send Request           │      │  (Model Deployed)  │
│                  │      │                           │      │                    │
│                  │◄─────┤ 4. Execute Actions        │◄─────┤ 3. Return Actions  │
└──────────────────┘      └───────────────────────────┘      └────────────────────┘
```
### How to Run a Server

Each server can be configured and run independently. Here are the general instructions.

1.  **Choose a server script** (e.g., `serve_franka.py`).
2.  **Configure the mode**: Open the script and set the `SUBTASK_MODE` variable:
    -   `SUBTASK_MODE = False` for standard action generation.
    -   `SUBTASK_MODE = True` for subtask + action generation.
3.  **Verify paths**: Ensure that `MODEL_PATH` and `STATS_PATH` (if applicable) point to the correct files for your chosen mode.
4.  **Start the server**:
    ```bash
    # Example for Franka server
    python serve_franka.py
    ```
### Examples   
Inference code examples are provided for four robot platforms: A2D, Franka, Agilex, and R1lite (ROS1 and ROS2 version).
-   **Agilex**
-   **Franka**
-   **R1-Lite**
-   **A2D**

## 📑 Citation
If you find this project useful, welcome to cite us.
```bib
@article{RoboBrain1.0,
    title={Robobrain: A unified brain model for robotic manipulation from abstract to concrete},
    author={Ji, Yuheng and Tan, Huajie and Shi, Jiayu and Hao, Xiaoshuai and Zhang, Yuan and Zhang, Hengyuan and Wang, Pengwei and Zhao, Mengdi and Mu, Yao and An, Pengju and others},
    journal={arXiv preprint arXiv:2502.21257},
    year={2025}
}

@article{RoboBrain2.0TechnicalReport,
    title={RoboBrain 2.0 Technical Report},
    author={BAAI RoboBrain Team},
    journal={arXiv preprint arXiv:2507.02029},
    year={2025}
}

@article{zhou2025roborefer,
    title={RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics},
    author={Zhou, Enshen and An, Jingkun and Chi, Cheng and Han, Yi and Rong, Shanyu and Zhang, Chi and Wang, Pengwei and Wang, Zhongyuan and Huang, Tiejun and Sheng, Lu and others},
    journal={arXiv preprint arXiv:2506.04308},
    year={2025}
}

@article{Reason-RFT,
    title={Reason-rft: Reinforcement fine-tuning for visual reasoning},
    author={Tan, Huajie and Ji, Yuheng and Hao, Xiaoshuai and Lin, Minglan and Wang, Pengwei and Wang, Zhongyuan and Zhang, Shanghang},
    journal={arXiv preprint arXiv:2503.20752},
    year={2025}
}

```





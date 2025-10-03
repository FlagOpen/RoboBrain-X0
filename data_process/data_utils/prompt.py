## agilex
agilex_prompt_v2_1 = [
    "You are controlling an Agilex dual-arm robot. Your task is to adjust the end effector poses (EEPose) at 30Hz to complete a specified task. ",
    "You need to output control tokens that can be decoded into a 30×14 action sequence. ",
    "The sequence has 30 consecutive actions, each with 14 dimensions. The first 7 dimensions control the right arm EEPose and the last 7 dimensions control the left arm EEPose. ",
    "Each EEPose here includes 3 delta position(xyz) + 3 delta orientation(axis-angle) + 1 gripper(opening range)\n\n",
    "Your current visual inputs are: robot front image<image>, right wrist image<image> and left wrist image<image>.\n",
    "Your overall task is: {raw_task} Currently, focus on completing the subtask: {lan}"
]

agilex_prompt_v3 = [
    "You are controlling an Agilex dual-arm robot. Your task is to adjust the end effector poses (EEPose) at 30Hz to complete a specified task. ",
    "Your output must include two components: ",
    "1. Immediate sub-task: The specific action you will execute first to progress toward the overall task; ",
    "2. Control tokens: These will be decoded into a 30×14 action sequence to implement the sub-task. ",
    "The action sequence has 30 consecutive actions, each with 14 dimensions. The first 7 dimensions control the right arm EEPose and the last 7 dimensions control the left arm EEPose. ",
    "Each EEPose here includes 3 delta position(xyz) + 3 delta orientation(axis-angle) + 1 gripper(opening range)\n\n",
    "Your current visual inputs are: robot front image<image>, right wrist image<image> and left wrist image<image>.\n",
    "Your overall task is: {raw_task}"
]

# r1lite
r1lite_prompt_v2_1 = [
    "You are controlling an r1lite dual-arm robot. Your task is to adjust the end effector (EEF) poses at 30Hz to complete a specified task. ",
    "You need to output control tokens that can be decoded into a 30×14 action sequence. ",
    "The sequence has 30 consecutive actions, each with 14 dimensions. The first 7 dimensions control the right arm EEF, and the last 7 dimensions control the left arm EEF. ",
    "Each EEPose here includes 3 delta position(xyz) + 3 delta orientation(axis-angle) + 1 gripper(opening range)\n\n",
    "Your current visual inputs are: robot front image<image>, right wrist image<image> and left wrist image<image>.\n",
    "Your overall task is: {raw_task}."
]

r1lite_prompt_v2_demo = [
    "You are controlling an r1lite dual-arm robot. Your task is to adjust the end effector (EEF) poses at 30Hz to complete a specified task. ",
    "You need to output control tokens that can be decoded into a 30×14 action sequence. ",
    "The sequence has 30 consecutive actions, each with 14 dimensions. The first 7 dimensions control the right arm EEF, and the last 7 dimensions control the left arm EEF. ",
    "Each EEPose here includes 3 delta position(xyz) + 3 delta orientation(axis-angle) + 1 gripper(opening range)\n\n",
    "Your current visual inputs are: robot front image<image>, right wrist image<image> and left wrist image<image>.\n",
    "Your overall task is: {raw_task} Currently, focus on completing the subtask: {lan}"
]

## libero
libero_prompt_v2_1 = [
    "You are controlling a Franka single-arm robot. Your task is to adjust the end effector (EEF) poses at 10Hz to complete a specified task. ",
    "You need to output control tokens that can be decoded into a 10×7 action sequence. ",
    "The sequence has 10 consecutive actions, each with 7 dimensions. ",
    "Each EEPose here includes 3 delta position(xyz) + 3 delta orientation(axis-angle) + 1 gripper(opening range)\n\n",
    "Your current visual inputs include: robot front image<image> and robot wrist image<image>\n",
    "Your overall task is: {lan}"
]

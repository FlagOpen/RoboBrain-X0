import json
import random
import numpy as np
import os
import jsonlines
from collections import defaultdict
from PIL import Image
import sys
sys.path.append("data_process/data_utils")
from prompt import *
import subprocess
import shlex


def data_merge(data_path, task_name):
    all_data = []
    for task_file in os.listdir(data_path):
        if task_file.split('.')[-1] == 'jsonl':
            with open(f"{data_path}/{task_file}", "r") as f:
                for line in f:
                    try:
                        json_item = json.loads(line)
                        all_data.append(json_item)
                    except:
                        continue

    print(len(all_data))
    with jsonlines.open(f'{data_path}/{task_name}_train.jsonl', 'w') as f:
        f.write_all(all_data)

def data_clean(data_path, task_name):
    rm_files = [ _ for _ in os.listdir(data_path) if _.split('.')[-1] == 'jsonl' and _ != f'{task_name}_train.jsonl']

    for rm_file in rm_files:
        rm_file_path = os.path.join(data_path, rm_file)
        subprocess.run(['rm', rm_file_path])


def change_prompt(data_path, task_name):

    if task_name in ['agilex', 'agilex_v2']:
        user_prompt = "".join(agilex_prompt_v2_1)
        data = []
        with jsonlines.open(f"{data_path}/{task_name}_train.jsonl", 'r') as f:
            for line in f:
                data.append(line)

        with open("data_process/data_utils/task_insts.json", "r") as f:
            rewrite_inst = json.load(f)

        for i in range(len(data)):
            raw_task = data[i]['raw_task']
            diverse_raw_task = rewrite_inst[raw_task][0]
            inst = data[i]['conversations'][0]['value'].split(': ')[-1]
            prompt = user_prompt.format(raw_task=diverse_raw_task.lower(), lan=inst.lower()).replace("  ", " ")
            data[i]['conversations'][0]['value'] = prompt


    elif task_name in ['libero']:
        user_prompt = "".join(libero_prompt_v2_1)
        data = []
        with jsonlines.open(f"{data_path}/{task_name}_train.jsonl", 'r') as f:
            for line in f:
                data.append(line)

        for i in range(len(data)):
            inst = data[i]['conversations'][0]['value'].split(': ')[-1].replace(' end.', '')
            prompt = user_prompt.format(lan=inst.lower())
            data[i]['conversations'][0]['value'] = prompt


    elif task_name == 'r1lite':
        user_prompt = "".join(r1lite_prompt_v2_1)
        data = []
        with jsonlines.open(f"{data_path}/{task_name}_train.jsonl", 'r') as f:
            for line in f:
                data.append(line)

        for i in range(len(data)):
            raw_task = data[i]['raw_task']
            prompt = user_prompt.format(raw_task=raw_task.lower())
            data[i]['conversations'][0]['value'] = prompt.replace('..', '.')

    elif task_name == 'r1lite_demo':
        user_prompt = "".join(r1lite_prompt_v2_demo)
        data = []
        with jsonlines.open(f"{data_path}/{task_name}_train.jsonl", 'r') as f:
            for line in f:
                data.append(line)

        for i in range(len(data)):
            raw_task = 'Arrange Items on the table and classify them into edible and inedible categories.'
            sub_task = data[i]['task']
            prompt = user_prompt.format(raw_task=raw_task.lower(), lan=sub_task.lower())
            data[i]['conversations'][0]['value'] = prompt.replace('..', '.') + '.'


    elif task_name == 'agilex_demo':
        user_prompt = "".join(agilex_prompt_v2_1)
        data = []
        with jsonlines.open(f"{data_path}/{task_name}_train.jsonl", 'r') as f:
            for line in f:
                data.append(line)

        for i in range(len(data)):
            raw_task = 'Arrange Items on the table and classify them into edible and inedible categories.'
            sub_task = data[i]['task']
            prompt = user_prompt.format(raw_task=raw_task.lower(), lan=sub_task.lower())
            data[i]['conversations'][0]['value'] = prompt.replace('..', '.') + '.'

    random.shuffle(data)
    with jsonlines.open(f"{data_path}/{task_name}_train.jsonl", 'w') as f:
        f.write_all(data)
    print(f'Final construct data: {len(data)}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="轨迹数据处理器参数")
    # 添加必要的命令行参数
    parser.add_argument("--data_path", type=str, default=f"data/data_version_name")
    parser.add_argument("--data_name", type=str, default="agilex")
    args = parser.parse_args()

    if not os.path.exists(f"{args.data_path}/{args.data_name}_train.jsonl"):
        # data merge
        data_merge(args.data_path, args.data_name)
        # remove subprocess file
        data_clean(args.data_path, args.data_name)

    if not os.path.exists(f"{args.data_path}/{args.data_name.split('_')[0]}_train_shuffle_prompt_v1.jsonl"):
        # modify prompt
        change_prompt(args.data_path, args.data_name)






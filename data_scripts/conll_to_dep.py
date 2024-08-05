import re
import os
import sys
import json
import numpy as np
from tqdm import tqdm
path = os.getcwd()
prefix = "bllip_"
postfix = "_conll"
for s in ["train_LG", "dev", "test"]:
    with open(os.path.join(path, prefix + s + postfix), 'r') as f:
        lines = f.readlines()
    
    sents = []
    sent = []
    action_list = []
    index2 = 0
    print("Processing {} sentences...".format(s))
    for line in tqdm(lines):
        line = line.strip()
        if line == "":
            if sent != []:
                sents.append(sent)
                sent = []
            else:
                sent = []
            continue
        line_split = line.split()
        if len(line_split) != 10:
            print(line_split)
            raise ValueError("line_split length is not 10")
        if line_split[1] == "left_arc" or line_split[1] == "right_arc":
            raise ValueError("line_split[1] is the same as operators")
        try:     
            parent = int(line_split[6]) - 1
            sent.append((line_split[1], parent))
        except ValueError:
            raise ValueError("parent is not int")

    print("Processing {} actions...".format(s))
    for sent in tqdm(sents):
        index2 += 1
        last_head_id = [0 for _ in range(len(sent))] # to record each word's last direct child
        for i in range(len(sent)):
            if sent[i][1] != -1:
                last_head_id[sent[i][1]] = i
        # print(last_head_id)
        stack = []
        actions = []
        actions2 = []
        index = 0
        step = 0
        length = len(sent)
        while index < length or len(stack) != 1:
            if index == length and step == len(stack) and len(stack) != 1:
                break
            if index == length and len(stack) != 1:
                step = len(stack)

            # print(index)
            # print(stack)
            # if index == length:
            #     exit()
            if len(stack) < 2 and index < length:
                stack.append(index)
                actions.append(sent[index][0])
                index += 1
            if len(stack) < 2:
                continue
            left = stack[-2]
            right = stack[-1]
            if sent[left][1] == right:
                stack[-1] = stack.pop()
                actions.append("left_arc")

            elif sent[right][1] == left:
                if last_head_id[right] >= index:
                    stack.append(index)
                    actions.append(sent[index][0])
                    index += 1
                else:
                    stack.pop()
                    actions.append("right_arc")
            else:
                if index < length:
                    stack.append(index)
                    actions.append(sent[index][0])
                    index += 1   
        if len(stack) == 1 and sent[stack[0]][1] == -1:
            actions.append("pop_root")
            action_list.append(actions)
    
    print("Writing {} actions...".format(s))
    with open(os.path.join(path, s + ".bllip_action"), 'w') as f:
        for actions in tqdm(action_list):
            f.write(' '.join(actions) + '\n')
        # else:
        #     raise ValueError("Not a valid dependency tree")

        
    
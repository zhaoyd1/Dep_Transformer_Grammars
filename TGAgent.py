import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from masking_bllip import utils as masking_utils
from masking_bllip.utils import TokenTypeRanges
from masking_bllip import masking_types as types
from masking_bllip import constants as mc
import time
from helping_utils.logger import configure_logger, get_logger


class TGAgent:
    def __init__(self, ranges: TokenTypeRanges, stack_size = -1, relative_mode = 0):
        # mode = 0, use stack relative depth
        # mode = 1, use relative linear depth
        self.ranges = ranges
        #-1 means no limit to max stack size
        self.stack_size = stack_size
        self.mode = relative_mode
        # stack_items: (subword_position, subword_id)
        self.stack = []
        self.length = 0
        self.composed_position = []
        # start of subwords, i.e '_xxx' 's subword_stack_depth  
        self.stack_word_head = []
        # compose doubled, 0->compose, 1->stack
        self.compose_count = 0
    
    def step(self, new_token, token_p, cur_pos, total_length, finished=False, pre_padding=0):
        # there may be padding in the sentence before start position 
        ttype = self.ranges.token_type_from_token(torch.tensor([token_p]), use_pytorch=True)[0]

        # total_length should be equal to cur_pos - start + 1
        masks = np.ones((total_length), dtype=np.int32)
        masks[cur_pos+1:] = 0
        # if cur_pos != self.length:
        #     mask[self.length:cur_pos] = 0
        self.length = cur_pos + 1
        token_for_next_input = [new_token, None] # changes when compose (left_arc changes to headword)

        if finished:
            relative_depth = np.zeros(total_length, dtype=np.int32)
            relative_depth[0:cur_pos+1] = np.arange(cur_pos, -1, -1, dtype=np.int32)
            if pre_padding != 0:
                pre_mask = np.zeros((pre_padding), dtype=np.int32)
                pre_depth = np.arange(cur_pos + pre_padding, cur_pos, -1, dtype=np.int32)
                masks = np.concatenate((pre_mask, masks))
                relative_depth = np.concatenate((pre_depth, relative_depth))
            
            return token_for_next_input, masks, relative_depth


        if self.mode == 0:
            if ttype == mc.STARTOFWORD or ttype == mc.SOS:
                self.push_stack_items((cur_pos, new_token), 'start')
                self.stacking_att_mask(masks)
                relative_depth = self.relative_stack_depth(cur_pos, total_length)
            
            elif ttype == mc.LEFTARC:
                if self.compose_count == 0:
                    self.compose_att_mask(masks, cur_pos)
                    relative_depth = self.relative_compose_depth(cur_pos, total_length, 'left')
                    token_for_next_input[1] = self.get_compose_headword_id('left')
                    self.set_composed_position()
                    self.pop_stack_items(2)
                    self.push_stack_items((cur_pos, token_for_next_input[1]), 'compose')
                    self.compose_count = 1
                else:
                    self.stacking_att_mask(masks)
                    relative_depth = self.relative_stack_depth(cur_pos, total_length)
                    self.composed_position.append(cur_pos)
                    token_for_next_input[1] = self.stack[-1][1]
                    self.compose_count = 0
            
            elif ttype == mc.RIGHTARC or ttype == mc.POPROOT:
                if self.compose_count == 0:
                    self.compose_att_mask(masks, cur_pos)
                    relative_depth = self.relative_compose_depth(cur_pos, total_length, 'right')
                    token_for_next_input[1] = self.get_compose_headword_id('right')
                    self.set_composed_position()
                    self.pop_stack_items(2)
                    self.push_stack_items((cur_pos, token_for_next_input[1]), 'compose')
                    self.compose_count = 1
                else:
                    self.stacking_att_mask(masks)
                    relative_depth = self.relative_stack_depth(cur_pos, total_length)
                    self.composed_position.append(cur_pos)
                    token_for_next_input[1] = self.stack[-1][1]
                    self.compose_count = 0
            
            elif ttype == mc.PAD:
                relative_depth = np.zeros(total_length, dtype=np.int32)
                relative_depth[0:cur_pos+1] = np.arange(cur_pos, -1, -1, dtype=np.int32)
            
            else:
                self.push_stack_items((cur_pos, new_token), 'subword')
                self.stacking_att_mask(masks)
                relative_depth = self.relative_stack_depth(cur_pos, total_length)

        if self.mode == 1:
            relative_depth = np.zeros(total_length, dtype=np.int32)
            relative_depth[0:cur_pos+1] = np.arange(cur_pos, -1, -1, dtype=np.int32)
        
        if pre_padding != 0:
            pre_mask = np.zeros((pre_padding), dtype=np.int32)
            pre_depth = np.arange(cur_pos + pre_padding, cur_pos, -1, dtype=np.int32)
            masks = np.concatenate((pre_mask, masks))
            relative_depth = np.concatenate((pre_depth, relative_depth))

        return token_for_next_input, masks, relative_depth


    def check_stack_size(self):
        if self.stack_size == -1:
            return True
        return len(self.stack) < self.stack_size
    
    def get_compose_headword_id(self, mode='left'):
        # last subword_id of headword
        # assert len(self.stack) >= 2
        # assert len(self.stack_word_head) >= 2
        if len(self.stack) < 2 or len(self.stack_word_head) < 2:
            return 0
        
        if mode == 'left': 
            return self.stack[-1][1]
        elif mode == 'right':
            return self.stack[self.stack_word_head[-1] - 1][1]
        
        return None

    def get_compose_left_start_pos(self):
        # left word's first subword position
        # assert len(self.stack) >= 2
        # assert len(self.stack_word_head) >= 2
        if len(self.stack) < 2 or len(self.stack_word_head) < 2:
            return 0
        
        return self.stack[self.stack_word_head[-2]][0]

    def get_compose_right_start_pos(self):
        # right word's first subword position
        # assert len(self.stack) >= 2
        # assert len(self.stack_word_head) >= 2
        if len(self.stack) < 2 or len(self.stack_word_head) < 2:
            return 0
        
        return self.stack[self.stack_word_head[-1]][0]
    
    def get_compose_left_end_pos(self):
        # left word's last subword position
        # assert len(self.stack) >= 2
        # assert len(self.stack_word_head) >= 2
        if len(self.stack) < 2 or len(self.stack_word_head) < 2:
            return 0
        
        return self.stack[self.stack_word_head[-1] - 1][0]

    def get_compose_right_end_pos(self):
        # right word's last subword position
        # assert len(self.stack) >= 2
        # assert len(self.stack_word_head) >= 2
        if len(self.stack) < 2 or len(self.stack_word_head) < 2:
            return 0
        
        return self.stack[-1][0]

    def pop_stack_items(self, number=2):
        #default to pop 2 words, compose action
        if number >= len(self.stack_word_head):
            return None
        depth = self.stack_word_head[-number]
        self.stack_word_head = self.stack_word_head[:-number]
        self.stack = self.stack[:depth]

    def push_stack_items(self, items, type='start'):
        # items: (subword_position, subword_id)
        self.stack.append(items)
        if type == 'start' or type == 'compose':
            self.stack_word_head.append(len(self.stack) - 1)
    
    def stacking_att_mask(self, mask):
        mask[self.composed_position] = 0
    
    def compose_att_mask(self, mask, cur):
        mask[:self.get_compose_left_start_pos()] = 0
        mask[self.get_compose_left_end_pos()+1:self.get_compose_right_start_pos()] = 0
        mask[self.get_compose_right_end_pos()+1:cur] = 0

    def set_composed_position(self):
        self.composed_position.extend(range(self.get_compose_left_start_pos(), self.get_compose_left_end_pos()+1))
        self.composed_position.extend(range(self.get_compose_right_start_pos(), self.get_compose_right_end_pos()+1))

    def relative_stack_depth(self, cur_pos, length):
        # stack relative depth encoding
        # stack: (subword_position, subword_id)
        # stack_word_head: (subword_stack_depth)
        # length, may contain padding
        relative_stack_depth = np.zeros(length, dtype=np.int32)
        end = self.stack[len(self.stack) - 1][0]
        for i in reversed(range(len(self.stack_word_head))):
            start = self.stack[self.stack_word_head[i]][0]
            relative_stack_depth[start:end+1] = len(self.stack_word_head) - i - 1
            if self.stack_word_head[i] > 0:
                end = self.stack[self.stack_word_head[i] - 1][0]
        
        return relative_stack_depth

    def relative_compose_depth(self, cur_pos, length, mode='left'):
        # compose relative depth encoding
        # stack: (subword_position, subword_id)
        # length, may contain padding
        relative_compose_depth = np.zeros(length, dtype=np.int32)
        if mode == 'left':
            start = self.get_compose_left_start_pos()
            end = self.get_compose_left_end_pos()
            relative_compose_depth[start:end+1] = -1
        elif mode == 'right':
            start = self.get_compose_right_start_pos()
            end = self.get_compose_right_end_pos()
            relative_compose_depth[start:end+1] = -1
        
        return relative_compose_depth

if __name__ == '__main__':
    ranges = TokenTypeRanges(1, 2, 6, 4, 100)
    agent = TGAgent(ranges, -1, 0)
    original_sent = torch.tensor([1, 6, 10, 6, 11, 12, 4, 6, 6, 6, 13, 4, 100, 100, 101, 111])
    maskrules = masking_utils.get_masking_rules(
        "stack_compose_double_closing_nt",
        sequence_length = 40,
        memory_length = 40,
        transparency_prob = 0.0,
        gather_into_new_memory=True, 
        transparency_depth_threshold=-1 
    )
    info_tuple = masking_utils.compute_token_types(
        {"inputs": original_sent[:-1], "labels": original_sent[1:]}, ranges
    )
    print(info_tuple['inputs'])
    chunks = maskrules.chunks_for_sequence(info_tuple['inputs'], info_tuple['inputs_ttypes'],
                                           info_tuple['labels'], info_tuple['labels_ttypes'])
    chunks = [types.Chunk(None, *chunk) for chunk in chunks]
    chunk = chunks[0]
    masks = chunk.attn_mask[:40, :40]
    relpos = chunk.attn_relpos[:len(masks), 20:20+len(masks)]
    src = chunk.inputs[:40]
    composed_pos = chunk.composed_position[:40]
    src = src[composed_pos]
    print(src)
    # with np.printoptions(threshold=np.inf):
    #     print(masks)
    #     print(relpos)
    # print(relpos)
    # relpos[:, 0] = 0
    # print(relpos)
    sent = [1, 6, 10]
    for i in range(len(sent)):
        next_token, mask, relative_depth = agent.step(sent[i], i, len(sent), False, 3)
        print(relative_depth)
        # if mask.all() != masks[i].all() or relative_depth.all() != relpos[i].all():  
        #     print(mask)
        #     print(relative_depth)


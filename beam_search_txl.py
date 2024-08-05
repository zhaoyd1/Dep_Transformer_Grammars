from typing import List
import torch
from dataclasses import dataclass
# from lightning.pytorch import seed_everything
import numpy as np
import numba
import numba.cuda as cuda
import argparse
import time
from helping_utils.logger import configure_logger, get_logger
from TGAgent import TGAgent
from masking_bllip.utils import TokenTypeRanges
from model_bllip_dep import TransformerGrammar
from copy import deepcopy
import json
import re
import os
from tqdm import tqdm
import sentencepiece as spm
import math

class BoringLM:
    # a random bi-gram lm
    def __init__(self, batch_size, vocab_size, _num_action):
        # batch x step(i - 1) x step(i)
        transition = torch.rand(batch_size, vocab_size, vocab_size)
        # add some bias to use nonterminals
        transition[:, :_num_action] += 100
        transition[:, _num_action:, :_num_action] += 100
        self.transition = torch.nn.Parameter(transition)  # for debug
        self.vocab_size = vocab_size

    def __call__(self, current_tokens):
        # input_ids: batch x beam x max_length
        # cursor: batch x beam, point to the current location

        input_ids = current_tokens.unsqueeze(-1).expand(-1, -1, self.vocab_size)  # batch x beam x vocab
        return self.transition.gather(1, input_ids)


class WordSyncProcessor:
    def __init__(self, shift_size, beam_size):
        assert (beam_size >= shift_size).all()
        self.shift_size = shift_size
        self.beam_size = beam_size

    def process(self, scores: torch.Tensor, finished_beams: torch.Tensor, impl="cpu", breakpoints=False):
        # scores = scores.detach().cpu().numpy()
        # finished_beams = finished_beams.numpy()

        # results = list(map(lambda x: self.process_one_cpu(*x), zip(scores, finished_beams)))
        # batched = list(map(torch.Tensor, zip(*results)))

        BATCH, BEAM, VOCAB = scores.shape
        scores = scores.clone()
        # set a sufficient large value to finished beams
        # then we will always select them at the beginning
        # scores[..., -1].masked_fill_(finished_beams, 1e9)
        scores[..., :-1].masked_fill_(finished_beams.unsqueeze(-1).expand(-1, -1, VOCAB - 1), -float("inf"))
        scores = scores.view(BATCH, -1)
        # beam_size * 2 is a sufficient upper bound
        topk_scores, topk_indices = torch.sort(scores, dim=-1, descending=True)
        if breakpoints:
            breakpoint()
        topk_indices.masked_fill_(topk_scores == -float("inf"), -99999)

        if impl == "cpu":
            indices = find_selected_indices_batch_cpu(
                topk_indices.cpu().numpy(),
                finished_beams.sum(1).cpu().numpy(),
                VOCAB,
                self.shift_size.cpu().numpy(),
                self.beam_size,
            )
            indices = torch.from_numpy(indices).to(scores.device)
        else:
            indices = cuda.device_array((BATCH, self.beam_size), np.int64)
            indices[:] = -1
            assert BATCH < 1024, "Need configure cuda kernel launch parameters"
            find_selected_indices_cuda[1, BATCH](
                cuda.as_cuda_array(topk_indices),
                cuda.as_cuda_array(finished_beams),
                indices,
                VOCAB,
                self.shift_size,
                self.beam_size,
            )
            indices = torch.as_tensor(indices, device="cuda").to(scores.device)
        return indices


@numba.jit(nopython=True, parallel=True)
def find_selected_indices_batch_cpu(sorted_indices, num_finished, vocab_size, shift_size, beam_size):
    result = np.zeros((len(sorted_indices), beam_size), dtype=np.int64)
    for bidx in numba.prange(len(sorted_indices)):
        result[bidx] = find_selected_indices_cpu(
            sorted_indices[bidx], num_finished[bidx], vocab_size, shift_size[bidx], beam_size
        )
    return result


@numba.jit(nopython=True)
def find_selected_indices_cpu(sorted_indices, num_finished, vocab_size, shift_size, beam_size):
    # shift_size = num_finished + shift_size
    shift_size = shift_size
    topk_size = beam_size - shift_size
    s_cursor, t_cursor = 0, 0
    output_indices = np.full(beam_size, -1, dtype=np.int64)

    for index in sorted_indices:
        if index == -99999:
            continue
        if (index % vocab_size) == vocab_size - 1 and s_cursor < shift_size:
            output_indices[t_cursor] = output_indices[s_cursor]
            output_indices[s_cursor] = index
            s_cursor += 1
            t_cursor += 1
        elif t_cursor - s_cursor < topk_size:
            output_indices[t_cursor] = index
            t_cursor += 1
        if t_cursor == beam_size:
            break

    return output_indices


@cuda.jit
def find_selected_indices_cuda(sorted_indices, num_finished, output_indices, vocab_size, shift_size_, beam_size):
    pos = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if pos < sorted_indices.shape[0]:
        shift_size = num_finished[pos] + shift_size_[pos]
        topk_size = beam_size - shift_size
        s_cursor, t_cursor = 0, 0

        output_indices = output_indices[pos]
        sorted_indices = sorted_indices[pos]

        for index in sorted_indices:
            if index == -99999:
                continue
            elif (index % vocab_size) == vocab_size - 1 and s_cursor < shift_size:
                output_indices[t_cursor] = output_indices[s_cursor]
                output_indices[s_cursor] = index
                s_cursor += 1
                t_cursor += 1
            elif t_cursor - s_cursor < topk_size:
                output_indices[t_cursor] = index
                t_cursor += 1
            if t_cursor == beam_size:
                break


@dataclass
class VocabMeta:
    action_range: tuple[int, int]


def word_sync_beam_search(
    model: BoringLM,
    ranges: TokenTypeRanges,
    tokens: torch.Tensor,
    # 1 for begin, we will run word sync at these places.
    is_subtoken_begin: torch.Tensor,
    startofword_id,
    vocab_size,
    beam_size,
    word_beam_size,
    shift_size,
    max_length,  # max length of the output sequence
    max_action_length_per_step,  # max steps between two words
    vocab_meta: VocabMeta,
):
    assert word_beam_size <= beam_size
    # tokens: batch x seq_length: the terminal sequences
    BATCH, SEQLEN = tokens.shape
    device = tokens.device
    with torch.no_grad():
    # init beams
        beams = torch.full((BATCH, beam_size, max_length), 0, device=device)
        TGAgents = [[TGAgent(ranges, -1, 1) for _ in range(beam_size)] for __ in range(BATCH)]
        past_keys = None # batch * beam * step * hidden (layer * w_dim)
        past_values = None
        padding_lengths = torch.full((BATCH, beam_size), 0, device=device)
        beam_sum_scores = torch.full((BATCH, SEQLEN), 0.0, device=device)
        beams[:, 0, 0] = tokens[:, 0]  # 0 is BOS
        # logger.info(beams)
        beam_token_scores = torch.full((BATCH, beam_size, max_length), 0.0, device=device)
        beam_scores = torch.full((BATCH, beam_size), -float("inf"), device=device)
        beam_scores[:, 0] = 0
        beam_cursors = torch.full([BATCH, beam_size], 1, device=device)  # point to the current location.
        word_count = torch.full((BATCH, beam_size), 0, device=device)
        action_count = torch.full((BATCH, beam_size), 0, device=device)
        # run word sync beam search
        # logger.info(SEQLEN)
        for terminal_cursor in range(1, SEQLEN):
            # logger.info(terminal_cursor)
            # logger.info(tokens)
            # if terminal_cursor == 59:
            #     # logger.info(tokens[:, terminal_cursor])
            #     breakpoint()
            next_word = tokens[:, terminal_cursor]
            # if terminal_cursor == SEQLEN-1:
            #         breakpoint()

            prev_compose = False
            # logger.info(terminal_cursor)
            # logger.info(next_word)
            # run beam search
            finished_beams = torch.zeros((BATCH, beam_size), dtype=torch.bool, device=device)
            completed_beams = torch.zeros((BATCH, beam_size), dtype=torch.bool, device=device)
            completed_beams[beam_scores == -float("inf")] = 1
            
            processor = WordSyncProcessor(
                # if is not subtoken begin, we only allow shift => no action will be taken
                torch.where(is_subtoken_begin[:, terminal_cursor], shift_size, beam_size),
                beam_size,
            )
            # logger.info(terminal_cursor)
            for k in range(max_action_length_per_step):
                if finished_beams.all() or completed_beams.all():
                    break
                
                # >>> TODO change here to support other models
                # batch x beam, just for BoringLM
                current_tokens = beams.gather(2, (beam_cursors - 1).unsqueeze(-1)).squeeze(-1)
                # current_tokens = tokens[:, terminal_cursor - 1].unsqueeze(-1).expand(BATCH, beam_size)
                cur_tokens = current_tokens.cpu().numpy()
                # print(cur_tokens)
                cur_pos = (beam_cursors - 1).cpu().numpy()
                # print(cur_pos)
                # if terminal_cursor == 33:
                #     logger.info(111)
                #     breakpoint()
                finished = finished_beams.cpu().numpy()
                completed = completed_beams.cpu().numpy()
                pre_padding = padding_lengths.cpu().numpy()
                new_tokens = []
                new_tokens_2 = []
                token_mask = []
                attn_masks = []
                relative_pos = []
                start_time = time.time()
                for bidx in range(BATCH):
                    for b in range(beam_size):
                        token_p = vocab_size if startofword_id[cur_tokens[bidx, b]] == 1 else cur_tokens[bidx, b]
                        token_pair, mask, relpos = TGAgents[bidx][b].step(cur_tokens[bidx, b], token_p, cur_pos[bidx, b], cur_pos[bidx, b] + 1, finished[bidx, b] | completed[bidx, b], pre_padding[bidx, b])
                        new_tokens.append(token_pair[0])
                        if token_pair[1] is not None:
                            new_tokens_2.append(token_pair[1])
                            token_mask.append(1)
                        else:
                            new_tokens_2.append(token_pair[0])
                            token_mask.append(0)
                        attn_masks.append(mask)
                        relative_pos.append(relpos)
                # logger.info(ttype)
                # logger.info(beams[0][0])
                # logger.info(attn_masks[0])
                # logger.info(relative_pos[0])
                # if terminal_cursor == 33:
                #     logger.info(new_tokens)
                #     logger.info(new_tokens_2)
                #     logger.info(attn_masks)
                #     logger.info(relative_pos)
                #     breakpoint()
                # if terminal_cursor == SEQLEN-1:
                #     breakpoint()
                        
                new_tokens = torch.LongTensor(new_tokens).cuda().reshape(BATCH*beam_size, -1)
                new_tokens_2 = torch.LongTensor(new_tokens_2).cuda().reshape(BATCH*beam_size, -1)
                token_mask = torch.BoolTensor(token_mask).cuda().reshape(BATCH*beam_size, -1)
                attn_masks = torch.BoolTensor(np.array(attn_masks)).cuda().reshape(BATCH*beam_size, 1, -1)
                relative_pos = torch.LongTensor(np.array(relative_pos)).cuda().reshape(BATCH*beam_size, 1, -1)
                if past_keys is not None:
                    shape = past_keys.shape
                    past_keys = past_keys.view(BATCH*beam_size, *shape[2:])
                    past_values = past_values.view(BATCH*beam_size, *shape[2:])
                
                
                
                current_tokens = current_tokens.to(device)
                beam_cursors = beam_cursors.to(device)
                finished_beams = finished_beams.to(device)
                padding_lengths = padding_lengths.to(device)
                # logger.info(time.time() - start_time)
                # logger.info(new_tokens)
                # logger.info(new_tokens_2)
                # logger.info(attn_masks)
                # logger.info(relative_pos)
                

                next_scores, new_keys, new_values = model.constrained_forward_gen(new_tokens,
                                                                                new_tokens_2,
                                                                                token_mask,
                                                                                past_keys,
                                                                                past_values,
                                                                                attn_masks,
                                                                                relative_pos,
                                                                                62,
                                                                                -1,
                                                                                'txl_arc') # batch x beam_size x vocab
                
                next_scores = next_scores.reshape(BATCH, beam_size, -1)
                if past_keys is not None:
                    shape = past_keys.shape
                    past_keys = past_keys.reshape(BATCH, beam_size, *shape[1:])
                    past_values = past_values.reshape(BATCH, beam_size, *shape[1:])
                next_scores = next_scores.log_softmax(-1)
                # logger.info(time.time() - start_time)
                if past_keys is None:
                    shape = new_keys.shape
                    new_keys = new_keys.view(BATCH, beam_size, *shape[1:])
                    new_values = new_values.view(BATCH, beam_size, *shape[1:])
                    past_keys = new_keys
                    past_values = new_values
                else:
                    shape = new_keys.shape
                    new_keys = new_keys.view(BATCH, beam_size, *shape[1:])
                    new_values = new_values.view(BATCH, beam_size, *shape[1:])
                    past_keys = torch.cat([past_keys, new_keys], dim=2)
                    past_values = torch.cat([past_values, new_values], dim=2)
                    # logger.info(time.time() - start_time)
                    past_keys_copy = past_keys.clone()
                    past_values_copy = past_values.clone()
                    past_keys_copy[:, :, 1:] = past_keys[:, :, :-1]
                    past_values_copy[:, :, 1:] = past_values[:, :, :-1]
                    # logger.info(time.time() - start_time)
                    # past_keys_copy[:, :, 0] = new_keys[:, :, 0]
                    # past_values_copy[:, :, 0] = new_values[:, :, 0]
                    past_keys[finished_beams] = past_keys_copy[finished_beams]
                    past_values[finished_beams] = past_values_copy[finished_beams]
                    padding_lengths[finished_beams] += 1
                    padding_lengths[beam_cursors >= max_length] += 1
                
                # <<< change here to support other models
                # if terminal_cursor == SEQLEN-1:
                #     breakpoint()
                # Gather probs of actions and next tokens
                # if prev_compose:
                #     score_mask = torch.zeros_like(next_scores, dtype=torch.int64)
                #     # print(score_mask)
                #     score_mask.scatter_(2, current_tokens.unsqueeze(-1), 1)
                #     # print(score_mask)
                #     score_mask = score_mask.bool()
                #     next_scores.masked_fill_(~score_mask, -float('inf'))
                #     next_scores.masked_fill_(score_mask, 0)
                
                next_token_scores = next_scores.gather(2, next_word[:, None, None].expand(BATCH, beam_size, -1))
                
                action_scores = next_scores[:, :, slice(*vocab_meta.action_range)]
                # print(action_scores)
                # exit()
                # if not prev_compose:
                next_action_mask = action_count > word_count - 2
                next_action_mask = next_action_mask.unsqueeze(-1).expand(-1, -1, action_scores.shape[-1])
                action_scores.masked_fill_(next_action_mask, -float('inf'))
                    # print(next_action_mask)
                    # print(action_scores)

                next_token_scores = torch.cat([action_scores, next_token_scores], dim=-1) 
                # if k == 1:   
                #     print(next_token_scores)
                #     exit()
                # print(next_token_scores)
                #     exit()
                # update beam scores
                next_token_scores.masked_fill_(finished_beams.unsqueeze(-1), -float("inf"))
                next_token_scores[..., -1].masked_fill_(finished_beams, 0)

                next_scores = beam_scores.unsqueeze(-1) + next_token_scores
                # if terminal_cursor == SEQLEN-1:
                #     breakpoint()
                # if k == 1:
                #     print(next_scores)

                # exit()
                # apply word sync
                # logger.info(time.time() - start_time)
                if terminal_cursor == SEQLEN-1:
                    breakpoints = False
                else:
                    breakpoints = False
                next_indices = processor.process(next_scores, finished_beams, impl='cpu', breakpoints=breakpoints)
                # logger.info(time.time() - start_time)
                invalid_beam_mask = next_indices == -1
                # logger.info(next_indices)
                next_indices = next_indices.clamp(0)  # make gather predictable
                _size = next_scores.shape[-1]
                next_beam_idx = next_indices // _size
                next_action_idx = next_indices % _size
                # set real vocab index
                restore_mask = next_action_idx == _size - 1
                action_mask = next_action_idx != _size - 1
                next_action_idx = torch.where(restore_mask, next_word[:, None].expand(-1, beam_size), next_action_idx)
                # logger.info(next_action_idx)
                next_action_idx[next_action_idx == 0] = vocab_meta.action_range[0]
                next_action_idx[next_action_idx == 1] = vocab_meta.action_range[0] + 1
                # select beams remaining in the beam
                # if terminal_cursor == SEQLEN-1:
                #     breakpoint()
                beams = beams.gather(1, next_beam_idx[..., None].expand(-1, -1, max_length))
                beam_token_scores = beam_token_scores.gather(1, next_beam_idx[..., None].expand(-1, -1, max_length))
                beam_scores = beam_scores.gather(1, next_beam_idx)
                action_count = action_count.gather(1, next_beam_idx)
                beam_cursors = beam_cursors.gather(1, next_beam_idx)
                beam_mask = finished_beams.gather(1, next_beam_idx)
                completed_beams = completed_beams.gather(1, next_beam_idx)
                # >>>
                past_keys = past_keys.gather(1, next_beam_idx[..., None, None].expand(-1, -1, past_keys.shape[-2], past_keys.shape[-1]))
                past_values = past_values.gather(1, next_beam_idx[..., None, None].expand(-1, -1, past_values.shape[-2], past_values.shape[-1]))
                padding_lengths = padding_lengths.gather(1, next_beam_idx)
                
                TGAgents = [[deepcopy(TGAgents[bidx][next_beam_idx[bidx, b]]) for b in range(beam_size)] for bidx in range(BATCH)]
                # <<<
                completed_beams[beam_cursors >= max_length] = 1
                beam_cursors = beam_cursors.clamp(0, max_length - 1)
                # update cursor and write newly generated tokens into beams
                beams.scatter_(2, beam_cursors.unsqueeze(-1), next_action_idx.unsqueeze(-1))
                # logger.info(beams[:, :, 0:10])
                # exit()
                beam_token_scores.scatter_(
                    2, beam_cursors.unsqueeze(-1), next_token_scores.flatten(1).gather(1, next_indices).unsqueeze(-1)
                )
                beam_scores = torch.where(beam_mask, beam_scores, next_scores.flatten(1).gather(1, next_indices))
                beam_scores.masked_fill_(invalid_beam_mask, -float("inf"))
                beam_cursors += (~beam_mask).int()
                # beam_cursors = beam_cursors.clamp(0, max_length - 1)
                # if not prev_compose:
                action_count += (action_mask & ~beam_mask).int()
                # print(action_count)
                # update finished beams
                finished_beams = restore_mask
                completed_beams[finished_beams] = 1
                completed_beams[invalid_beam_mask] = 1
                # prev_compose = not prev_compose
                if k == max_action_length_per_step - 1:
                    beam_scores.masked_fill_(~finished_beams, -float("inf"))
                # logger.info(time.time() - start_time)
                # if terminal_cursor == SEQLEN-1:
                #     breakpoint()
            word_count += is_subtoken_begin[:, terminal_cursor].unsqueeze(-1)
            
            # breakpoint()
            beam_sum_scores[:, terminal_cursor] = beam_scores.logsumexp(-1)
            # prune beams
            nlargest = torch.topk(beam_scores, word_beam_size, -1)[0][:, word_beam_size - 1, None]
            beam_scores.masked_fill_(beam_scores < nlargest, -float("inf"))

        mask = torch.arange(max_length, device=device)[None, None, :] > beam_cursors[..., None]
        beams.masked_fill_(mask, -1)
        beam_token_scores.masked_fill_(mask, 0)
    return beams, beam_token_scores, beam_scores, beam_sum_scores

def load_vocab(path):
    f = open(path, 'r')
    vocab = [line.strip().split()[0] for line in f.readlines()]
    word2idx = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(vocab)
    startofword_id = [0 for _ in range(vocab_size)]
    for i in range(len(vocab)):
        if vocab[i] == '<s>':
            bos_id = i
        elif vocab[i] == '</s>':
            eos_id = i
        elif vocab[i] == '<pad>':
            pad_id = i
        elif vocab[i] == 'left_arc':
            left_arc = i
        elif vocab[i] == 'right_arc':
            right_arc = i
        elif vocab[i] == 'pop_root':
            pop_root = i
        elif vocab[i].startswith('▁'):
            startofword_id[i] = 1
    return vocab, vocab_size, word2idx, bos_id, eos_id, pad_id, left_arc, right_arc, pop_root, startofword_id

class TestSuiteParser:
    def __init__(self, test_suite_file):
        self.test_suite_file = test_suite_file
        self.read_test_suite()
        self.answers = [0 for _ in range(len(self.meta_data["data"]))]

    def read_test_suite(self):
        data_file = "test_suites/json/{}.json".format(self.test_suite_file)
        with open(data_file, "r") as f:
            data = json.load(f)
        self.meta_data = {
            "formula": data["predictions"][0]["formula"],
            "data": self.get_sents(data),
        }

    def get_sents(self, data):
        all_ex = []
        for item in data["items"]:
            curr_ex = {}
            for cond in item["conditions"]:
                regions = [x["content"] for x in cond["regions"]]
                curr_ex[cond["condition_name"]] = regions
            all_ex.append(curr_ex)
        return all_ex

    def extract_formulas(self, surprisal_dict):
        formula = self.meta_data["formula"]
        keys = re.findall(r"%([\w|-]+)%", formula)
        keys = set(keys)
        for key in keys:
            positions = set(re.findall(r"\((\d+);%{}%".format(key), formula))
            for position in positions:
                formula = formula.replace(
                    "({};%{}%)".format(position, key),
                    str(surprisal_dict[key][int(position)]),
                )
        ### replace [ with ( and ] with ) to make it a valid math expression

        formula = formula.replace("[", "(")
        formula = formula.replace("]", ")")
        return formula

    def get_example(self, idx):
        return self.meta_data["data"][idx]

    def evaluate_example(self, idx, evaluator, verbose=False):
        examples = self.get_example(idx)
        phen2surprisals = {}
        for phen in examples:
            
            target_surprisals, logprobs, target_idxs, _ = evaluator.get_surprisals(
                examples[phen]
            )
            if verbose:
                print("Regions: {}".format(examples[phen]))
                print(logprobs)
            phen2surprisals[phen] = [0] + target_surprisals

        extracted_formula = self.extract_formulas(phen2surprisals)
        self.answers[idx] = extracted_formula

    def evaluate_all(self, evaluator=None):
        for idx in tqdm(range(len(self.meta_data["data"]))):
            self.evaluate_example(idx, evaluator)
        return

def eval_math_expr(expr):
    try:
        return eval(expr)
    except:
        return math.nan
    
if __name__ == "__main__":
    # seed_everything(42)

    configure_logger('logs/beam_txl.log')
    logger = get_logger()
    vocab, vocab_size, word2idx, bos, eos, pad_id, left_arc, right_arc, pop_root, startofword_id = load_vocab('tokenizer/spm.vocab')
    # 0,1,2 actions. 3~9 terminals
    BATCH = 1
    BEAM = 100
    WORD_BEAM = 10
    SHIFT = 0
    VOCAB = vocab_size
    MAX_LEN = 500
    torch.manual_seed(123456)
    np.random.seed(123456)
    ranges = TokenTypeRanges(bos, pad_id, vocab_size, left_arc, right_arc)
    vocab_meta = VocabMeta([left_arc, right_arc + 1])
    # subtoken_begin = torch.randn(BATCH, MAX_LEN // 3) < 0.5  # [:, 0]  is always a begin
    original = []
    # original_startofword = []
    subtoken_begins = []
    original_length = []
    original_seq_length = []
    checkpoint = torch.load('models/standard_rbt_txl_1.pt')
    model = checkpoint['model']
    model.eval()
    model.cuda()
    sp = spm.SentencePieceProcessor(model_file='tokenizer/spm.model')
    file_list = os.listdir("test_suites/json/.")
    # print(file_list)
    for file in file_list:
        test_suite_parser = TestSuiteParser(file[:-5])
        logger.info(file[:-5])
        BEAM = 100
        WORD_BEAM = 10
            
        # print(test_suite_parser.meta_data["formula"])
        for idx in tqdm(range(len(test_suite_parser.meta_data["data"]))):
            examples = test_suite_parser.get_example(idx)
            phen2surprisals = {}
            for phen in examples:
                # logger.info(examples[phen])
                encoded = sp.Encode(examples[phen] + ["."], out_type=int)
                # logger.info(sp.Encode(" ".join(examples[phen]), out_type=int))
                tgt_idx = []
                encoded.insert(0, [bos])
                encoded.append([pop_root])
                encoded.append([eos])
                # logger.info(encoded)
                word_idx = -1
                prev_idx = -1
                for word in encoded:
                    word_idx += len(word)
                    tgt_idx.append((prev_idx, word_idx))
                    prev_idx = word_idx 
                tgt_idx = tgt_idx[1:-1]
                encoded = [x for word in encoded for x in word]

                # target_surprisals, logprobs, target_idxs, _ = evaluator.get_surprisals(
                #     examples[phen]
                # )
                subtoken_begin = [1 if startofword_id[word] == 1 else 0 for word in encoded]
                subtoken_begin[-2] = 1 # for pop_root
                subtoken_begin[-1] = 1 # for eos
                tokens = torch.LongTensor(encoded).cuda().reshape(1, -1)
                word_num = sum(subtoken_begin) - 2 # -2 for pop_root and eos 
                subtoken_begin = torch.BoolTensor(subtoken_begin).cuda().reshape(1, -1)
                # tokens = tokens[:, :5]
                # subtoken_begin = subtoken_begin[:, :5]
                # print(subtoken_begin)
                # exit()
                MAX_LEN = len(tokens[0]) + (word_num - 1)
                beams, beam_token_scores, beam_scores, beam_sum_scores = word_sync_beam_search(
                    model, ranges, tokens, subtoken_begin, startofword_id, vocab_size, BEAM, WORD_BEAM, SHIFT, MAX_LEN, 15, vocab_meta
                )
                # logger.info(beams[:, 0:10, :])
                # logger.info(beam_scores[:, 0:10])
                # logger.info(beam_token_scores[:, 0:10, :].sum(-1))
                # exit()
                scores = -beam_sum_scores.cpu().numpy() # - means surprisals
                target_surprisals = [scores[0][tgt_idx[i][1]] - scores[0][tgt_idx[i][0]] for i in range(len(tgt_idx))]
                # print(target_surprisals)
                # logger.info(target_surprisals)
                phen2surprisals[phen] = [0] + target_surprisals

            extracted_formula = test_suite_parser.extract_formulas(phen2surprisals)
            test_suite_parser.answers[idx] = extracted_formula
        
        acc = 0.0
        for formula in test_suite_parser.answers:
            answer = eval_math_expr(formula)
            logger.info(f"score: {answer}")
            acc += answer
        
        logger.info(f"correct rate: {acc / len(test_suite_parser.answers)}")
    # with open('data/bllip_test_raw.csv', 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         sent = [bos] + [int(word) for word in line.strip().split(',')] + [pop_root] + [eos]
    #         original.append(sent)
    #         sent_startofword = [vocab_size if startofword_id[word] == 1 else word for word in sent]
    #         subtoken_begin = [1 if startofword_id[word] == 1 else 0 for word in sent]
    #         # subtoken_begin[0] = 1
    #         subtoken_begin[-2] = 1
    #         subtoken_begin[-1] = 1
    #         original_seq_length.append(len(sent) - 1)
    #         subtoken_begins.append(subtoken_begin)
    #         # original_startofword.append(sent_startofword)
    # with torch.no_grad():
    #     for i in range(0, len(original)):
    #         if i < 4:
    #             continue

    #         # original[i] = [1, 72, 7191, 76, 59, 2]
    #         # subtoken_begins[i] = [0, 1, 1, 1, 1, 0]
    #         word_num = sum(subtoken_begins[i]) - 2
    #         # logger.info(word_num)
    #         tokens = torch.LongTensor(original[i]).cuda().reshape(1, -1)
    #         subtoken_begin = torch.BoolTensor(subtoken_begins[i]).cuda().reshape(1, -1)
    #         # logger.info(tokens)
    #         # logger.info(subtoken_begin)
    #         MAX_LEN = len(tokens[0]) + 2 * (word_num - 1)
    #         # logger.info(MAX_LEN)
    #         # print(subtoken_begin)
    #         # exit()
    #         beams, beam_token_scores, beam_scores, beam_sum_scores = word_sync_beam_search(
    #             model, ranges, tokens, subtoken_begin, startofword_id, vocab_size, BEAM, WORD_BEAM, SHIFT, MAX_LEN, 15, vocab_meta
    #         )
    #         logger.info(beams[0, 0:5])

    #         logger.info(beam_sum_scores[0, -1].item())
    #         logger.info(beam_scores[0, 0:5])
    #         logger.info(len(tokens[0]))
    #         logger.info(beam_sum_scores[0, -1].item()/len(tokens[0]))
    #         # exit() 

    # beams2, beam_token_scores2, beam_scores2, beam_sum_scores = word_sync_beam_search(
    #     model, ranges, tokens, torch.ones_like(subtoken_begin), BEAM, WORD_BEAM, SHIFT, MAX_LEN, 3, vocab_meta
    # )
    
    # breakpoint()

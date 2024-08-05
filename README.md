# Dep_Transformer_Grammars

Code Repository for our ACL '24 paper: ["Dependency Transformer Grammars: Integrating Dependency Structures into Transformer Language Models"](https://arxiv.org/abs/2407.17406v1).

## Data Processing

Follow the instructions in `README.MD` in `data_scripts`.

## Environment

You can follow `environments.yml` for reference environment. 

You should also set up the cpp_extension for masking by running `sh cpp_ext_build.sh`.
(The cpp code in masking_bllip is modified from https://github.com/google-deepmind/transformer_grammars/tree/main/transformer_grammars/models/masking.)

## Training

Follow the `scripts/train_bllip_dep_gen.sh` for `DTG` training. The paths in the file should be changed to your desired path. The hyperparameters we use for training are shown in this file.

We also have `scripts/train_bllip_txl_dep.sh` as reference for `txl (trans)` training.

## Evaluation
### Syntactic Generalization
For evaluation, we provide our implementation of Word Synchronous Beam Search for `DTG` and `txl (trans)`. 

The implementation is mainly for [SyntaxGym (SG)](https://github.com/cpllab/syntactic-generalization) tests. You can do simple modification to the `main` function for other use (like BLiMP evaluation).

To evaluate SG, run `python beam_search_standard.py` for `DTG` and `python beam_search_txl.py` for `txl (trans)`. (Don't forget to change the model and the tokenizer in the code to your own path.)

### Perplexity

You can use the Biaffine parser implemented in https://github.com/yzhangcs/parser to sample dependency parse trees for a sentence. 

Then evaluate the log_likelihood for each tree with simple modification of `train_dep_gen.py` (make use of the `eval` function). 

To calculate the perplexity, please refer to our paper for more details.

## Notes

Our model is a sentence-level syntactic language model, so DTGs only model a complete sentence. A document-level version of DTG is what we are now working on and maybe released in the future.








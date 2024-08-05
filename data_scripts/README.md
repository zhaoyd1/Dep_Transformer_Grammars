# BLLIP Data Build
## Data download and dependency data generation
First, download the BLLIP dataset from https://catalog.ldc.upenn.edu/LDC2000T43. Move the   `bliip_87_89_wsj` folder into `/data_scripts`.

Then 
1. run `python get_wsj_bllip.py`
2. run `python strip_func.py < bllip_train_LG.xxx > bllip_train_LG` (also for `bllip_dev.xxx` and `bllip_test.xxx`) 
3. run `python clean.py`
4. run `java -cp stanford-parser_3.3.0.jar edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -keepPunct -conllx -treeFile bllip_train_LG_clean > bllip_train_LG_conll` (also for dev and test)
5. run `python conll_to_dep.py`

We will get three files `train_LG.bllip_action`, `dev.bllip_action`, `test.bllip_action` as our dependency transition data.

## Tokenization
Then we build the sentencepiece tokenizer with `python tokenizer_build.py`. 

With this tokenizer, we can follow the instruction in https://github.com/google-deepmind/transformer_grammars to tokenize the data and do post-processing. (in section `Training and using a TG model/Data preparation/With SentencePiece/Tokenizing the data (SentencePiece)`)

<font color=Red>!!!</font> Tips: To change the postprocess procedure correctly into the dependency form, you can simply change <font color=Blue>line 208</font> in `transformer_grammars/data/sp_utils.py` into 

`elif re.fullmatch(r"\([A-Z]+", token) or token in ['left_arc', 'right_arc', 'pop_root']:`

Finally, we should get three files (train/dev/test) ending with `.csv`, which we use for our training. You can move them into a `Dep_Transformer_Grammars/data` folder for further use. 
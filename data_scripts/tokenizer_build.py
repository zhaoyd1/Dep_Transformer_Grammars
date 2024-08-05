import sentencepiece as spm
spm.SentencePieceTrainer.Train(input='data/train_LG.bllip_action', model_prefix='tokenizer/spm',vocab_size=32719, \
                               character_coverage=1.0, pad_id=0, bos_id=1, eos_id=2, unk_id=3, \
                               user_defined_symbols='left_arc,right_arc,pop_root', \
                               max_sentence_length=100000, shuffle_input_sentence=True
                               )

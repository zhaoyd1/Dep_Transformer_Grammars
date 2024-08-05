from os.path import join
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
import nltk
import glob
from tqdm import tqdm

# train_splits = ["0" + str(i) for i in range(2, 10)] + [str(i) for i in range(10, 22)]
# test_splits = ["23"]
# dev_22_splits = ["22"]
# dev_24_splits = ["24"]



path_1987 = 'bliip_87_89_wsj/1987/w7_%03d'
path_1988 = 'bliip_87_89_wsj/1988/w8_%03d'
path_1989 = 'bliip_87_89_wsj/1989/w9_%03d'

train_path_XS = [path_1987 % id for id in [71, 122]] + \
                [path_1988 % id for id in [54, 107]] + \
                [path_1989 % id for id in [28, 37]]

train_path_SM = [path_1987 % id for id in [35, 43, 48, 54, 61, 71, 77, 81, 96, 122]] + \
                [path_1988 % id for id in [24, 54, 55, 59, 69, 73, 76, 79, 90, 107]] + \
                [path_1989 % id for id in [12, 13, 15, 18, 21, 22, 28, 37, 38, 39]]

train_path_MD = [path_1987 % id for id in [5, 10, 18, 21, 22, 26, 32, 35, 43, 47, 48, 49, 51, 54, 55, 56, 57, 61, 62, 65, 71, 77, 79, 81, 90, 96, 100, 105, 122, 125]] + \
                [path_1988 % id for id in [12, 13, 14, 17, 23, 24, 33, 39, 40, 47, 48, 54, 55, 59, 69, 72, 73, 76, 78, 79, 83, 84, 88, 89, 90, 93, 94, 96, 102, 107]] + \
                [path_1989 % id for id in range(12, 42)]

train_path_LG = [path_1987 % id for id in range(3, 128)] + \
                [path_1988 % id for id in range(3, 109)] + \
                [path_1989 % id for id in range(12, 42)]


dev_path = ['../raw/bliip_87_89_wsj/1987/w7_001', '../raw/bliip_87_89_wsj/1988/w8_001', '../raw/bliip_87_89_wsj/1989/w9_010']
test_path = ['../raw/bliip_87_89_wsj/1987/w7_002', '../raw/bliip_87_89_wsj/1988/w8_002', '../raw/bliip_87_89_wsj/1989/w9_011']

def glob_files(paths):
    return [
        fname for path in paths for fname in sorted(
            glob.glob(join(path, "*"))
            )]
    
def write_to_file(paths, outfile, add_top=False, mode=0):
    # print(glob_files(paths, mode))
    # exit()
    global total_line
    global total_doc
    all_paths = [[path] for path in glob_files(paths)]
    for path in all_paths:
        reader = BracketParseCorpusReader('.', path)
        with open(outfile, 'a') as f:
            length = len(reader.parsed_sents())
            for tree in reader.parsed_sents():
                tree_rep = tree.pformat(margin=1e100)
                if add_top:
                    tree_rep = "(TOP %s)" % tree_rep
                assert('\n' not in tree_rep)
                f.write(tree_rep)
                total_line += 1
                f.write("\n")
            total_doc += 1
    

def write_to_file2(paths, outfile, add_top=False, mode=0):
    # print(glob_files(paths, mode))
    # exit()
    # reader = BracketParseCorpusReader('.', glob_files([paths[0]]))
    # reader2 = BracketParseCorpusReader('.', glob_files([paths[1]]))
    # reader3 = BracketParseCorpusReader('.', glob_files([paths[2]]))
    paths_1 = [[path] for path in glob_files([paths[0]])]
    paths_2 = [[path] for path in glob_files([paths[1]])]
    paths_3 = [[path] for path in glob_files([paths[2]])]

    with open(outfile, 'w') as f:
        length = mode * 500
        
        for path in paths_1:
            reader = BracketParseCorpusReader('.', path)
            cnt = 0
            for tree in tqdm(reader.parsed_sents()[:length]):
                tree_rep = tree.pformat(margin=1e100)
                if add_top:
                    tree_rep = "(TOP %s)" % tree_rep
                assert('\n' not in tree_rep)
                f.write(tree_rep)
                f.write("\n")
                cnt += 1
            length -= cnt
            if length <= 0:
                break
        
        length = mode * 500
        for path in paths_2:
            reader = BracketParseCorpusReader('.', path)
            cnt = 0
            for tree in tqdm(reader.parsed_sents()[:length]):
                tree_rep = tree.pformat(margin=1e100)
                if add_top:
                    tree_rep = "(TOP %s)" % tree_rep
                assert('\n' not in tree_rep)
                f.write(tree_rep)
                f.write("\n")
                cnt += 1
            length -= cnt
            if length <= 0:
                break
        
        length = mode * 500
        for path in paths_3:
            reader = BracketParseCorpusReader('.', path)
            cnt = 0
            for tree in tqdm(reader.parsed_sents()[:length]):
                tree_rep = tree.pformat(margin=1e100)
                if add_top:
                    tree_rep = "(TOP %s)" % tree_rep
                assert('\n' not in tree_rep)
                f.write(tree_rep)
                f.write("\n")
                cnt += 1
            length -= cnt
            if length <= 0:
                break
        

total_line = 0
total_doc = 0
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_root")
    parser.add_argument("--revised_root")
    parser.add_argument("--mode", type=str, default="LG")
    args = parser.parse_args()
    
    train_path = locals()["train_path_" + args.mode]
    
    write_to_file2(test_path, 'bliip_test.xxx', args.add_top, 2)
    write_to_file2(dev_path, 'bliip_dev.xxx', args.add_top, 1)
    print(len(train_path))
    for i in tqdm(range(0, len(train_path), 10)):
        write_to_file(train_path[i:min(i+10, len(train_path))], 'bllip_train_' + args.mode + '.xxx', False, 0)
        # print(min(i+10, len(train_path)))
    
    print(total_doc)
    print(total_line)
    

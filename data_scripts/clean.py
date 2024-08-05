import re
for s in ['dev', 'test', 'train_LG']:
    file = 'bllip_' + s
    output_file = 'bllip_' + s + '_clean'
    f1 = open(file, 'r')
    f2 = open(output_file, 'w')
    lines = f1.readlines()
    sents = [line.strip() for line in lines]
    for sent in sents:
        new_sent = re.sub(r"#\d+", '', sent)
        f2.write(new_sent + '\n')
    f1.close()
    f2.close()   
        
            
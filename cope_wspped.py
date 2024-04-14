import argparse
## add file name
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('filename', type=str, help='filename')
args=parser.parse_args()
with open(args.filename) as f:
    ## read every line and analyze
    epoch_runtime = []
    for line in f:

        ## recognize this patern '[2024-04-11 06:29:00.858] [info] [trainer.cpp:62] Epoch Runtime: 205524ms' and extract the num
        if line.find('Epoch Runtime') != -1:
            ## find the index of the first number
            start = line.find('Epoch Runtime') + len('Epoch Runtime: ')
            ## find the index of the first non-number
            end = start
            while line[end].isdigit():
                end += 1
            ## extract the number
            num = line[start:end]
            epoch_runtime.append(int(num))
    ## calculate the average 保留两位小数
    print(round(sum(epoch_runtime) / len(epoch_runtime)/1000,2))
    epoch_runtime=epoch_runtime[1:]
    ## print wo warm up
    print(round(sum(epoch_runtime) / len(epoch_runtime)/1000,2))
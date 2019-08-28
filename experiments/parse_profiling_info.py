import pstats
import argparse

'''
to profile a python script:

python -m cProfile -o [output-file-name] [name-of-file-to-run] [--options for the run for that file]
'''
parser = argparse.ArgumentParser()

parser.add_argument('--file-to-parse', type=str, default=None)
parser.add_argument('--sort-by', type=str, default='tottime' , help='Valid options are \
					calls, cumulative, file, ncalls, pcalls, time, tottime, name')
parser.add_argument('--num-of-entries', type=int, default=100)

args = parser.parse_args()

p = pstats.Stats(args.file_to_parse)
p.sort_stats(args.sort_by).print_stats(int(args.num_of_entries))
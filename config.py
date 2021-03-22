import argparse

args = argparse.ArgumentParser(description='Argparse Open~')

args.add_argument('--y', type=int, default=9)
args.add_argument('--n', type=int, default=3)
args.add_argument('--d', type=int, default=1) # 1이 hn / 2가 sort


args = args.parse_args()
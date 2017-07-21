
import os
import sys
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description='Deep RL ATRP draw learning curve')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--extra_arguments', default=None)
    parser.add_argument('--max_files', default=10, type=int)
    parser.add_argument('--file_step', default=1, type=int)
    args = parser.parse_args()

    saved_weights_list = os.listdir(args.output_dir)
    saved_weights_list.sort()
    saved_weights_list.sort(key=len)
    extra_arguments = args.extra_arguments.strip().split()
    with open(os.devnull, 'w') as devnull:
        for saved_weights in saved_weights_list[:args.max_files:args.file_step]:
            full_path = os.path.join(args.output_dir, saved_weights)
            run_list = ['python', 'atrp_evaluator.py', '--render', 'F', '--read_weights', full_path]
            for arg in extra_arguments:
                run_list.append(arg)
            result = subprocess.run(run_list, stdout=subprocess.PIPE, stderr=devnull)
            avg_reward = float(result.stdout.split()[-1])
            print('### saved_weights in {} ###'.format(saved_weights))
            print(result.stdout.decode().replace('\\n', '\n'))

if __name__ == '__main__':
    main()

import re
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os

def extract_epoch_avg_diff(log_file_path: str):
    epoch_stats = defaultdict(lambda: {'diff_sum': 0.0, 'exp_diff_sum': 0.0, 'count': 0})
    pattern = re.compile(
        r'INFO:\s*(\d{4})\s+\d{5} diff=([0-9.]+), exp_diff=([0-9.]+)'
    )
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                diff = float(match.group(2))
                exp_diff = float(match.group(3))
                epoch_stats[epoch]['diff_sum'] += diff
                epoch_stats[epoch]['exp_diff_sum'] += exp_diff
                epoch_stats[epoch]['count'] += 1
    result = {}
    for epoch, stats in sorted(epoch_stats.items()):
        count = stats['count']
        if count > 0:
            avg_diff = stats['diff_sum'] / count
            avg_exp_diff = stats['exp_diff_sum'] / count
            result[epoch] = {'avg_diff': avg_diff, 'avg_exp_diff': avg_exp_diff}
    return result

def plot_epoch_diff(log_file_path: str):
    result = extract_epoch_avg_diff(log_file_path)
    epochs = sorted(result.keys())
    avg_diffs = [result[e]['avg_diff'] for e in epochs]
    avg_exp_diffs = [result[e]['avg_exp_diff'] for e in epochs]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, avg_diffs, marker='o', label='avg_diff')
    plt.plot(epochs, avg_exp_diffs, marker='s', label='avg_exp_diff')
    plt.xlabel('Epoch')
    plt.ylabel('Average Value')
    plt.title('Epoch vs. Average diff/exp_diff')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/epoch_diff_jt.png' if 'jt' in log_file_path else 'assets/epoch_diff.png')
    plt.show()

def calc_metrics_mean(log_file_path: str):
    datasets = ['Kodak24', 'BSD300', 'Set14']
    metrics = ['PSNR_DN', 'SSIM_DN', 'PSNR_EXP', 'SSIM_EXP', 'PSNR_MID', 'SSIM_MID']
    stats = {ds: {m: [] for m in metrics} for ds in datasets}

    pattern = re.compile(
        r'INFO:\s*(\w+)\s*-\s*img:[^ ]+\s*-\s*PSNR_DN:\s*([0-9.]+)\s*dB;\s*SSIM_DN:\s*([0-9.]+);'
        r'\s*PSNR_EXP:\s*([0-9.]+)\s*dB;\s*SSIM_EXP:\s*([0-9.]+);'
        r'\s*PSNR_MID:\s*([0-9.]+)\s*dB;\s*SSIM_MID:\s*([0-9.]+)\.'
    )

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ds = match.group(1)
                if ds in datasets:
                    for i, m in enumerate(metrics):
                        stats[ds][m].append(float(match.group(i+2)))

    data = []
    for ds in datasets:
        if stats[ds][metrics[0]]: 
            row = [ds] + [sum(stats[ds][m])/len(stats[ds][m]) for m in metrics]
            data.append(row)

    df = pd.DataFrame(data, columns=['dataset'] + metrics)
    print(df)


parser = argparse.ArgumentParser()
parser.add_argument('--train_log_path', type=str)
parser.add_argument('--valid_log_path', type=str)
args = parser.parse_args()

if __name__ == "__main__":
    if args.train_log_path:
        plot_epoch_diff(args.train_log_path)
    if args.valid_log_path:
        calc_metrics_mean(args.valid_log_path)

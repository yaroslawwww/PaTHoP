import json
import numpy as np

from argparse import ArgumentParser
from src      import TSProcessor


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        help='Configuration file',
        required=True
    )
    args = parser.parse_args()

    with open(args.config) as stream:
        config = json.load(stream)

    with open(config['time_series']) as stream:
        time_series = np.loadtxt(stream, delimiter=',')

    tsp = TSProcessor(config['points_in_template'], config['max_template_spread'])
    tsp.fit(time_series[:config['split']])
    result, _ = tsp.push(config['steps'], config['eps'], 0.01, 3)

    with open('result.txt', 'w+') as stream:
        np.savetxt(stream, result, delimiter=',')


if __name__ == '__main__':
    main()

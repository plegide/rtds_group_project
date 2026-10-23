"""The command-line options the run scripts share: which scenario, and which model."""

import argparse


def _parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--scenario", default="A", choices=["A", "B"],
                        help="A: visit 3 drives the risk (Part 1). B: visit 1 drives it.")
    return parser


def parse_scenario(description):
    return _parser(description).parse_args().scenario


def parse_scenario_and_model(description):
    parser = _parser(description)
    parser.add_argument("--model", default="forest", choices=["forest", "lstm"],
                        help="forest: random survival forest (Part 1). lstm: the LSTM (Part 2).")
    args = parser.parse_args()
    return args.scenario, args.model

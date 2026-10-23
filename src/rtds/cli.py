"""The one command-line option the run scripts share: which scenario to run."""

import argparse


def parse_scenario(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--scenario", default="A", choices=["A", "B"],
                        help="A: visit 3 drives the risk (Part 1). B: visit 1 drives it.")
    return parser.parse_args().scenario

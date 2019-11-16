"""
Class Argument parser

"""

import argparse


class general_parser():
    """
    general usage class of arg parser
    """
    
    def __init__(self, description:str = "" , usage:str = "") -> None:
        self.parser = argparse.ArgumentParser(description=description, usage=usage)

    # TODO: impelement necessary function for an general argument parser class
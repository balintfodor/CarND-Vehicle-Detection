import argparse
import sys

from trainer import Trainer
from detector import Detector
from reader import FrameReader, collect_images


class ArgParser(object):
    # sub command argparse code from https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html
    def __init__(self):
        parser = argparse.ArgumentParser(description='Vehicle Detection')
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            sys.exit(1)
        getattr(self, args.command)()

    def train(self):
        parser = argparse.ArgumentParser(description='Classifier Training')
        parser.add_argument('vehicle_folder', type=str)
        parser.add_argument('non_vehicle_folder', type=str)
        parser.add_argument('--test', action='store_true')
        args = parser.parse_args(sys.argv[2:])

        vehicle_images = collect_images(args.vehicle_folder)
        non_vehicle_images = collect_images(args.non_vehicle_folder)
        trainer = Trainer(args)
        trainer.train(vehicle_images, non_vehicle_images)

    def detect(self):
        parser = argparse.ArgumentParser(description='Detection')
        parser.add_argument('train_file', type=str)
        parser.add_argument('input', type=str)
        parser.add_argument('output', type=str)
        args = parser.parse_args(sys.argv[2:])

        trainer = Trainer(args)
        trainer.load(args.train_file)
        reader = FrameReader(args.input)
        detector = Detector(trainer, reader, args)
        detector.process(args.output)

def main():
    ArgParser()

if __name__ == "__main__":
    main()

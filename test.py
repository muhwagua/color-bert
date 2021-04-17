import argparse

from utils import load_trainer


def main(args):
    trainer = load_trainer(args.checkpoint_path)
    accuracy = trainer.test()
    print(accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    args = parser.parse_args()
    main(args)

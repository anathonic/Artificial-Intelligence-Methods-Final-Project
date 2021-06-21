import argparse

# argumenty wywołania

command = argparse.ArgumentParser()
command.add_argument("--mode",help="train/start")
mode = command.parse_args().mode

# definicja folderów z datasetem

train_dir = 'data/emotions'


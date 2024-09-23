import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Model Training Inputs")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/train_config.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory where the data is stored",
    )
    parser.add_argument(
        "--model_log_dir",
        type=str,
        default="./model_logs",
        help="Directory where the models are/to be saved",
    )
    return parser.parse_args(args)

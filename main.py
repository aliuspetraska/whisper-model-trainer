import torch

from services.model_trainer import ModelTrainerService


def main():
    model_trainer = ModelTrainerService()

    num_gpus = torch.cuda.device_count()
    print('num_gpus: ', num_gpus)

    try:
        print("login")
        model_trainer.login()
        print("load")
        model_trainer.load()
        print("get_data_source")
        model_trainer.get_data_source()
        print("train")
        model_trainer.train()
    except KeyboardInterrupt:
        print("program interrupted by user")
    except Exception as e:
        print(f"an unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
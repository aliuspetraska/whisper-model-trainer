from services.model_trainer import ModelTrainerService


def main():
    model_trainer = ModelTrainerService()

    try:
        model_trainer.login()
        model_trainer.load()
        model_trainer.get_data_source()
        model_trainer.train()
    except KeyboardInterrupt:
        print("program interrupted by user")
    except Exception as e:
        print(f"an unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
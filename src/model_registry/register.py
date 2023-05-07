def register_model(run_id, model_path, model_name, model_description=None):
    # Register the model to the MLflow Model Registry
    model_uri = "runs:/{}/model".format(run_id)
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        description=model_description
    )
    return registered_model


def get_args():
    parser = argparse.ArgumentParser(description='Train pokemon classifier')
    parser = get_telemetry_args(parser)
    parser.add_argument('--train_path', type=str,
                        default='data/splits/train', help='Training data root')
    parser.add_argument('--val_path', type=str,
                        default='data/splits/val', help='Validation data root')
    parser.add_argument('--model_path', type=str,
                        default='saved_models', help='Saved model path')
    parser.add_argument('--num_epochs', type=int,
                        default=20, help='Number of epochs of the training')
    parser.add_argument('--batch_size', type=int,
                        default=8, help='Batch size of the training')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='Learning rate of the training')
    args = parser.parse_args()
    return args.experiment_name, args.train_path, args.val_path, args.model_path, args.num_epochs, args.batch_size, args.learning_rate


if __name__ == "__main__":
    run()

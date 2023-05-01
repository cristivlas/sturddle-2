#! /usr/bin/env python3
'''
Average out several tensorflow models (ensemble-learning)
'''
import argparse
import os

 # https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def average_models(models):
    avg_weights = None
    for i, model in enumerate(models):
        model_weights = model.get_weights()
        if i == 0:
            avg_weights = [weight / len(models) for weight in model_weights]
        else:
            for j in range(len(model_weights)):
                avg_weights[j] += model_weights[j] / len(models)
    averaged_model = tf.keras.models.clone_model(models[0])
    averaged_model.set_weights(avg_weights)
    return averaged_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Averages multiple TensorFlow Keras models')
    parser.add_argument('model_dirs', type=str, nargs='+', help='directories of the input models')
    parser.add_argument('-o', '--output', type=str, default='averaged_model', help='directory for the output averaged model')
    args = parser.parse_args()

    import tensorflow as tf

    models = []
    for model_dir in args.model_dirs:
        models.append(tf.keras.models.load_model(model_dir, custom_objects={'_clipped_mae': None}))

    averaged_model = average_models(models)
    averaged_model.save(args.output)

#!/usr/bin/env python3

import argparse
import json
import os

# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras.models import load_model
from tensorflow import keras


def serialize_weights(model, output_path):
    # Prepare a dictionary to store weights and biases
    weights_dict = {}

    for layer in model.layers:
        if not layer.weights:
            continue
        if isinstance(layer, keras.layers.Dense):
            weights = layer.get_weights()
            weights_dict[layer.name] = {
                'weights': weights[0].tolist(),
                'biases': weights[1].tolist(),
                'input_dim': np.array(weights[0]).shape[0],
                'output_dim': np.array(weights[0]).shape[1],
            }

    # Save to a JSON file
    with open(output_path, 'w') as f:
        json.dump(weights_dict, f, indent=2)

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Serialize Keras model weights to json')

    # Add the arguments
    parser.add_argument('ModelPath', metavar='modelpath', type=str, help='Path to the Keras model')
    parser.add_argument('-o', '--output', help='Output JSON file', default='weights.json')

    # Execute the parse_args() method
    args = parser.parse_args()

    # Load the model
    model = load_model(args.ModelPath, custom_objects={'_clipped_mae' : None})

    # Write as JSON
    serialize_weights(model, args.output)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import argparse
import json
import os
import tempfile

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

def deserialize_weights(model, input_path):
    # Load from a JSON file
    with open(input_path, 'r') as f:
        weights_dict = json.load(f)

    for layer in model.layers:
        if not layer.weights:
            continue
        if isinstance(layer, keras.layers.Dense) and layer.name in weights_dict:
            weights = weights_dict[layer.name]

            # Reshape weights and biases from lists into their correct shapes
            weights_array = np.array(weights['weights'])
            biases_array = np.array(weights['biases'])

            # Ensure the dimensions are as expected
            assert weights_array.shape == (weights['input_dim'], weights['output_dim']), \
                f"Expected shape {(weights['input_dim'], weights['output_dim'])}, but got {weights_array.shape}"

            assert biases_array.shape == (weights['output_dim'],), \
                f"Expected shape {(weights['output_dim'],)}, but got {biases_array.shape}"

            # Set the weights for this layer
            layer.set_weights([weights_array, biases_array])

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Serialize Keras model weights to json')

    # Add the arguments
    parser.add_argument('ModelPath', metavar='modelpath', type=str, help='Path to the Keras model')
    parser.add_argument('-o', '--output', help='Output JSON file', default='weights.json')
    parser.add_argument('--test', action='store_true')

    # Execute the parse_args() method
    args = parser.parse_args()

    # Load the model
    model = load_model(args.ModelPath, custom_objects={'_clipped_mae' : None})

    # Write as JSON
    serialize_weights(model, args.output)

    if args.test:
        deserialize_weights(model, args.output)
        with tempfile.NamedTemporaryFile() as temp_file:
            serialize_weights(model, temp_file.name)
            with open(temp_file.name, 'rb') as f1, open(args.output, 'rb') as f2:
                assert f1.read() == f2.read()

if __name__ == '__main__':
    main()

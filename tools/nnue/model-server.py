#!/usr/bin/env python3
import json
import threading
from functools import wraps

from flask import Flask, abort, jsonify, request

WHITELIST = [
    '127.0.0.1',
]

app = Flask(__name__)
lock = threading.Lock()

def limit_remote_addr():
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if request.remote_addr not in WHITELIST:
                abort(403)  # Forbidden
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/get-model', methods=['GET'])
@limit_remote_addr()
def read():
    try:
        with lock:
            with open('model.json', 'r') as f:
                data = json.load(f)
        return jsonify({'model': data}), 200
    except FileNotFoundError:
        app.logger.error('Model file not found.')
        return jsonify({'status': 'failure', 'error': 'Model file not found.'}), 404
    except Exception as e:
        app.logger.error(f'Error in /get: {e}')
        return jsonify({'status': 'failure', 'error': str(e)}), 500

@app.route('/put-model', methods=['POST'])
@limit_remote_addr()
def write():
    try:
        file = request.files['model']
        with lock:
            file.save('model.json')
        return jsonify({'status': 'success'}), 200
    except Exception as e:
        app.logger.error(f'Error in /put: {e}')
        return jsonify({'status': 'failure', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

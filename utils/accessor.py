import json
import os

import torch


def load_json(filename):
    with open(filename, "r") as json_file:
        return json.load(json_file)


def save_json(content, filename):
    with open(filename, "w") as json_file:
        json.dump(content, json_file)


def save_model(model, path):
    print('Model saved to path: {}'.format(path))
    torch.save(model.state_dict(), path)


def load_model(model, path):
    print('Model loaded from path: {}'.format(path))
    model.load_state_dict(torch.load(path))


def create_file(path):
    if not os.path.exists(path):
        f = open(path, 'w')
        f.close()

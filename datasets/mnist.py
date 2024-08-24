#!/usr/bin/env python3

import numpy as np
import os

def load(dir: str = "."):
  training_images = _convert_images(
    _read_ubyte(os.path.join(dir, "train-images.idx3-ubyte"), 16))
  training_labels = _convert_labels(
    _read_ubyte(os.path.join(dir, "train-labels.idx1-ubyte"), 8))
  training_data = list(zip(training_images, training_labels))

  test_images = _convert_images(
    _read_ubyte(os.path.join(dir, "t10k-images.idx3-ubyte"), 16))
  test_labels = _convert_labels(
    _read_ubyte(os.path.join(dir, "t10k-labels.idx1-ubyte"), 8))
  test_data = list(zip(test_images, test_labels))

  return training_data, test_data

def _read_ubyte(fn: str, offset: int) -> np.array:
  with open(fn, "rb") as f:
    data = f.read()
  data = data[offset:]
  return np.frombuffer(data, dtype=np.uint8)

def _convert_labels(labels: np.array) -> list:
  size = 1
  count = len(labels) // size

  def to_vector(i: int) -> np.array:
    v = np.zeros((10, 1))
    v[i] = 1.0
    return v

  return [to_vector(l) for l in labels.reshape(count, size, 1)]

def _convert_images(images: np.array) -> list:
  size = 28 * 28
  count = len(images) // size
  return [i.astype(np.float32) / 255. for i in images.reshape(count, size, 1)]


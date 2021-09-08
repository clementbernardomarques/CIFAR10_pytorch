import torch
from PIL import Image
import torchvision.transforms as transforms
from scipy.special import softmax

import numpy as np
import argparse

import onnxruntime as rt


def inference(path_im):
    # Loading model for inference
    sess = rt.InferenceSession("./best_model/model_CIFAR10.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    # classes of CIFAR 10
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Loading image as RGB
    img = Image.open(path_im).convert('RGB')

    # Tensorization
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)

    # Resize as 32x32 image
    resize = transforms.Resize([32, 32])
    img = resize(img)

    # Normalization  of the image
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    img = normalize(img)

    # Reshape to fit with the input expected
    im = torch.reshape(img, (-1, 3, 32, 32))

    # Prediction of the model
    pred = sess.run([label_name], {input_name: np.array(im).astype(np.float32)})[0]

    # Organization of the results to be understood by normal human being
    pred = softmax(pred)
    pred_class = classes[np.argmax(pred)]
    score = round(float(pred.max()) * 100, 1)

    print("The result is {} with {}% as score".format(pred_class, score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--PATH", required=True)
    args = parser.parse_args()

    inference(path_im=args.PATH)

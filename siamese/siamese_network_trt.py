import cv2
import torch
import numpy as np
from PIL import Image

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from torchvision import transforms

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

class SiameseNetworkTRT:
    def __init__(self, backbone="resnet18", feed_shape=(224, 224)):
        self.context = None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(feed_shape)
        ])

    def load_model(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.device_input1, self.device_input2 = [None] * 2
        for binding in self.engine:
            if self.engine.binding_is_input(binding):  # we expect only one input
                input_shape = self.engine.get_binding_shape(binding)
                input_size = trt.volume(input_shape) * self.engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
                if self.device_input1 is None:
                    self.device_input1 = cuda.mem_alloc(input_size)
                elif self.device_input2 is None:
                    self.device_input2 = cuda.mem_alloc(input_size)
                else:
                    raise Exception("Network expects more than 2 inputs.")
            else:  # and one output
                self.output_shape = self.engine.get_binding_shape(binding)
                # create page-locked memory buffers (i.e. won't be swapped to disk)
                self.host_output = cuda.pagelocked_empty(trt.volume(self.output_shape) * self.engine.max_batch_size, dtype=np.float32)
                self.device_output = cuda.mem_alloc(self.host_output.nbytes)

        # Create a stream in which to copy inputs/outputs and run inference.
        self.stream = cuda.Stream()

    def predict(self, image1, image2, preprocess=True):
        '''
        Returns the similarity value between two images.

            Parameters:
                    image1 (np.array): Raw first image that is read using cv2.imread
                    image2 (np.array): Raw second image that is read using cv2.imread
                    preprocess (bool): Only provided for "eval_tensorrt.py". Otherwise always true when providing raw images.
            Returns:
                    output (float): Similarity of the passed pair of images in range (0, 1)
        '''
        if self.context is None:
            raise Exception(F"Context not found! Please load model first using 'load_model' function on this object.")

        if preprocess:
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

            image1 = Image.fromarray(image1).convert("RGB")
            image2 = Image.fromarray(image2).convert("RGB")

            image1 = self.transform(image1).float().numpy()
            image2 = self.transform(image2).float().numpy()

        image1 = image1.astype(np.float32)
        image2 = image2.astype(np.float32)

        cuda.memcpy_htod_async(self.device_input1, image1, self.stream)
        cuda.memcpy_htod_async(self.device_input2, image2, self.stream)

        self.context.execute_async(bindings=[int(self.device_input1), int(self.device_input2), int(self.device_output)], stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_output, self.device_output, self.stream)
        self.stream.synchronize()

        output_data = torch.Tensor(self.host_output).reshape(self.engine.max_batch_size, self.output_shape[0])
        return output_data[0][0].item()
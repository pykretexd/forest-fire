import cv2
import numpy as np
from openvino import AsyncInferQueue, Core
import openvino.runtime.properties as props


class Model:
    def __init__(self):
        self.core = Core()
        self.config = {props.hint.performance_mode(
        ): props.hint.PerformanceMode.LATENCY}

        model_xml_path = '../model/saved_model.xml'
        self.compiled_model = self.core.compile_model(
            model=model_xml_path,
            device_name='CPU',
            config=self.config
        )
        self.__result = None

        # Expected shape: [B, H, W, C]
        self.input_layer = self.compiled_model.input(0)
        self.height = int(self.input_layer.partial_shape[1].to_string())
        self.width = int(self.input_layer.partial_shape[2].to_string())

    def create_infer_queue(self):
        infer_queue = AsyncInferQueue(self.compiled_model, 0)
        infer_queue.set_callback(self._callback)
        return infer_queue

    def set_result(self, result):
        self.__result = result

    def get_result(self):
        result = self.__result
        self.clear_result()
        return result

    def clear_result(self):
        self.__result = None

    def preprocess(self, frame):
        resized_frame = cv2.resize(frame, (self.width, self.height))
        return np.expand_dims(resized_frame, axis=0).astype(np.float32)

    def postprocess(self, res):
        ANSWERS = ['Fire', 'No fire']
        fire = ANSWERS[np.argmax(res)]

        fire_prob, no_fire_prob = res
        prob = max(fire_prob, no_fire_prob) - min(fire_prob, no_fire_prob)
        return fire, prob

    def _callback(self, infer_request, user_data):
        res = infer_request.get_output_tensor(0).data.flatten()
        self.set_result(self.postprocess(res))

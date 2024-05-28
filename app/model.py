import cv2
import numpy as np
from openvino.runtime import AsyncInferQueue, Core
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
        self.infer_queue = self.create_infer_queue()

    def create_infer_queue(self):
        infer_queue = AsyncInferQueue(self.compiled_model, 0)
        infer_queue.set_callback(self._callback)
        return infer_queue

    def predict(self, frame):
        model_input = self.preprocess(frame)
        self.infer_queue.start_async({self.input_layer: model_input})
        self.infer_queue.wait_all()
        fire, prob = self.get_result()
        return fire, prob

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
        index = np.argmax(res)
        fire = ANSWERS[index]
        prob = res[index]
        return fire, prob

    def _callback(self, infer_request, user_data):
        res = infer_request.get_output_tensor(0).data.flatten()
        self.set_result(self.postprocess(res))

class InferencePipeline:
    def __init__(self, model, preprocessor, postprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def run(self, input_data):
        processed_data = self.preprocessor.process(input_data)
        model_output = self.model.predict(processed_data)
        final_output = self.postprocessor.process(model_output)
        return final_output
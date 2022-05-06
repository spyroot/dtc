from tacotron2.model_specs.model_spec import ModelSpec


class DTC(ModelSpec):
    def __init__(self, transcoder_spec, generator_spec):
        self.transcoder_spec = transcoder_spec
        self.generator_spec  = generator_spec

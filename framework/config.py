import yaml
import os

class Config:
    def __init__(self, filename):
        self.filename = filename
        self.__parse_yaml(filename)
        assert "training" in self.config
        assert "model" in self.config
        assert "input_features" in self.config

        self.__getRuntimeInfo()

    def __parse_yaml(self, filename):
        with open(filename) as fp:
            self.config = yaml.load(fp)
        self.config['runtime'] = {}

    def __getRuntimeInfo(self):
        self.runtime['model_dir'] = os.path.dirname(os.path.realpath(self.filename))

    def __getitem__(self, item):
        return self.config[item]

    @property
    def training(self):
        return self.config.get('training', None)

    @property
    def model(self):
        return self.config.get('model', None)

    @property
    def input_features(self):
        return self.config.get('input_features', None)

    @property
    def runtime(self):
        return self.config.get('runtime', None)

if __name__=="__main__":
    config = Config("scrap/hparam.yaml")
    for k,v in config.config.items():
        print(k,v)
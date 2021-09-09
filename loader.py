import yaml
import json


class Loader:
    def __init__(self, input_obj):
        if isinstance(input_obj, dict):
            input_dict = input_obj
        else:
            input_dict = input_obj.__dict__
        custom_params = {k:v for k,v in input_dict.items() if v is not None}

        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)[input_dict['config']]
        
        combined_dict = {**config, **custom_params}
        self._generate_loader(combined_dict)
        self.save_params()

    def _generate_loader(self, combined_dict):
        for key, val in combined_dict.items():
            if isinstance(val, (int, str, list, dict, float, tuple)):
                setattr(self, key, val)

    def save_params(self):
        with open('checkpoint/params_{}.json'.format(self.name), 'w') as f:
            json.dump(self.__dict__, f)


def load_params(path):
    with open(path, 'r') as f:
        params = json.load(f)

    return Loader(params)
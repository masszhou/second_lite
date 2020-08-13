import yaml
import inspect
import copy

from typing import Tuple, Dict, Callable


def parse_cfg(path: str) -> Dict:
    with open(path, "r") as f:
        class_cfg = yaml.safe_load(f.read())
    return class_cfg


def update_detect_range(old_cfg: Dict, new_xy_range: Tuple[int, int, int, int]) -> Dict:
    """
    @param: new_xy_range: [xmin, ymin, xmax, ymax]
    """
    new_cfg = copy.deepcopy(old_cfg)

    def replace_range(d):
        for k, v in d.items():
            if isinstance(v, dict):
                replace_range(v)
            else:
                if "anchor_range" in d:
                    d["anchor_range"][:2] = new_xy_range[:2]
                    d["anchor_range"][3:5] = new_xy_range[2:]
                elif "point_cloud_range" in d:
                    d["point_cloud_range"][:2] = new_xy_range[:2]
                    d["point_cloud_range"][3:5] = new_xy_range[2:]

    replace_range(new_cfg)
    return new_cfg


def find_params(functor: Callable, param_dict: Dict):
    filtered_param = {k: v for k, v in param_dict.items()
                      if k in [p.name for p in inspect.signature(functor).parameters.values()]}
    return filtered_param

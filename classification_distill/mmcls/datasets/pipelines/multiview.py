from ..builder import PIPELINES
from .compose import Compose
import copy


@PIPELINES.register_module()
class MultiBranch(object):
    def __init__(self, **transform_group):
        self.transform_group = {k: Compose(v)
                                for k, v in transform_group.items()}

    def __call__(self, results):
        multi_results = dict()
        for k, v in self.transform_group.items():
            res = v(copy.deepcopy(results))
            if res is None:
                return None
            # res["img_metas"]["tag"] = k
            multi_results[k] = res
        return multi_results

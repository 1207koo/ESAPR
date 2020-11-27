from .base import BaseModel
from utils import all_subclasses
from utils import import_all_subclasses
import_all_subclasses(__file__, __name__, BaseModel)

MODELS = {c.code():c
		  for c in all_subclasses(BaseModel)
		  if c.code() is not None}


def model_factory(args, num_items = None):
	model = MODELS[args.model_code]
	return model(args)

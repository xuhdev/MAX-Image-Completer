import tensorflow as tf
import logging
from PIL import Image
from config import DEFAULT_MODEL_PATH, MODEL_META_DATA as model_meta
from maxfw.model import MAXModelWrapper
import os

from core.model_DCGAN import DCGAN

logger = logging.getLogger()

#argument dictionary
args = {"approach": "adam",
        "lr":0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "hmcEps": 0.001,
        "hmcBeta": 0.2,
        "hmcL": 100,
        "hmcAnneal": 1,
        "nIter":1000,
        "imgSize":64,
        "lam":0.1,
        "checkpointDir":"assets/checkpoint",
        "outDir":"/workspace/assets/center_mask",
        "outInterval":50,
        "maskType":"center",
        "centerScale": 0.25,
        "imgs":''
        }

class ModelWrapper(MAXModelWrapper):
    """Model wrapper for TensorFlow models in SavedModel format"""

    MODEL_META_DATA = model_meta

    def __init__(self, path=DEFAULT_MODEL_PATH):
        logger.info('Loading model from: {}...'.format(path))
       
        # checking for checkpoint directory
        checkpointDir = args["checkpointDir"]
        assert(os.path.exists(checkpointDir))
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        

    def predict(self, x, image_mask_type):
        
        checkpointDir = args["checkpointDir"]
        assert(os.path.exists(checkpointDir))
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:
            dcgan = DCGAN(sess, image_size=args["imgSize"],batch_size=1,
                          checkpoint_dir=checkpointDir, lam=0.1)
            args["imgs"] = x
            args["maskType"] = image_mask_type
          
            dcgan.complete(args)

        
        #Read output directory
        
        assert(os.path.exists(args["outDir"]))
        output_image_path = args["outDir"] + '/completed/*'
        
        return output_image_path

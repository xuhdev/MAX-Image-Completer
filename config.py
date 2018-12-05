# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False

# Application settings

# API metadata
API_TITLE = 'Model Asset Exchange Server'
API_DESC = 'An API for serving models'
API_VERSION = '0.1'

# default model
MODEL_NAME = 'DCGAN Image Completer Model'
DEFAULT_MODEL_PATH = 'assets/{}'.format(MODEL_NAME)
MODEL_LICENSE = 'MIT'

MODEL_META_DATA = {
    'id': 'DCGAN',
    'name': 'DCGAN Image Completer Model',
    'description': 'DCGAN Image completion model; trained using Tensorflow on CelebA dataset',
    'type': 'image-completion',
    'license': '{}'.format(MODEL_LICENSE)
}

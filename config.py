# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False

# Application settings

# API metadata
API_TITLE = 'MAX Image Completer'
API_DESC = 'Recognize and extract faces in an image and complete the corrupted portions.'
API_VERSION = '1.1.0'

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

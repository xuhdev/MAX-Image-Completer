import io
import os
import glob
import shutil

from flask_restplus import fields
from werkzeug.datastructures import FileStorage
from flask import make_response, abort
from PIL import Image
import re

from maxfw.core import MAX_API, PredictAPI
from core.model import ModelWrapper
from api.pre_process import alignMain


# Creating a JSON response model: https://flask-restplus.readthedocs.io/en/stable/marshalling.html#the-api-model-factory
label_prediction = MAX_API.model('LabelPrediction', {
    'label_id': fields.String(required=False, description='Label identifier'),
    'label': fields.String(required=True, description='Class label'),
    'probability': fields.Float(required=True)
})


# Set up parser for input data (http://flask-restplus.readthedocs.io/en/stable/parsing.html)
input_parser = MAX_API.parser()
# Example parser for file input
input_parser.add_argument('file', type=FileStorage, location='files', required=True, help='An image file (encoded as PNG or JPG/JPEG)')
input_parser.add_argument('mask_type', type=str, default='center', required=True,
             choices=('random', 'center', 'left', 'grid'),
             help='Available options for mask_type are random, center, left and grid. ')


class ModelPredictAPI(PredictAPI):

    model_wrapper = ModelWrapper()
    
    @MAX_API.doc('predict')
    @MAX_API.expect(input_parser)
    def post(self):
        """Make a prediction given input data"""

        args = input_parser.parse_args()
        
        if not args['file'].mimetype.endswith(('jpg', 'jpeg', 'png')):
            abort(400, 'Invalid file type/extension. Please provide an image in JPEG or PNG format.')
            
        image_input_read = Image.open(args['file'])
        image_mask_type = args['mask_type']
        # creating directory for storing input
        input_directory = '/workspace/assets/input/file'
        if not os.path.exists(input_directory):
            os.mkdir(input_directory)
        # clear input directory
        input_folder = '/workspace/assets/input/file/'
        for file in glob.glob(input_folder + '*'):
            try:
                try:
                    os.remove(file)
                except:
                    shutil.rmtree(file)
            except:
                continue
        # save input image
        image_input_read = image_input_read.convert('RGB')
        image_input_read.save('/workspace/assets/input/file/input.jpg')
        # face detection, alignment and resize using openface
        args = {
               'inputDir':input_directory,
               'outputDir':'/workspace/assets/input/file/align',
               'landmarks':'innerEyesAndBottomLip',
               'dlibFacePredictor':'/workspace/openface/models/dlib/shape_predictor_68_face_landmarks.dat',
               'verbose':True,
                'size':64,
                'skipMulti':False,
                'fallbackLfw':None,
                'mode':'align'
               }
        try:
            coordinates = alignMain(args)
            coordinates_string = str(coordinates)
            pattern='^\s*\[\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*\]\s*$'
            m = re.match(pattern, coordinates_string)
            if m:
                final_coordinates = '[[{},{}],[{},{}]]'.format(m.group(1), m.group(2), m.group(3),m.group(4))
            
        except:
            #abort if there face is not detected
            abort(400, 'No face was detected in the image.')

        # store aligned input
        input_data = '/workspace/assets/input/file/align/file/input.png'
        #
        image_path = self.model_wrapper.predict(input_data, image_mask_type)
        """
        preparing image collage
        """
        new_collage_path = "/workspace/assets/center_mask/completed/Collage.jpg"
        img_columns = 5
        img_rows = 4
        img_width = 320
        img_height = 256
        thumbnail_width = img_width//img_columns
        thumbnail_height = img_height//img_rows
        size = thumbnail_width, thumbnail_height
        new_collage = Image.new('RGB', (img_width, img_height))
        images_list = []
        filenames = []
        for filename in glob.glob(image_path):
            filenames.append(filename)
        
        filenames.sort()
        for i in filenames:
            im=Image.open(i)
            im.thumbnail(size)
            images_list.append(im)
        
        i = 0
        x = 0
        y = 0
        for col in range(img_columns):
            for row in range(img_rows):
                new_collage.paste(images_list[i], (x, y))
                i += 1
                y += thumbnail_height
            x += thumbnail_width
            y = 0

        new_collage.save(new_collage_path)
                
        """
        end of collage creation process
        """
        img = Image.open(new_collage_path, mode='r')
        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format='JPEG')
        imgByteArr = imgByteArr.getvalue()

        response = make_response(imgByteArr)
        response.headers.set('Content-Type', 'image/jpeg')
        response.headers.set('Content-Disposition', 'attachment', filename='result.jpg')
        response.headers.set('coordinates', final_coordinates)

        return response

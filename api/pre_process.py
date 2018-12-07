import openface
from openface.openface import helper
from openface.openface.data import iterImgs
import cv2
import os
import random
import shutil



def alignMain(args):
    helper.mkdirP(args['outputDir'])

    imgs = list(iterImgs(args['inputDir']))

    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)

    landmarkMap = {
        'outerEyesAndNose': openface.openface.AlignDlib.OUTER_EYES_AND_NOSE,
        'innerEyesAndBottomLip': openface.openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    }
    if args['landmarks'] not in landmarkMap:
        raise Exception("Landmarks unrecognized: {}".format(args['landmarks']))

    landmarkIndices = landmarkMap[args['landmarks']]

    align = openface.openface.AlignDlib(args['dlibFacePredictor'])
    
    nFallbacks = 0
    
    outDir = os.path.join(args['outputDir'], imgs[0].cls)
    helper.mkdirP(outDir)
    outputPrefix = os.path.join(outDir, imgs[0].name)
    imgName = outputPrefix + ".png"
        
    if os.path.isfile(imgName):
        if args['verbose'] == True:
            os.remove(imgName)
            print("  + Already found, skipping.")
    else:
        rgb = imgs[0].getRGB()
        if rgb is None:
            if args['verbose'] == True:
                print("  + Unable to load.")
            outRgb = None
        else:
            
            coordinates = align.getLargestFaceBoundingBox(rgb)
            
            outRgb = align.align(args['size'], rgb,
                                    landmarkIndices=landmarkIndices,
                                    skipMulti=args['skipMulti'])
            if outRgb is None and args['verbose'] == True:
                print("  + Unable to align.")

        if args['fallbackLfw'] == None and outRgb is None:
            nFallbacks += 1
            deepFunneled = "{}/{}.jpg".format(os.path.join(args['fallbackLfw'],
                                                               imgs[0].cls),
                                                  imgs[0].name)
            shutil.copy(deepFunneled, "{}/{}.jpg".format(os.path.join(args['outputDir'],
                                                                          imgs[0].cls),
                                                             imgs[0].name))

        if outRgb is not None:
            if args['verbose'] == True:
                print("  + Writing aligned file to disk.")
            outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
            print('writing')
            cv2.imwrite(imgName, outBgr)
            return coordinates

    if args['fallbackLfw'] == None:
        print('nFallbacks:', nFallbacks)





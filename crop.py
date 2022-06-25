import cv2
import numpy
import sys
import os
import time

from numpy import pi
from PIL import Image


startTime = time.time()


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CWHITE = '\33[37m'
    RESET = '\033[0m'


def elapsedTime():
    rawTime = time.time() - startTime
    formatTime = round(rawTime, 1)
    strTime = str(formatTime)
    return strTime


newPath = sys.argv[1].rsplit('/', 1)[0]


def generateCropTile(imageIn, tiles):
    for tile in tiles:
        generateTile = imageIn.crop(tile['box'])

        generateTile.save(newPath + '/tiles/' +
                          'tile-' + tile['name'] + '.png')

        print(bcolors.OKGREEN + '[' + elapsedTime() +
              ' INFO]' + bcolors.RESET + ' Created tile: ' + 'tile-' + tile['name'])


imgIn = Image.open(sys.argv[1])
inSize = imgIn.size

os.mkdir(newPath + '/tiles')

print(bcolors.OKGREEN + '[' + elapsedTime() +
      ' INFO]' + bcolors.RESET + ' Created new path: ' + newPath + '/tiles')

tileWidth = inSize[0] / 4
tiles = [{
    "name": "back",
    "box": (0, tileWidth, tileWidth, 2*tileWidth)
},
    {
    "name": "left",
    "box": (tileWidth, tileWidth, 2*tileWidth, 2*tileWidth)
},
    {
    "name": "front",
    "box": (2*tileWidth, tileWidth, 3*tileWidth, 2*tileWidth)
},
    {
    "name": "right",
    "box": (3*tileWidth, tileWidth, 4*tileWidth, 2*tileWidth)
},
    {
    "name": "up",
    "box": (2*tileWidth, 0, 3*tileWidth, tileWidth)
},
    {
    "name": "down",
    "box": (2*tileWidth, 2*tileWidth, 3*tileWidth, 3*tileWidth)
}
]

Tile = generateCropTile(imgIn, tiles)

# imgOut = Image.fromarray(cubemap)
# backTile.save(sys.argv[1].rsplit('/', 1)[0] + "/cubemap.png")

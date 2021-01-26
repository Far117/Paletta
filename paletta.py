from enum import Enum
from math import sqrt
from sys import argv
import random

from PIL import Image

class ColorSpace(Enum):
    RGB = 0
    XYZ = 1
    LAB = 2

class KMCSimulation:
    dataset = []
    clusters = {} # A dictionary of centroids->the set of points in that cluster

    dis = 0 # dissimilarity
    
    def __init__(self, numClusters, dataset):
        self.numClusters = numClusters
        self.dataset = dataset

        initialCentroids = random.sample(dataset, numClusters) # avoids picking the same pixel more than once
        for centroid in initialCentroids:
            self.clusters[centroid] = set()
            self.clusters[centroid].add(centroid)
            
        return

    def recalculateCentroids(self):
        centroidMoved = False

        toChange = set()
        
        for oldCentroid, cluster in self.clusters.items():
            newCentroid = self.findCentroid(cluster)
            
            if newCentroid != oldCentroid:
                centroidMoved = True
                toChange.add((oldCentroid, newCentroid))

        for oldCentroid, newCentroid in toChange:
            self.clusters[newCentroid] = self.clusters.pop(oldCentroid)

        return centroidMoved

    def recluster(self):
        centroids = self.clusters.keys()
        #print(f"Current centroids: {centroids}")

        for centroid in centroids:
            self.clusters[centroid] = set()

        for pixel in self.dataset:
            closestCentroid = min(map(lambda c: (self.distance(c, pixel), c), centroids), key = lambda pair: pair[0])[1]
            self.clusters[closestCentroid].add(pixel)
            #print(f"The pixel {pixel} was closest to the centroid {closestCentroid}")

        return    

    def iterate(self):

        print("Running KMC...")
        
        again = True
        passes = 0
        while again:
            self.recluster()
            again = self.recalculateCentroids()
            passes += 1
            print(f"Iterations: {passes}")

        #print("Done. Rounding final values...")

        #toChange = set()
        #for r, g, b in self.clusters.keys():
        #    newR = round(r)
        #    newG = round(g)
        #    newB = round(b)

        #    toChange.add(((r, g, b), (newR, newG, newB)))

        #for oldCentroid, newCentroid in toChange:
        #    self.clusters[newCentroid] = self.clusters.pop(oldCentroid)

        print("Done!")
        return self.clusters

    def calculateDissimilarity(self):
        self.dis = self.dissimilarity(self.clusters)

        return self.dis
    
    def getDissimilarity(self):
        return self.dis

    def findCentroid(self, cluster):
        '''Finds the center of a cluster of pixles.'''
        
        center = [0, 0, 0]
        for pixel in cluster:
            r, g, b = pixel
            center[0] += r
            center[1] += g
            center[2] += b

        size = len(cluster)
        center = tuple(map(lambda e: e / size, center))

        return center

    def distance(self, point1, point2):
        '''Calculates the Euclidean distance between the two pixels.'''

        return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

    def distanceSquared(self, point1, point2):
        '''Calculates the Euclidean distance between the two pixels, squared.
           This is more efficient than taking the square root and then undoing it.'''

        return (point1[0] + point2[0])**2 + (point1[1] + point2[1])**2 + (point1[2] + point2[2])**2

    def variability(self, centroid, cluster):
        '''Calculates the variability of a given cluster of pixels.'''
    
        var = 0

        for pixel in cluster:
            var += distanceSquared(centroid, pixel)

        return var

    def dissimilarity(self, clusters):
        '''Calculates the dissimilarity of a goup of clusters of pixels'''

        dis = 0

        for centroid, cluster in clusters.items():
            dis += self.variability(centroid, cluster)

        return dis

def collapsePixel(pixel, palette):
    for centroid, cluster in palette.items():
        if applyColorSpaceTransform(rgbToXyz, pixel) in cluster:
            return centroid

def tupleDot(t1, t2):
    '''Returns the dot product of two 3-tuples'''
    
    return t1[0] * t2[0] + t1[1] * t2[1] + t1[2] * t2[2]
        
def applyColorSpaceTransform(matrix, values):
    '''Given a color space transformation matrix, multiplies it by the given
       3-tuple to convert a pixel's color from one space to another. For example,
       RGB->XYZ'''
    
    return tuple(map(lambda row: tupleDot(row, values), matrix))

# Courtesy of http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
# and http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html

# D65 White point
whitepointXYZ = (0.95047, 1, 1.08883)
class ColorTransform:
    
    @staticmethod
    def rgbToXyz(pixel):

        # Gamma correction. See https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
        v = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055)**2.4
        
        r, g, b = pixel
        r = v(r / 255)
        g = v(g / 255)
        b = v(b / 255)

        
        
        x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

        return (x, y, z)

    @staticmethod
    def xyzToRgb(pixel):

        # Inverse gamma correction. See https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
        vi = lambda c: c * 12.92 if c <= 0.0031308 else 1.055 * c**(1/2.4) - 0.055
        
        x, y, z = pixel
        r =  3.2404542 * x - 1.5371385 * y - 0.4985314 * z
        g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
        b =  0.0556434 * x - 0.2040259 * y + 1.0572252 * z

        r = vi(r) * 255
        g = vi(g) * 255
        b = vi(b) * 255

        return (r, g, b)

    @staticmethod
    def xyzToLab(pixel):
        x, y, z = pixel
        xw, yw, zw = whitepointXYZ

        xr = x / xw
        yr = y / yw
        zr = z / zw

        e = 216/24389
        k = 24389/27

        f = lambda v: v**(1/3) if v > e else (k*v + 16) / 116

        l = 116*f(yr) - 16
        a = 500*(f(xr) - f(yr))
        b = 200*(f(yr) - f(zr))

        return (l, a, b)

    @staticmethod
    def labToXyz(pixel):
        l, a, b = pixel
        xw, yw, zw = whitepointXYZ

        e = 216/24389
        k = 24389/27

        fy = (l + 16) / 116
        fx = a / 500 + fy
        fz = fy - b/200

        xr = fx**3 if fx**3 > e else (116 * fx - 16) / k
        yr = ((l + 16) / 116)**3 if l > k*e else l / k
        zr = fz**3 if fz**3 > e else (116 * fz - 16) / k
        
        x = xr * xw
        y = yr * yw
        z = zr * zw

        return (x, y, z)

    @staticmethod
    def rgbToLab(pixel):
        return ColorTransform.xyzToLab(ColorTransform.rgbToXyz(pixel))

    @staticmethod
    def labToRgb(pixel):
        return ColorTransform.xyzToRgb(ColorTransform.labToXyz(pixel))

class Palettizer:

    filename = ""
    colors = 0
    colorSpace = None

    image = None
    sim = None
    pixels = None

    results = None
    
    def __init__(self, filename, colors, colorSpace = ColorSpace.RGB):
        self.filename = filename
        self.colors = colors
        self.colorSpace = colorSpace

        self.image = Image.open(filename) # TODO: Error handling
        self.pixels = set(image.getdata())

        if self.colorSpace == ColorSpace.XYZ:
            self.pixels = set(map(ColorTransform.rgbToXyz, self.pixels))
        elif self.colorSpace == ColorSpace.LAB:
            self.pixels = set(map(ColorTransform.rgbToLab, self.pixels))

        self.sim = KMCSimulation(colors, self.pixels)

        return

    def palettize(self):
        self.results = self.sim.iterate()

        if self.colorSpace != colorSpace.RGB:
            colorTransform = ColorTransform.labToRgb if self.colorSpace == ColorSpace.LAB else ColorTransform.xyzToRgb
            rgbResults = {}
            
            for centroid, cluster in self.results.items():
               rgbResults[colorTransform(centroid)] = set(map(colorTransform, cluster))

            self.results = rgbResults
            
        self.round()

        return self

    def applyPalette(self, imageData):
        for i in range(len(imageData)):
            for centroid, cluster in self.results.items():
                if imageData[i] in cluster:
                    imageData[i] = centroid

        return imageData

    def round(self):

        rounded = {}
        roundTuple = lambda t: (round(t[0]), round(t[1]), round(t[2]))
        
        for centroid, cluster in self.results.items():
            rounded[roundTuple(centroid)] = set(map(roundTuple, cluster))

        self.results = rounded

        return self

def testSpaces(color):
    xyz = ColorTransform.rgbToXyz(color)
    xyzi = ColorTransform.xyzToRgb(xyz)

    lab = ColorTransform.xyzToLab(xyz)
    labi = ColorTransform.labToXyz(lab)

    rgbi = ColorTransform.xyzToRgb(labi)

    print(f"RGB Color:\t{color}\nRGB->XYZ:\t{xyz}\nXYZ->RGB:\t{xyzi}\nRGB->XYZ->LAB:\t{lab}\nLAB->XYZ:\t{labi}\nLAB->XYZ->RGB:\t{rgbi}")

#testSpaces((25, 50, 10))
#quit()
        
if __name__ == "__main__":

    if len(argv) != 5:
        print("Syntax: <input> <output> <number of colors> <color space (rgb/xyz/lab)>")
    else:
        filename = argv[1]
        outputname = argv[2]
        numColors = int(argv[3])
        colorSpace = argv[4].lower()

        if colorSpace == "rgb":
            colorSpace = ColorSpace.RGB
        elif colorSpace == "xyz":
            colorSpace = ColorSpace.XYZ
        elif colorSpace == "lab" or colorSpace == "l*a*b*":
            colorSpace = ColorSpace.LAB
        else:
            print(f"Unknown color space: {colorSpace}. Valid options are RGB, XYZ, and L*A*B* (LAB)")
            quit()

        image = Image.open(filename)
        
        palettizer = Palettizer(filename, numColors, colorSpace)
        image.putdata(palettizer.palettize().applyPalette(list(image.getdata())))
        image.save(outputname)
        
    quit()
quit()
    
image = Image.open("redblue.png")
pixels = set(map(lambda p: applyColorSpaceTransform(rgbToXyz, p), image.getdata()))

sim = KMCSimulation(8, pixels)
results = sim.iterate()

print(f"Dissimilarity: {sim.calculateDissimilarity()}")

newPixels = list(image.getdata())

for i in range(len(newPixels)):
    #print(collapsePixel(newPixels[i], results))
    newPixels[i] = collapsePixel(newPixels[i], results)

image.putdata(newPixels)
#image.show()
image.save("redblue8.png")

# image.show()

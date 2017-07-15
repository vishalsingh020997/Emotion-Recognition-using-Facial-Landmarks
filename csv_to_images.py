
import os
import csv
import argparse
import numpy as np 
import scipy.misc

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', required=True, help="path of the csv file")
parser.add_argument('-o', '--output', required=True, help="path of the output directory")

args = parser.parse_args()

w, h = 48, 48
image = np.zeros((h, w), dtype=np.uint8)
id = 1
with open(args.file, 'r') as csvfile:
    datareader = csv.reader(csvfile, delimiter =',')
    headers = next(datareader)
    print (headers) 
    for row in datareader:  
        emotion = row[0]
        pixels = list(map(int, row[1].split()))
        usage = row[2]
   
        pixels_array = np.asarray(pixels)

        image = pixels_array.reshape(w, h)
        image_folder = os.path.join(args.output, emotion)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        image_file =  os.path.join(image_folder , str(id) + '.jpg')
        scipy.misc.imsave(image_file, image)
        id += 1 
        if id % 100 == 0:
            print('Processed {} images'.format(id))

print("Finished processing {} images".format(id))

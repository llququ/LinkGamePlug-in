# -*- coding: utf-8 -*-
"""
Created on Tue Apr 02 09:32:25 2019

@author: wedoctor
"""
#absolute path
png_path = "/home/zhao/disk/back_dir/dataset/JPEGImages/"
xml_path = "/home/zhao/disk/back_dir/dataset/Annotations/"
img_dir = 'D:/1/data/'

import os
import xml.etree.ElementTree as ET
import os, sys, cv2, random
    
classes=['banana','blueberry','grape','peach','pineapple','strawberry']
#name = dict()


def get_objs_label(objs, size):
    label=''
    global n
    for i in objs:
        bndbox = i.find("bndbox")
        xmin = bndbox.find("xmin").text
        xmin = float(xmin)/size[0]
        xmax = bndbox.find("xmax").text
        xmax = float(xmax)/size[0]
        ymin = bndbox.find("ymin").text
        ymin = float(ymin)/size[1]
        ymax = bndbox.find("ymax").text
        ymax = float(ymax)/size[1]
        name_idx = i.find("name").text
        if name_idx not in classes:
            continue
        cls_id = classes.index(name_idx)
        #if name_idx not in name.keys():
         #   name[name_idx] = n
         #   n += 1
        label += str(cls_id) + "\t" + str(xmin) + "\t" + str(ymin) + "\t" + str(xmax) + "\t" + str(ymax) + "\t"
    return label

def get_lst_line(idx, filename):  
    #idx = line.split()[0]
    #filename = line.split()[2]
    xmlfile = ET.parse(filename.replace("jpg", "xml"))
    width = xmlfile.find("size").find("width").text
    height = xmlfile.find("size").find("height").text
    objs = xmlfile.findall("object")
    #print filename, len(objs)
    label = str(idx) + "\t" + "4\t5\t" + str(width) + "\t" + str(height) + "\t" + str(get_objs_label(objs, [float(width), float(height)])) + str(filename) +"\n"
    return label

#新建lst,填入get_lst_line()
file_names = os.listdir(img_dir)
file_names.sort()
file_num = int(len(file_names) / 2)

for idx in range(file_num):
    #img = cv2.imread(img_dir + file_names[2 * idx])
    filename = img_dir + file_names[2 * idx]
    with open("D:/1/test.lst", "a") as f:
        label = get_lst_line(idx, filename)
        f.write(label)

           
                
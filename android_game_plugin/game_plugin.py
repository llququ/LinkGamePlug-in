# -*- coding: utf-8 -*-
import auto_adb
import YoloModel as yolo
import sys
import subprocess
import time
import random
from mxnet import image
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision
from mxnet import cpu
from itertools import *


rgb_mean = nd.array([123, 117, 104])
rgb_std = nd.array([58.395, 57.12, 57.375])
data_shape = 448
scales = [[3.3004, 3.59034],
          [9.84923, 8.23783]]
class_names = ['banana','blueberry','grape','peach','pineapple','strawberry']
num_class = 6
width = 1440
heigh = 2560
ctx = cpu()

adb = auto_adb.auto_adb()
def pull_screenshot():
    process = subprocess.Popen('adb shell screencap -p', shell=True, stdout=subprocess.PIPE)
    screenshot = process.stdout.read()
    if sys.platform == 'win32':
        screenshot = screenshot.replace(b'\r\n', b'\n')
    f = open('autojump.png', 'wb')
    f.write(screenshot)
    f.close()


def tap_onscreen(x, y):
    process = subprocess.Popen('adb shell input tap %s %s' % (x, y), shell=True, stdout=subprocess.PIPE)

def yolo_network():
    batch_size = 4

    pretrained = vision.get_model('resnet18_v1', pretrained=True).features
    net = nn.HybridSequential()
    for i in range(len(pretrained) - 2):
        net.add(pretrained[i])


    # use 2 classes, 1 as dummy class, otherwise softmax won't work
    predictor = yolo.YOLO2Output(num_class, scales)
    predictor.initialize()
    net.add(predictor)

    net.load_parameters('C:/Users/tao/Documents/500epoch.params', ctx=ctx)
    return net
def process_image(fname):
    with open(fname, 'rb') as f:
        im = image.imdecode(f.read())
    # resize to data_shape
    data = image.imresize(im, data_shape, data_shape)
    # minus rgb mean, divide std
    data = (data.astype('float32') - rgb_mean) / rgb_std
    return data.transpose((2,0,1)).expand_dims(axis=0), im

    # convert to batch x channel x height xwidth
def predict(x):
    net = yolo_network()
    x = net(x)
    output, cls_prob, score, xywh = yolo.yolo2_forward(x, num_class, scales)
    return nd.contrib.box_nms(output.reshape((0, -1, 6)))

def linlink(point1, point2, out):
    x1 = int((point1[2] + point1[4]) / 2 * width)
    y1 = int((point1[3] + point1[5]) / 2 * heigh)
    x2 = int((point2[2] + point2[4]) / 2 * width)
    y2 = int((point2[3] + point2[5]) / 2 * heigh)
    ret = False

    result = lineCase(x1,y1,x2,y2,out)
    if result:
        # print 1
        ret = True
    else:
        result = onceCorner(x1,y1,x2,y2, out)
        if result:
            # print 2
            ret = True
        else:
            result = doubleCorner(x1,y1,x2,y2, out)
            if result:
                # print 3
                ret = True
            else:
                pass
    return ret
#直连
def lineCase(x1,y1,x2,y2,out, corner=False):
    ret = False
    flag = True
    if abs(x1 - x2) < 110:
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        y_distance = abs(y1 - y2)
        if y_distance < 360 and corner == False:
            ret = True
        elif y_distance < 360 and corner:
            ret = check_point(x2, y2, out)
        else:
            for i in range(ymin+240, ymax, 240):
                result = check_point(x1, i, out)
                if result == False:
                    flag = False
                    break
            if flag != False:
                ret = True

    if abs(y1 - y2) < 110:
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        x_distance = abs(x1 - x2)
        if x_distance < 360 and corner == False:
            ret = True
        elif x_distance < 360 and corner:
            ret = check_point(x2, y2, out)
        else:
            for i in range(xmin+240, xmax, 240):
                result = check_point(i, y1, out)
                if result == False:
                    flag = False
                    break
            if flag != False:
                ret = True
    return ret

#一次拐角
def onceCorner(x1,y1,x2,y2, out):
    ret = False
    flag1 = lineCase(x1,y1,x2,y1, out, True)
    flag2 = lineCase(x2,y2,x2,y1, out, True)
    if flag1 and flag2:
        ret = True
    flag3 = lineCase(x1,y1,x1,y2,out,True)
    flag4 = lineCase(x2,y2,x1,y2,out,True)
    if flag3 and flag4:
        ret = True
    return ret

#二次拐角
def doubleCorner(x1,y1,x2,y2, out):
    ret = False
    for px in range(-1,2):
        break_all = False
        for py in range(-1,2):
            tmpx = x1 + 240*px
            tmpy = y1 + 240*py
            if check_point(tmpx, tmpy, out)==False:
                continue
            if (tmpx == x1 and tmpy != y1) or (tmpy == y1 and tmpx != x1):
                line = lineCase(x1,y1,tmpx,tmpy,out,True)
                once = onceCorner(tmpx,tmpy,x2,y2, out)
                if line and once:
                    ret = True
                    break_all = True
                    break
        if break_all:
            break
    return ret

#检查x,y点，如果存在方块则为false,不存在为true
def check_point(x, y, points):
    ret = True
    for item in points:
        item = item.asnumpy()
        class_id, score = int(item[0]), item[1]
        if class_id < 0 or score < 0.85:
            continue
        if x > item[2]* width and x < item[4]* width and y > item[3]* heigh and y < item[5]* heigh:
                ret = False
    return ret


def tap(output, threshold):
    lists = [[] for i in range(6)]
    match_num = 0
    flag = False
    for out in output:
        out = out.asnumpy()
        class_id, score = int(out[0]), out[1]
        if class_id < 0 or score < threshold:
            continue
        lists[class_id].append(out)

    random.shuffle(lists)
    # for list in lists:
    #     flag = tap_list(list, box_num)
    #     if flag == True:
    #         return

    for sameList in lists:
        for points in combinations(sameList, 2):
            if points[0][0] == -1 or points[1][0] == -1:
                continue

           # tic = time.time()
            llk = linlink(points[0],points[1],output)
          #  print 'lin %.1f sec' % (time.time() - tic)

            if llk:
                match_num = match_num+1
                points[0][0] = -1
                points[1][0] = -1
                x1 = (points[0][2] + points[0][4]) / 2 * width
                y1 = (points[0][3] + points[0][5]) / 2 * heigh
                x2 = (points[1][2] + points[1][4]) / 2 * width
                y2 = (points[1][3] + points[1][5]) / 2 * heigh

                time.sleep(0.3)
                tap_onscreen(x1, y1)
                time.sleep(0.3)
                tap_onscreen(x2, y2)

        #     if match_num == 4:
        #         flag = True
        #         break
        # if flag == True:
        #     break
        # 重新截图：
        # new_box = new_box_num(0.85)
        # if (new_box  < box_num):
        #     print '新方框数 %s,旧方框数 %s' % (new_box, box_num)
        #     print '对比数量不相等，重新截图'
        #     return


def main():
    while True:
        print '开始截图'
        tic = time.time()
        time.sleep(0.5)
        pull_screenshot()
        print '%.1f sec' % (time.time() - tic)
        print '截图完成'
        x, im = process_image('C:/Users/tao/PycharmProjects/android_game_plugin/autojump.png')
        print '预测'
        out = predict(x.as_in_context(ctx))
        tap(out[0], 0.85)

if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        adb.run('kill-server')
        print('bye')
        exit(0)








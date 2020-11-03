# -*- coding: UTF-8 -*-
import base64
import requests
import json
import sys
# sys.path.append('G:\\爬虫\\py_aiplat_demo\\SDK')
import apiutil
import os
import time

'''
人脸分析打分
'''
App_ID = 'ai提供的id'
App_Key = 'ai提供的key'


def ai_image(image_data):
    time.sleep(0.5)
    ai_obj = apiutil.AiPlat(App_ID, App_Key)
    print('-----------')
    rsp = ai_obj.getRenlianFenxi(image_data)
    if rsp['ret'] == 0:
        for i in rsp['data']['face_list']:
            print(i['beauty'])
        print('----')
        return int(i['beauty'])
    else:
        # print('无返回')
        print(rsp['ret'])
        return int(rsp['ret'])


if __name__ == '__main__':
    num_files = 0

    os.chdir('女神吧妹子图')
    path = os.getcwd()
    for i in os.listdir(path):
        num_files += 1
    print(num_files)
    for root, dirs, files in os.walk(path):
        # print(files)
        for each in files:
            f = open(root + '\\' + each, 'rb')
            ls_f = f.read()
            beauty = ai_image(ls_f)
            f.close()
            if beauty != 0:
                if beauty < 80:
                    os.remove(root + '\\' + each)
                elif beauty == 16404:
                    os.remove(root + '\\' + each)
                else:
                    print('ok')
            time.sleep(0.5)
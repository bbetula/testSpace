#将多张图片合成为一张
from PIL import Image
import os
input_path="Learning Models/picture/"
output_path="Learning Models/picture/"
num=3
name=["LS","GD","Newton"]
def picture_process(input_path,output_path):
    #将图片合成为一张 横向分布即可
    #首先找到所有图片的路径
    path_list=[]
    for i in range(num):
        path_list.append(input_path+name[i]+".png")
    #打开图片
    img_list=[Image.open(i) for i in path_list]
    #获取图片的宽度和高度
    width=img_list[0].size[0]
    height=img_list[0].size[1]
    #创建新的图片
    new_img=Image.new("RGB",(width*num,height))
    #将图片粘贴到新的图片上
    for i in range(num):
        new_img.paste(img_list[i],(i*width,0))
    new_img.save(output_path+"result.png")

picture_process(input_path,output_path)
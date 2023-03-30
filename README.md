cv2.dct报错：在 '__init__.py' 中找不到引用 'dct'
[解决方法](https://blog.csdn.net/hxm_520/article/details/121172546)
把site-packages中的cv2.pyd文件复制到其父目录中即可

测试图集源文件在[链接](https://www.vcl.fer.hr/comofod/download.html)处下载，本程序在该数据集上选取了部分图片放在data文件夹下。

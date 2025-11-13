L2M + RSMPNet 实时融合版

文件内容很简单：
- run_integrated.py  主程序，直接运行这个就行  
- configs/integrated.yaml  配置文件，改路径用  
- README.txt  这份说明  

一、环境  
建议用 Python3.8 以上，PyTorch、OpenCV、NumPy、PyYAML 装上就行  
要用 GPU 的话 CUDA 11 起步  
安装命令大概这样：
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118  
pip install opencv-python numpy pyyaml  

二、配置  
打开 configs/integrated.yaml  
- 把 L2M 的模型路径（l2m.entry.path）改成自己的  
- 把 RSMPNet 的模型路径（rsmp.entry.path）也改成自己的  
- 再改下输入方式，是文件夹跑图像还是直接接摄像头  

两个模型都用预训练好的，不需要训练，也不会保存中间 npz 文件，全程在线跑  

三、输入方式  
可以两种：  
1）离线模式：放好 RGB 图片和深度 npy 文件，路径在配置里改  
2）实时相机模式：改成 camera，设置摄像头编号  

四、运行  
命令：  
python run_integrated.py configs/integrated.yaml  

想省显卡内存可以把 device 改成 cpu  
viz.show=true 会开窗口实时显示，false 就只保存结果  
fps_limit 控制帧率，默认 30  

五、结果  
运行后能看到语义结果叠加在原图上，左上角有 FPS  
如果在配置里设了 save_dir，就会自动保存每帧图像  

六、常见问题  
- 模型加载报 key mismatch：改成 strict=False 就行  
- 两边输出尺寸不对：统一改下配置里 input.h 和 input.w  
- 窗口打不开：把 viz.show 关掉  

七、说明  
这版程序是把 L2M 和 RSMPNet 串成一个完整流程，  
直接用预训练权重就能实时跑，不用导 npz，不用分两段。  
能在真机或者录好的数据上直接验证效果。

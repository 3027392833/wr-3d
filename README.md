https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h
zZ7Nw5gPV5HSXhkd
[FDK]					#--滤波反投影
#总开关
FDKEnabled	=1
RebuildMode	=0  		#0表示一层一层重建，1表示所有层一次重建	
SID		=2200				
Distance	=481.4		#平板到旋转中心距离
GridON		=0			#是否有滤线栅

AngleRange	=360		#最大扫描角度范围
View		=409		#最大扫描角度范围内的投影数	
DefautAngleEnabled = 0	#是否按默认的角度重建， 0-否， 1-是
allowRange	= 0.55		#允许重建的角度范围，

AngleCal = 0			#一圈角度矫正	
AngleOffset = 45		#初始默认角度起始值	
AngleOffsetIn = 0		#输入初始默认角度起始值	

starinset = 1
endinset = 820

Width		=1024		#原始图大小
Height		=2900 
PixelSpaceW	=0.417		#图像素间距
PixelSpaceH	=0.417

MidChannel		=512	#中心通道
MidRow			=1450	#中心层

Offset		=400		#基础图像有400的偏移需要去掉
FPS		=12				#透视帧率

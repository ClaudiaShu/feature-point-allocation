import numpy as np 
import os
import pickle
import cv2
import xml.etree.cElementTree as ET
import operator



test_pic_series = []
w = 7952
h = 5304
ImageDimensions = [w, h]
FocalLengthPixels = 7528.0492610197816
PrincipalPoint = [3990.9149165798517, 2627.7534998291699]

######readfiles######

def read_xml(dir, name):
    tree = ET.parse(dir+"Block_1 - AT -export.xml")  # 打开xml文档
    root = tree.getroot()  # 获得root节点

    for root1 in root.findall('Block'):
        for root2 in root1.findall('Photogroups'):
            for photogroup in root2.findall('Photogroup'):
                for img_dimention in photogroup.findall('ImageDimensions'):
                    #长宽焦距，所有大像片都一样，可以不用读取直接设置参数
                    w = img_dimention.find("Width").text
                    h = img_dimention.find("Height").text
                focalP = photogroup.find('FocalLengthPixels').text
                for principal_point in photogroup.findall('PrincipalPoint'):
                    ppX = principal_point.find('x').text
                    ppY = principal_point.find('y').text
                for photo in photogroup.findall('Photo'):
                    M = np.zeros([3, 3])
                    C = np.zeros([1, 3])
                    # ids = []
                    # ids.append(photo.find('Id').text)
                    pht_name_ = photo.find('ImagePath').text.split('\\')[2]
                    # ids.append(pht_name_.split('.')[0])

                    # print(operator.eq(pht_name_.split('.')[0],name))

                    if operator.eq(pht_name_.split('.')[0],name):

                        for pht_pose in photo.findall('Pose'):
                            
                            for rotate_R in pht_pose.findall('Rotation'):
                                #大图的旋转矩阵
                                M[0][0] = float(rotate_R.find('M_00').text)
                                M[0][1] = float(rotate_R.find('M_01').text)
                                M[0][2] = float(rotate_R.find('M_02').text)
                                M[1][0] = float(rotate_R.find('M_10').text)
                                M[1][1] = float(rotate_R.find('M_11').text)
                                M[1][2] = float(rotate_R.find('M_12').text)
                                M[2][0] = float(rotate_R.find('M_20').text)
                                M[2][1] = float(rotate_R.find('M_21').text)
                                M[2][2] = float(rotate_R.find('M_22').text)
                            for center_pht in pht_pose.findall('Center'):
                                #center点都为相对坐标
                                x = float(center_pht.find('x').text)
                                y = float(center_pht.find('y').text)
                                z = float(center_pht.find('z').text)

                                # print(M)

                        # R_C = np.vstack((M, C))

                        # print('R_C:')
                        # print(R_C)
                        C_center=np.mat([x,y,z]).reshape(3,1)
                        return M, C_center
                        

def read_pkl(dir, name_pic):
    # pts = []
    pts_shift = np.zeros((6, 1))
    # print(name_pic)

    for file in os.listdir(dir):
        name, ext = os.path.splitext(file)
        name = name.split('_')[0]
        # print(name)    
  
        if ext == '.pkl' and operator.eq(name, name_pic):
            # print(0)
            # print(name)

            pklfile = dir + file
            with open(pklfile, 'rb') as f:
                r = pickle.load(f)

                boxes = r['rois']
                class_ids = r['class_ids']
                startpoint = r['startpoint']
                x0 = startpoint[0]
                y0 = startpoint[1]
                scores = r['scores']
                class_names = r['class_names']
                
                # print(boxes)
                N = boxes.shape[0]
                # print(N)
                if N:
                    for i in range(N):
                        y1, x1, y2, x2 = boxes[i]
                        class_id = class_ids[i]
                        label = class_names[class_id]
                        # print(label)
                        if label == 'mandatory':
  
                            pts_shift[0] = x0#中图在大图中的坐标x
                            pts_shift[1] = y0#中图在大图中的坐标y
                            pts_shift[2] = x1#盒子左上x
                            pts_shift[3] = y1#盒子左上y
                            pts_shift[4] = x2
                            pts_shift[5] = y2

                            # print(pts_shift)

                            return pts_shift
                            

                           
    # print(pts_shift)   
    

def read_pic(pic_dir):
    count_pic = 0
    for file in os.listdir(pic_dir):
        name, ext = os.path.splitext(file)

        if ext == '.jpg':
            count_pic = count_pic + 1
            test_pic_series.append(name)
    
    # print(test_pic_series)
    return count_pic

######siftmatch######

def run_sift(output_dir, pic_dir, name1, name2):

    data_dir = '../test/data_test/'
    coordinate_1 = read_pkl(data_dir, name1)
    coordinate_2 = read_pkl(data_dir, name2)
    x1s = int(coordinate_1[0]+coordinate_1[2])
    y1s = int(coordinate_1[1]+coordinate_1[3])
    x2s = int(coordinate_2[0]+coordinate_2[2])
    y2s = int(coordinate_2[1]+coordinate_2[3])
    
    # print(x1s)
    # print(y1s)
    # print(x2s)
    # print(y2s)

    img_1 = cv2.imread(pic_dir+name1+'.jpg', cv2.IMREAD_GRAYSCALE)
    img_2 = cv2.imread(pic_dir+name2+'.jpg', cv2.IMREAD_GRAYSCALE)

    #imagename = name.split('_')[0]

    h1, w1 = img_1.shape
    h2, w2 = img_2.shape

    # print(w1, h1, w2, h2)

    # SIFT特征计算
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img_1, None)
    kp2, des2 = sift.detectAndCompute(img_2, None)

    # Flann特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    goodMatch = []

    
    f1 = open(output_dir+'fpt/'+name1+'_'+name2+'.txt', 'w')
    f2 = open(output_dir+'fpt/'+name2+'_'+name1+'.txt', 'w')
    f1.close
    f2.close

    for m, n in matches:
        # goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
        if m.distance < 0.5*n.distance:
            goodMatch.append(m)

            # print(m.trainIdx)
            # print(m.queryIdx)
            # print(kp1[m.queryIdx].pt)
            # print(kp2[m.trainIdx].pt)

            f1 = open(output_dir+'fpt/'+name1+'_'+name2+'.txt', 'a')
            f2 = open(output_dir+'fpt/'+name2+'_'+name1+'.txt', 'a')
             
            (x1,y1) = kp1[m.queryIdx].pt
            f1.write(str(x1+x1s)+'\t'+str(y1+y1s)+'\n')
            # print(x1)
            # print(y1)
            (x2,y2) = kp2[m.trainIdx].pt
            f2.write(str(x2+x2s)+'\t'+str(y2+y2s)+'\n')
            # print(x2)
            # print(y2)

            f1.close
            f2.close


    # 增加一个维度
    goodMatch = np.expand_dims(goodMatch, 1)
    # print(goodMatch[:20])
    # print(goodMatch[0])
    img_out = cv2.drawMatchesKnn(img_1, kp1, img_2, kp2, goodMatch[:15], None, flags=2)

    # cv2.imshow('image', img_out)#展示图片
    cv2.imwrite(output_dir+'sift/'+name2+'_'+name1+'.jpg', img_out)
    cv2.waitKey(0)#等待按键按下
    cv2.destroyAllWindows()#清除所有窗口

########get3d########

def l_mat(In,R,coor):
    l = np.mat(np.zeros((2,3)))
    f = In[0,2]
    xo = In[0,0]
    yo = In[0,1]

    x = coor[0]
    y = coor[1]
    # print(coor)
    
    l[0,0] = f*R[0,0] + (x-xo)*R[0,2]
    l[0,1] = f*R[1,0] + (x-xo)*R[1,2]
    l[0,2] = f*R[2,0] + (x-xo)*R[2,2]
    l[1,0] = f*R[0,1] + (y-yo)*R[0,2]
    l[1,1] = f*R[1,1] + (y-yo)*R[1,2]
    l[1,2] = f*R[2,1] + (y-yo)*R[2,2]
    
    return l
    
def l_approximate(In,R,coor,Ex):
    l_app = np.mat(np.zeros((2,1)))
    f = In[0,2]
    xo = In[0,0]
    yo = In[0,1]
    x = coor[0]
    y = coor[1]
    Xs = Ex[0,0]
    Ys = Ex[1,0]
    Zs = Ex[2,0]
    # print(Ex)

    l_app[0,0] = (f*R[0,0]*Xs + f*R[1,0]*Ys + f*R[2,0]*Zs
         + (x-xo)*R[0,2]*Xs + (x-xo)*R[1,2]*Ys + (x-xo)*R[2,2]*Zs)
    l_app[1,0] = (f*R[0,1]*Xs + f*R[1,1]*Ys + f*R[2,1]*Zs
         + (y-yo)*R[0,2]*Xs + (y-yo)*R[1,2]*Ys + (y-yo)*R[2,2]*Zs)
    
    return l_app

def get_data(output_dir, data_dir, pic_dir, data1, data2):

    data1x = []
    data1y = []
    data2x = []
    data2y = []

    i = 0
    j = 0
    with open(output_dir+'fpt/'+data1+'_'+data2+'.txt', 'r') as f1:
        
        while True:
            lines = f1.readlines(10000)
            if not lines:
                break
            for line in lines:
                line = line.strip('\n')
                data1x.append(line.split('\t', 1 )[0])
                data1y.append(line.split('\t', 1 )[1])
                i=i+1
        
    with open(output_dir+'fpt/'+data2+'_'+data1+'.txt', 'r') as f2:
        
        while True:
            lines = f2.readlines(10000)
            if not lines:
                break
            for line in lines:
                line = line.strip('\n')
                data2x.append(line.split('\t', 1 )[0])
                data2y.append(line.split('\t', 1 )[1])
                j=j+1

    if(i!=j):
        print("Wrong matches of keypoints!")
        return 0

    k = 0
    while(k<i):
        run_locate(output_dir, float(data1x[k]),float(data1y[k]),float(data2x[k]),float(data2y[k]), data1, data2)
        #print(k)
        k = k+1   

def run_locate(output_dir, x1,y1,x2,y2,pic1,pic2):
    R1, C_center1 = read_xml(data_dir, pic1)
    R2, C_center2 = read_xml(data_dir, pic2)

    #data list
    left_HomonymousImagePoints = (x1, y1) #左同名点 
    right_HomonymousImagePoints = (x2, y2) #右同名点
    #内参（x0,y0,f),相机相同，所以值固定
    #f
    In = np.mat([PrincipalPoint[0],PrincipalPoint[1],FocalLengthPixels])
    # print(In.shape)
    K = np.mat([[In[0,2],0,In[0,0]],[0,In[0,2],In[0,1]],[0,0,1]]).reshape(3,3)
   
    # RL = read_xml(data_dir, pic1)
    # RR = read_xml(data_dir, pic2)

    t1 = -R1 * C_center1
    t2 = -R2 * C_center2

    Proj1 = np.mat(K*np.hstack((R1,t1)))
    Proj2 = np.mat(K*np.hstack((R2,t2)))

    calculate_3DX(left_HomonymousImagePoints, right_HomonymousImagePoints, Proj1, Proj2)

def calculate_3DX(kp1, kp2, Proj1, Proj2):
	
    A0 = np.mat(kp1[0] * Proj1[2,:] - Proj1[0,:])
    A1 = np.mat(kp1[1] * Proj1[2,:] - Proj1[1,:])
    A2 = np.mat(kp2[0] * Proj2[2,:] - Proj2[0,:])
    A3 = np.mat(kp2[1] * Proj2[2,:] - Proj2[1,:])

    train_data = np.mat(np.vstack((A0,A1,A2,A3)))
    U,sigma,VT = np.linalg.svd(train_data)
    posx = VT[3,:].T
    posx_ = posx / posx[3][0]
    position = posx_[0:3]

    # print(position)
    with open(output_dir+'3dpt/'+'cordinate.txt', 'a') as f:
        f.write("%f;%f;%f\n" %(position[0,0],position[1,0],position[2,0]))

    return position


#######testpart######


if __name__ == "__main__":
    pic_dir = '../test/pic_test/'#切割后‘50’限速文件
    data_dir = '../test/data_test/'#原图jpg   npy   pkl   xml文件
    output_dir = '../test/output_dir/'#特征点（fpt)，三维点(3dpt)输出文件
    
    f = open(output_dir+'3dpt/'+'cordinate.txt', 'w')
    f.close
    
    read_pic(pic_dir)

    for pic1 in test_pic_series:
        for pic2 in test_pic_series:
            if pic1 != pic2:
                try:
                    run_sift(output_dir, pic_dir, pic1, pic2)
                    get_data(output_dir, data_dir, pic_dir, pic1, pic2)
                    print(pic1+'_'+pic2+'success!')
                except:
                    print("match error!")
                    pass


                
    
    # run_sift(output_dir, pic_dir, 'DSC00244', 'DSC00253')
    # get_data(output_dir, data_dir, pic_dir, 'DSC00244', 'DSC00253')
    # run_sift(output_dir, pic_dir, 'DSC00262', 'DSC00265')
    # get_data(output_dir, data_dir, pic_dir, 'DSC00262', 'DSC00265')

    pass
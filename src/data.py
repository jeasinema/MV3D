from kitti_data import pykitti
# from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED
# from kitti_data.draw import *
from kitti_data.io import *
import net.utility.draw as draw
from net.processing.boxes3d import *
from config import TOP_X_MAX,TOP_X_MIN,TOP_Y_MAX,TOP_Z_MIN,TOP_Z_MAX, \
    TOP_Y_MIN,TOP_X_DIVISION,TOP_Y_DIVISION,TOP_Z_DIVISION
from config import cfg
import os
import cv2
import numpy
import glob
from multiprocessing import Pool
from collections import OrderedDict
import config
import ctypes
import math

if config.cfg.USE_CLIDAR_TO_TOP:
    SharedLib = ctypes.cdll.LoadLibrary('/home/stu/MV3D/src/lidar_data_preprocess/'
                                        'Python_to_C_Interface/ver3/LidarTopPreprocess.so')

class Preprocess(object):


    def rgb(self, rgb):
        # FIXME
        #rgb = crop_image(rgb)
        #return rgb
        return cv2.resize(rgb, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT)) #!!! order!!


    def bbox3d(self, obj):
        return box3d_compose(translation= obj.translation, rotation= obj.rotation, size= obj.size)


    def label(self, obj):
        label=0
        if obj.type=='Van' or obj.type=='Truck' or obj.type=='Car' or obj.type=='Tram':# todo : only  support 'Van'
            label = 1
        return label


    #def lidar_to_top(self, lidar :np.dtype) ->np.ndarray:
    def lidar_to_top(self, lidar):
        if cfg.USE_CLIDAR_TO_TOP:
            top = clidar_to_top(lidar)
        else:
            top = lidar_to_top(lidar)

        return top

    def lidar_to_front(self, lidar):

        def cal_height(point):
            return np.clip(point[2] + cfg.VELODYNE_HEIGHT, a_min=0, a_max=None)
        def cal_distance(point):
            return math.sqrt(sum(np.array(point)**2))
        def cal_intensity(point):
            return point[3]
        def to_front(point):
            return (
                int(math.atan2(point[1], point[0])/cfg.VELODYNE_ANGULAR_RESOLUTION),
                int(math.atan2(point[2], math.sqrt(point[0]**2 + point[1]**2)) \
                    /cfg.VELODYNE_VERTICAL_RESOLUTION)
            )

        # using the same crop method as top view
        idx = np.where (lidar[:,0]>TOP_X_MIN)
        lidar = lidar[idx]
        idx = np.where (lidar[:,0]<TOP_X_MAX)
        lidar = lidar[idx]

        idx = np.where (lidar[:,1]>TOP_Y_MIN)
        lidar = lidar[idx]
        idx = np.where (lidar[:,1]<TOP_Y_MAX)
        lidar = lidar[idx]

        idx = np.where (lidar[:,2]>TOP_Z_MIN)
        lidar = lidar[idx]
        idx = np.where (lidar[:,2]<TOP_Z_MAX)
        lidar = lidar[idx]

        r, c, l = [], [], []
        for point in lidar:
            pc, pr = to_front(point)
            if cfg.FRONT_C_MIN < pc < cfg.FRONT_C_MAX and cfg.FRONT_R_MIN < pr < cfg.FRONT_R_MAX: 
                c.append(pc)
                r.append(pr)
                l.append(point)
        c, r = np.array(c).astype(np.int32), np.array(r).astype(np.int32)
        c += int(cfg.FRONT_C_OFFSET)
        r += int(cfg.FRONT_R_OFFSET)

        ## FIXME simply /2 for resize
        c //= 2
        r //= 2

        channel = 3 # height, distance, intencity
        front = np.zeros((cfg.FRONT_WIDTH, cfg.FRONT_HEIGHT, channel+1), dtype=np.float32)
        for point, p_c, p_r in zip(l, c, r):
            if 0 <= p_c < cfg.FRONT_WIDTH and 0 <= p_r < cfg.FRONT_HEIGHT:
                front[p_c, p_r, 0:channel] *= front[p_c, p_r, channel]
                front[p_c, p_r, 0:channel] += np.array([cal_height(point), cal_distance(point), cal_intensity(point)])
                front[p_c, p_r, channel] += 1
                front[p_c, p_r, 0:channel] /= front[p_c, p_r, channel]

        return front[:, :, 0:channel]

    def lidar_to_front_fast(self, lidar):

        def cal_height(points):
            return np.clip(points[:, 2] + cfg.VELODYNE_HEIGHT, a_min=0, a_max=None).astype(np.float32).reshape((-1, 1))
        def cal_distance(points):
            return np.sqrt(np.sum(points**2, axis=1)).astype(np.float32).reshape((-1, 1))
        def cal_intensity(points):
            return points[:, 3].astype(np.float32).reshape((-1, 1))
        def to_front(points):
            return np.array([
                np.arctan2(points[:, 1], points[:, 0])/cfg.VELODYNE_ANGULAR_RESOLUTION,
                np.arctan2(points[:, 2], np.sqrt(points[:, 0]**2 + points[:, 1]**2)) \
                    /cfg.VELODYNE_VERTICAL_RESOLUTION
            ], dtype=np.int32).T

        # using the same crop method as top view
        idx = np.where (lidar[:,0]>TOP_X_MIN)
        lidar = lidar[idx]
        idx = np.where (lidar[:,0]<TOP_X_MAX)
        lidar = lidar[idx]

        idx = np.where (lidar[:,1]>TOP_Y_MIN)
        lidar = lidar[idx]
        idx = np.where (lidar[:,1]<TOP_Y_MAX)
        lidar = lidar[idx]

        idx = np.where (lidar[:,2]>TOP_Z_MIN)
        lidar = lidar[idx]
        idx = np.where (lidar[:,2]<TOP_Z_MAX)
        lidar = lidar[idx]

        points = to_front(lidar)
        ind = np.where(cfg.FRONT_C_MIN < points[:, 0])
        points, lidar = points[ind], lidar[ind]
        ind = np.where(points[:, 0] < cfg.FRONT_C_MAX)
        points, lidar = points[ind], lidar[ind]
        ind = np.where(cfg.FRONT_R_MIN < points[:, 1])
        points, lidar = points[ind], lidar[ind]
        ind = np.where(points[:, 1] < cfg.FRONT_R_MAX)
        points, lidar = points[ind], lidar[ind]

        points[:, 0] += int(cfg.FRONT_C_OFFSET)
        points[:, 1] += int(cfg.FRONT_R_OFFSET)
        points //= 2

        ind = np.where(0 <= points[:, 0])
        points, lidar = points[ind], lidar[ind]
        ind = np.where(points[:, 0] < cfg.FRONT_WIDTH)
        points, lidar = points[ind], lidar[ind]
        ind = np.where(0 <= points[:, 1])
        points, lidar = points[ind], lidar[ind]
        ind = np.where(points[:, 1] < cfg.FRONT_HEIGHT)
        points, lidar = points[ind], lidar[ind]

        channel = 3 # height, distance, intencity
        front = np.zeros((cfg.FRONT_WIDTH, cfg.FRONT_HEIGHT, channel), dtype=np.float32)
        weight_mask = np.zeros_like(front[:, :, 0])
        def _add(x):
            weight_mask[int(x[0]), int(x[1])] += 1
        def _fill(x):
            front[int(x[0]), int(x[1]), :] += x[2:]
        np.apply_along_axis(_add, 1, points)
        weight_mask[weight_mask == 0] = 1  # 0 and 1 are both 1
        buf = np.hstack((points, cal_height(lidar), cal_distance(lidar), cal_intensity(lidar)))
        np.apply_along_axis(_fill, 1, buf)
        front /= weight_mask[:, :, np.newaxis]

        return front  


proprocess = Preprocess()



def filter_center_car(lidar):
    idx = np.where(np.logical_or(numpy.abs(lidar[:, 0]) > 4.7/2, numpy.abs(lidar[:, 1]) > 2.1/2))
    lidar = lidar[idx]
    return lidar

## objs to gt boxes ##
def obj_to_gt_boxes3d(objs):

    num        = len(objs)
    gt_boxes3d = np.zeros((num,8,3),dtype=np.float32)
    gt_labels  = np.zeros((num),    dtype=np.int32)

    for n in range(num):
        obj = objs[n]
        b   = obj.box
        label=0
        if obj.type=='Van' or obj.type=='Truck' or obj.type=='Car' or obj.type=='Tram':# todo : only  support 'Van'
            label = 1

        gt_labels [n]=label
        gt_boxes3d[n]=b

    return  gt_boxes3d, gt_labels

def draw_top_image(lidar_top):
    top_image = np.sum(lidar_top,axis=2) # simply do sum on all the channel
    top_image = top_image-np.min(top_image)
    divisor = np.max(top_image)-np.min(top_image)
    top_image = (top_image/divisor*255)
    top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)
    return top_image

def draw_front_image(lidar_front):
    # input: (cfg.FRONT_WIDTH, cfg.FRONT_HEIGHT, 3)
    front_image = np.sum(lidar_front, axis=2)
    front_image = front_image-np.min(front_image)
    divisor = np.max(front_image) - np.min(front_image)
    front_image = (front_image/divisor*255)
    front_image = np.dstack((front_image, front_image, front_image)).astype(np.uint8)
    return front_image

def clidar_to_top(lidar):
    if (cfg.DATA_SETS_TYPE == 'didi' or cfg.DATA_SETS_TYPE == 'test'):
        lidar=filter_center_car(lidar)

    # Calculate map size and pack parameters for top view and front view map (DON'T CHANGE THIS !)
    Xn = math.floor((TOP_X_MAX - TOP_X_MIN) / TOP_X_DIVISION)
    Yn = math.floor((TOP_Y_MAX - TOP_Y_MIN) / TOP_Y_DIVISION)
    Zn = math.floor((TOP_Z_MAX - TOP_Z_MIN) / TOP_Z_DIVISION)

    top_flip = np.ones((Xn, Yn, Zn + 2), dtype=np.double)  # DON'T CHANGE THIS !

    num = lidar.shape[0]  # DON'T CHANGE THIS !

    # call the C function to create top view maps
    # The np array indata will be edited by createTopViewMaps to populate it with the 8 top view maps
    SharedLib.createTopMaps(ctypes.c_void_p(lidar.ctypes.data),
                            ctypes.c_int(num),
                            ctypes.c_void_p(top_flip.ctypes.data),
                            ctypes.c_float(TOP_X_MIN), ctypes.c_float(TOP_X_MAX),
                            ctypes.c_float(TOP_Y_MIN), ctypes.c_float(TOP_Y_MAX),
                            ctypes.c_float(TOP_Z_MIN), ctypes.c_float(TOP_Z_MAX),
                            ctypes.c_float(TOP_X_DIVISION), ctypes.c_float(TOP_Y_DIVISION),
                            ctypes.c_float(TOP_Z_DIVISION),
                            ctypes.c_int(Xn), ctypes.c_int(Yn), ctypes.c_int(Zn)
                            )
    top = np.flipud(np.fliplr(top_flip))
    return top


## lidar to top ##
# density and intensity is right
def lidar_to_top(lidar):

    idx = np.where (lidar[:,0]>TOP_X_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,0]<TOP_X_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar[:,1]>TOP_Y_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,1]<TOP_Y_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar[:,2]>TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,2]<TOP_Z_MAX)
    lidar = lidar[idx]

    if (cfg.DATA_SETS_TYPE == 'didi' or cfg.DATA_SETS_TYPE == 'test' or cfg.DATA_SETS_TYPE=='didi2'):
        lidar=filter_center_car(lidar)


    pxs=lidar[:,0]
    pys=lidar[:,1]
    pzs=lidar[:,2]
    prs=lidar[:,3]
    qxs=((pxs-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    qys=((pys-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    #qzs=((pzs-TOP_Z_MIN)//TOP_Z_DIVISION).astype(np.int32)
    qzs=(pzs-TOP_Z_MIN)/TOP_Z_DIVISION
    quantized = np.dstack((qxs,qys,qzs,prs)).squeeze()

    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)
    height  = Xn - X0
    width   = Yn - Y0
    channel = Zn - Z0  + 2
    # print('height,width,channel=%d,%d,%d'%(height,width,channel))
    top = np.zeros(shape=(height,width,channel), dtype=np.float32)


    # histogram = Bin(channel, 0, Zn, "z", Bin(height, 0, Yn, "y", Bin(width, 0, Xn, "x", Maximize("intensity"))))
    # histogram.fill.numpy({"x": qxs, "y": qys, "z": qzs, "intensity": prs})

    if 1:  #new method
        for x in range(Xn):
            ix  = np.where(quantized[:,0]==x)
            quantized_x = quantized[ix]
            if len(quantized_x) == 0 : continue
            yy = -x

            for y in range(Yn):
                iy  = np.where(quantized_x[:,1]==y)
                quantized_xy = quantized_x[iy]
                count = len(quantized_xy)
                if  count==0 : continue
                xx = -y

                top[yy,xx,Zn+1] = min(1, np.log(count+1)/math.log(32))
                max_height_point = np.argmax(quantized_xy[:,2])
                top[yy,xx,Zn]=quantized_xy[max_height_point, 3]

                for z in range(Zn):
                    iz = np.where ((quantized_xy[:,2]>=z) & (quantized_xy[:,2]<=z+1))
                    quantized_xyz = quantized_xy[iz]
                    if len(quantized_xyz) == 0 : continue
                    zz = z

                    #height per slice
                    max_height = max(0,np.max(quantized_xyz[:,2])-z)
                    top[yy,xx,zz]=max_height
    return top

def lidar_to_top_fast(lidar):

    idx = np.where (lidar[:,0]>TOP_X_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,0]<TOP_X_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar[:,1]>TOP_Y_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,1]<TOP_Y_MAX)
    lidar = lidar[idx]

    idx = np.where (lidar[:,2]>TOP_Z_MIN)
    lidar = lidar[idx]
    idx = np.where (lidar[:,2]<TOP_Z_MAX)
    lidar = lidar[idx]

    if (cfg.DATA_SETS_TYPE == 'didi' or cfg.DATA_SETS_TYPE == 'test' or cfg.DATA_SETS_TYPE=='didi2'):
        lidar=filter_center_car(lidar)

    # get index in grid
    lidar[:, 0] = ((lidar[:, 0]-TOP_X_MIN)//TOP_X_DIVISION).astype(np.int32)
    lidar[:, 1] = ((lidar[:, 1]-TOP_Y_MIN)//TOP_Y_DIVISION).astype(np.int32)
    lidar[:, 2] = ((lidar[:, 2]-TOP_Z_MIN)/TOP_Z_DIVISION).astype(np.float32)

    # make grid
    X0, Xn = 0, int((TOP_X_MAX-TOP_X_MIN)//TOP_X_DIVISION)+1
    Y0, Yn = 0, int((TOP_Y_MAX-TOP_Y_MIN)//TOP_Y_DIVISION)+1
    Z0, Zn = 0, int((TOP_Z_MAX-TOP_Z_MIN)/TOP_Z_DIVISION)
    height  = Xn - X0
    width   = Yn - Y0
    channel = Zn - Z0 + 2 # height map, intensity and density
    # make cell
    grid = np.zeros((height, width, 1), dtype=np.object)
    def _fill(x):
        if grid[int(x[0]), int(x[1]), 0] == 0:
            grid[int(x[0]), int(x[1]), 0] = []
        grid[int(x[0]), int(x[1]), 0].append(x)
    def _sort_by_height(x):
        x = x[0]
        if x != 0:
            h = np.array([i[2] for i in x])
            grid[int(x[0][0]), int(x[0][1]), 0] = list(np.array(x)[h.argsort(kind='heapsort')])
    np.apply_along_axis(_fill, 1, lidar)
    np.apply_along_axis(_sort_by_height, 2, grid)

    # make top
    top = np.zeros((height, width, channel), dtype=np.float32)
    def _cal_height(x):
        x = x[0]
        if x != 0:
            i = 0
            for z in range(Zn):
                if z+1 < x[i][2]: continue
                try:
                    while not (x[i][2] <= z+1 and x[i+1][2] > z+1):
                        i += 1
                except: # all the point are in this interval
                    top[int(x[0][0]), int(x[0][1]), z] = max(x[-1][2]-z, 0)
                    break

                top[int(x[0][0]), int(x[0][1]), z] = max(x[i][2]-z, 0)
                i += 1    
    def _cal_density(x):
        x = x[0]
        if x != 0:
            top[int(x[0][0]), int(x[0][1]), Zn+1] = min(1, np.log(len(x)+1)/np.log(32))
    def _cal_intensity(x): # there may be a moderate diff from origin version since if 2 point has the same height 
        x = x[0]
        if x != 0:
            top[int(x[0][0]), int(x[0][1]), Zn] = x[-1][3]
    np.apply_along_axis(_cal_height, 2, grid)
    np.apply_along_axis(_cal_density, 2, grid)    
    np.apply_along_axis(_cal_intensity, 2, grid)

    return top



def get_all_file_names(data_seg, dates, drivers):
    # todo: check if all files from lidar, rgb, gt_boxes3d is the same
    lidar_dir = os.path.join(data_seg, "top")
    load_indexs = []
    for date in dates:
        for driver in drivers:
            # file_prefix is something like /home/stu/data/preprocessed/didi/lidar/2011_09_26_0001_*
            file_prefix = lidar_dir + '/' + date + '_' + driver + '_*'
            driver_files = glob.glob(file_prefix)
            name_list = [file.split('/')[-1].split('.')[0] for file in driver_files]
            load_indexs += name_list
    return load_indexs


def proprecess_rgb(save_preprocess_dir,dataset,date,drive,frames_index,overwrite=False):
    try:
        dataset_dir=os.path.join(save_preprocess_dir,'rgb',date,drive)
# os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(dataset_dir)
        count = 0
        for n in frames_index:
            path=os.path.join(dataset_dir,'%05d.png' % n)
            if overwrite==False and os.path.isfile(path):
                count += 1
                continue
            print('rgb images={}'.format(n))
            rgb = dataset.rgb[count][0]
            rgb = proprocess.rgb(rgb)

            # todo fit it to didi dataset later.
            cv2.imwrite(os.path.join(path), rgb)
            # cv2.imwrite(save_preprocess_dir + '/rgb/rgb_%05d.png'%n,rgb)
            count += 1
        print('rgb image save done\n')
    except:
        pass
     

def generate_top_view(save_preprocess_dir,dataset,objects,date,drive,frames_index,
                      overwrite=False,dump_image=True):
    try:
        dataset_dir = os.path.join(save_preprocess_dir, 'top', date, drive)
# os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(dataset_dir)

        count = 0
        lidars=[]
        pool=Pool(3)
        for n in frames_index:
            path=os.path.join(dataset_dir,'%05d.npy' % n)
            if overwrite==False and os.path.isfile(path):
                count += 1
                continue
            lidars.append(dataset.velo[count])
            count += 1
        if cfg.USE_CLIDAR_TO_TOP:
            print('use clidar_to_top')
            t0 = time.time()
            tops = pool.map(clidar_to_top,lidars)
            # tops=[clidar_to_top(lidar) for lidar in lidars]
            print('time = ',time.time() -t0)
        else:
            t0 = time.time()
            tops = pool.map(lidar_to_top,lidars)
            # tops=[lidar_to_top(lidar) for lidar in lidars]
            print('time = ', time.time() - t0)

        count = 0
        for top in tops:
            n=frames_index[count]
            path = os.path.join(dataset_dir, '%05d.npy' % n)
            # top = top.astype(np.float16)
            # np.save(path, top)
            np.savez_compressed(path, top_view=top)
            print('top view {} saved'.format(n))
            count+=1


        if dump_image:
            dataset_dir = os.path.join(save_preprocess_dir, 'top_image', date, drive)
# os.makedirs(dataset_dir, exist_ok=True)
            os.makedirs(dataset_dir)


            top_images=pool.map(draw_top_image,tops)
            # top_images=[draw_top_image(top) for top in tops]

            count = 0
            for top_image in top_images:
                n = frames_index[count]
                top_image_path = os.path.join(dataset_dir,'%05d.png' % n)

                # draw bbox on top image
                if objects!=None:
                    gt_boxes3d, gt_labels = obj_to_gt_boxes3d(objects[count])
                    top_image = draw_box3d_on_top(top_image, gt_boxes3d, color=(0, 0, 80))
                cv2.imwrite(top_image_path, top_image)
                count += 1
                print('top view image {} saved'.format(n))
        pool.close()
        pool.join()
    except:
        pass




def preprocess_bbox(save_preprocess_dir,objects,date,drive,frames_index,overwrite=False):

    try:
        bbox_dir = os.path.join(save_preprocess_dir, 'gt_boxes3d', date, drive)
# os.makedirs(bbox_dir, exist_ok=True)
        os.makedirs(bbox_dir)


        lable_dir = os.path.join(save_preprocess_dir, 'gt_labels', date, drive)
# os.makedirs(lable_dir, exist_ok=True)
        os.makedirs(lable_dir)


        count = 0
        for n in frames_index:
            bbox_path=os.path.join(bbox_dir,'%05d.npy' % n)
            lable_path=os.path.join(lable_dir,'%05d.npy' % n)
            if overwrite==False and os.path.isfile(bbox_path):
                count += 1
                continue

            if overwrite==False and os.path.isfile(lable_path):
                count += 1
                continue

            print('boxes3d={}'.format(n))

            obj = objects[count]
            gt_boxes3d, gt_labels = obj_to_gt_boxes3d(obj)

            np.save(bbox_path, gt_boxes3d)
            np.save(lable_path, gt_labels)
            count += 1
    except:
        pass

def draw_top_view_image(save_preprocess_dir,objects,date,drive,frames_index,overwrite=False):
    try:
        dataset_dir = os.path.join(save_preprocess_dir, 'top_image', date, drive)
# os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(dataset_dir)


        count = 0
        for n in frames_index:
            top_image_path=os.path.join(dataset_dir,'%05d.png' % n)
            if overwrite==False and os.path.isfile(top_image_path):
                count += 1
                continue

            print('draw top view image ={}'.format(n))

            top = np.load(os.path.join(save_preprocess_dir,'top',date,drive,'%05d.npy.npz' % n) )
            top = top['top_view']
            top_image = draw_top_image(top)

            # draw bbox on top image
            if objects != None:
                gt_boxes3d = np.load(os.path.join(save_preprocess_dir,'gt_boxes3d',date,drive,'%05d.npy' % n))
                top_image = draw_box3d_on_top(top_image, gt_boxes3d, color=(0, 0, 80))
            else:
                print('Not found gt_boxes3d,skip draw bbox on top image')

            cv2.imwrite(top_image_path, top_image)
            count += 1
        print('top view image draw done\n')
    except:
        pass

def dump_lidar(save_preprocess_dir,dataset,date,drive,frames_index,overwrite=False):
    try:
        dataset_dir = os.path.join(save_preprocess_dir, 'lidar', date, drive)
# os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(dataset_dir)


        count = 0
        for n in frames_index:

            lidar_dump_path=os.path.join(dataset_dir,'%05d.npy' % n)
            if overwrite==False and os.path.isfile(lidar_dump_path):
                count += 1
                continue

            print('lidar data={}'.format(n))
            lidar = dataset.velo[count]
            np.save(lidar_dump_path, lidar)
            count += 1
        print('dump lidar data done\n')
    except:
        pass

def dump_bbox_on_camera_image(save_preprocess_dir,dataset,objects,date,drive,frames_index,overwrite=False):
    try:
        dataset_dir = os.path.join(save_preprocess_dir, 'gt_box_plot', date, drive)
# os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(dataset_dir)


        count = 0
        for n in frames_index:
            print('rgb images={}'.format(n))
            rgb = dataset.rgb[count][0]
            rgb = (rgb * 255).astype(np.uint8)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            rgb=crop_image(rgb)

            objs = objects[count]
            gt_boxes3d, gt_labels = obj_to_gt_boxes3d(objs)
            img = draw.draw_box3d_on_camera(rgb, gt_boxes3d)
            new_size = (img.shape[1] // 3, img.shape[0] // 3)
            img = cv2.resize(img, new_size)
            cv2.imwrite(os.path.join(dataset_dir,'%05d.png' % n), img)
            count += 1
        print('gt box image save done\n')
    except:
       pass

def crop_image(image):
    image_crop=image.copy()
    left=cfg.IMAGE_CROP_LEFT  #pixel
    right=cfg.IMAGE_CROP_RIGHT
    top=cfg.IMAGE_CROP_TOP
    bottom=cfg.IMAGE_CROP_BOTTOM
    bottom_index= -bottom if bottom!= 0 else None
    right_index = -right if right != 0 else None
    image_crop=image_crop[top:bottom_index,left:right_index,:]
    return image_crop

def is_evaluation_dataset(date, drive):
    if date=='Round1Test' or date == 'test_car' or date == 'test_ped':
        return True
    else:
        return False

def data_in_single_driver(raw_dir, date, drive, frames_index=None):

    if (cfg.DATA_SETS_TYPE == 'didi2'):
        img_path = os.path.join(raw_dir, date, drive, "image_02", "data")
    elif (cfg.DATA_SETS_TYPE == 'didi'):
        img_path = os.path.join(raw_dir, date, drive, "image_02", "data")
    elif cfg.DATA_SETS_TYPE == 'kitti':
        img_path = os.path.join(raw_dir, date, date + "_drive_" + drive + "_sync", "image_02", "data")
    elif(cfg.DATA_SETS_TYPE == 'test'):
        img_path = os.path.join(raw_dir, date, drive, "image_02", "data")
    else:
        raise ValueError('unexpected type in cfg.DATA_SETS_TYPE item: {}!'.format(cfg.DATA_SETS_TYPE))

    if frames_index is None:
        nb_frames = len(glob.glob(img_path+"/*.png"))
        frames_index = range(nb_frames)

    # spilt large numbers of frame to small chunks
    if (cfg.DATA_SETS_TYPE == 'test'):
        max_cache_frames_num = 3
    else:
        max_cache_frames_num = 3
    if len(frames_index)>max_cache_frames_num:
        frames_idx_chunks=[frames_index[i:i+max_cache_frames_num] for i in range(0,len(frames_index),max_cache_frames_num)]
    else:
        frames_idx_chunks=[frames_index]

    for i, frames_index in enumerate(frames_idx_chunks):
        # The range argument is optional - default is None, which loads the whole dataset
        dataset = pykitti.raw(raw_dir, date, drive, frames_index) #, range(0, 50, 5))

        # read objects
        tracklet_file = os.path.join(dataset.data_path, 'tracklet_labels.xml')
        print(tracklet_file)
        if os.path.isfile(tracklet_file):
            objects = read_objects(tracklet_file, frames_index)
        elif is_evaluation_dataset(date, drive):
            objects=None
            print("Skip read evaluation_dataset's tracklet_labels file : ")
        else:
            raise ValueError('read_objects error!!!!!')

        # Load some data
        # dataset.load_calib()         # Calibration data are accessible as named tuples
        # dataset.load_timestamps()    # Timestamps are parsed into datetime objects
        # dataset.load_oxts()          # OXTS packets are loaded as named tuples
        # dataset.load_gray()         # Left/right images are accessible as named tuples
        dataset.load_left_rgb()
        dataset.load_velo()          # Each scan is a Nx4 array of [x,y,z,reflectance]


        ############# convert   ###########################
        save_preprocess_dir = cfg.PREPROCESSING_DATA_SETS_DIR

        if 1:  ## rgb images --------------------
            proprecess_rgb(save_preprocess_dir, dataset, date, drive, frames_index, overwrite=False)


        if 1:  ##generate top view --------------------
            generate_top_view(save_preprocess_dir, dataset,objects, date, drive, frames_index,
                              overwrite=True,dump_image=True)

        if 1 and objects!=None:  ## preprocess boxes3d  --------------------
            preprocess_bbox(save_preprocess_dir, objects, date, drive, frames_index, overwrite=True)

        if 0: ##draw top image with bbox
            draw_top_view_image(save_preprocess_dir, objects, date, drive, frames_index, overwrite=True)


        # dump lidar data
        if 0:
            dump_lidar(save_preprocess_dir, dataset, date, drive, frames_index, overwrite=False)

        if 1 and objects!= None: #dump gt boxes
            dump_bbox_on_camera_image(save_preprocess_dir, dataset, objects, date, drive, frames_index, overwrite=True)

        ############# analysis ###########################
        # if 0: ## make mean
        #     mean_image = np.zeros((400,400),dtype=np.float32)
        #     frames_index=20
        #     for n in frames_index:
        #         print(n)
        #         top_image = cv2.imread(save_preprocess_dir + '/top_image/'+date+'_'+drive+'_%05d.npy'%n,0)
        #         mean_image += top_image.astype(np.float32)
        #
        #     mean_image = mean_image / len(frames_index)
        #     cv2.imwrite(save_preprocess_dir + '/top_image/top_mean_image'+date+'_'+drive+'.png',mean_image)
        #
        #
        # if 0: ## gt_3dboxes distribution ... location and box, height
        #     depths =[]
        #     aspects=[]
        #     scales =[]
        #     mean_image = cv2.imread(save_preprocess_dir + '/top_image/top_mean_image'+date+'_'+drive+'.png',0)
        #
        #     for n in frames_index:
        #         print(n)
        #         gt_boxes3d = np.load(save_preprocess_dir + '/gt_boxes3d/'+date+'_'+drive+'_%05d.npy'%n)
        #
        #         top_boxes = box3d_to_top_box(gt_boxes3d)
        #         draw_box3d_on_top(mean_image, gt_boxes3d,color=(255,255,255), thickness=1, darken=1)
        #
        #         for i in range(len(top_boxes)):
        #             x1,y1,x2,y2 = top_boxes[i]
        #             w = math.fabs(x2-x1)
        #             h = math.fabs(y2-y1)
        #             area = w*h
        #             s = area**0.5
        #             scales.append(s)
        #
        #             a = w/h
        #             aspects.append(a)
        #
        #             box3d = gt_boxes3d[i]
        #             d = np.sum(box3d[0:4,2])/4 -  np.sum(box3d[4:8,2])/4
        #             depths.append(d)
        #
        #     depths  = np.array(depths)
        #     aspects = np.array(aspects)
        #     scales  = np.array(scales)
        #
        #     numpy.savetxt(save_preprocess_dir + '/depths'+date+'_'+drive+'.txt',depths)
        #     numpy.savetxt(save_preprocess_dir + '/aspects'+date+'_'+drive+'.txt',aspects)
        #     numpy.savetxt(save_preprocess_dir + '/scales'+date+'_'+drive+'.txt',scales)
        #     cv2.imwrite(save_preprocess_dir + '/top_image/top_rois'+date+'_'+drive+'.png', mean_image)


def preproces(mapping, frames_index):
    # if mapping is none, using all dataset under raw_data_sets_dir.
    if mapping is None:
        paths = glob.glob(os.path.join(cfg.RAW_DATA_SETS_DIR ,'*'))
        map_key = [os.path.basename(path) for path in paths]
        map_value = [os.listdir(bag_name) for bag_name in map_key]
        mapping = {k: v for k, v in zip(map_key, map_value)}

    # looping through
    for key in mapping.keys():
        if mapping[key] is None:
            paths = glob.glob(os.path.join(cfg.RAW_DATA_SETS_DIR, key, '*'))
            if len(paths) == 0:
                raise ValueError('can not found any file in:{}'.format(os.path.join(cfg.RAW_DATA_SETS_DIR, key, '*')))
            drivers_des=[os.path.basename(path) for path in paths]
        else:
            drivers_des=mapping[key]
        for driver in drivers_des:
            print('date {} and driver {}'.format(key, driver))
            data_in_single_driver(cfg.RAW_DATA_SETS_DIR, key, driver, frames_index)

# main #################################################################33
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    if (cfg.DATA_SETS_TYPE == 'didi'):
        data_dir = {'1': ['15', '10']}
        data_dir = OrderedDict(data_dir)
        frames_index = None  # None
    elif (cfg.DATA_SETS_TYPE == 'didi2'):
        dir_prefix = '/home/stu/round12_data/raw/didi'

        bag_groups = ['suburu_pulling_to_left',
                 'nissan_following_long',
                 'suburu_following_long',
                 'nissan_pulling_to_right',
                 'suburu_not_visible',
                 'cmax_following_long',
                 'nissan_driving_past_it',
                 'cmax_sitting_still',
                 'suburu_pulling_up_to_it',
                 'suburu_driving_towards_it',
                 'suburu_sitting_still',
                 'suburu_driving_away',
                 'suburu_follows_capture',
                 'bmw_sitting_still',
                 'suburu_leading_front_left',
                 'nissan_sitting_still',
                 'nissan_brief',
                 'suburu_leading_at_distance',
                 'bmw_following_long',
                 'suburu_driving_past_it',
                 'nissan_pulling_up_to_it',
                 'suburu_driving_parallel',
                 'nissan_pulling_to_left',
                 'nissan_pulling_away', 'ped_train']

        bag_groups = ['suburu_pulling_to_left',
                     'nissan_following_long',
                     'nissan_driving_past_it',
                     'cmax_sitting_still',
                      'cmax_following_long',
                     'suburu_driving_towards_it',
                     'suburu_sitting_still',
                     'suburu_driving_away',
                     'suburu_follows_capture',
                     'bmw_sitting_still',
                     'suburu_leading_front_left',
                     'nissan_sitting_still',
                     'suburu_leading_at_distance',
                     'suburu_driving_past_it',
                     'nissan_pulling_to_left',
                     'nissan_pulling_away', 'ped_train']

        # use orderedDict to fix the dictionary order.
        data_dir = OrderedDict([(bag_group, None) for bag_group in bag_groups])
        print('ordered dictionary here: ', data_dir)

        frames_index=None  #None
    elif cfg.DATA_SETS_TYPE == 'kitti':
        data_dir = {'2011_09_26': ['0001', '0017', '0029', '0052', '0070', '0002', '0018', '0035', '0056', '0079',
                                   '0019', '0036', '0005', '0057', '0084', '0020', '0039', '0059', '0086', '0011',
                                   '0023', '0046', '0060', '0091','0013', '0027', '0048', '0061', '0015', '0028',
                                   '0051', '0064']}

        frames_index = None # [0,5,8,12,16,20,50]
    elif cfg.DATA_SETS_TYPE == 'test':
        data_dir = {'1':None, '2':None}
        data_dir = OrderedDict(data_dir)
        frames_index=None
    else:
        raise ValueError('unexpected type in cfg.DATA_SETS_TYPE item: {}!'.format(cfg.DATA_SETS_TYPE))

    import time
    t0=time.time()

    preproces(data_dir, frames_index)

    print('use time : {}'.format(time.time()-t0))





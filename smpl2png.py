import os
from star.pytorch.star import STAR
import torch
import numpy as np 
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.collections import PolyCollection

def obj2png(dir, scale_factor, i):
    V, F = [], []
    with open(dir) as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                V.append([float(x) for x in values[1:4]])
            elif values[0] == 'f':
                F.append([int(x) for x in values[1:4]])
    V, F = np.array(V), np.array(F)-1
       
    # 归一化处理
    V = (V-(V.max(0)+V.min(0))/2)/max(V.max(0)-V.min(0))
    
    # 放大顶点坐标
    V *= scale_factor

    T = V[F][...,:2]
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_axes([0,0,1,1], xlim=[-1,+1], ylim=[-1,+1],
                    aspect=1, frameon=False)
    collection = PolyCollection(T, closed=True, linewidth=0.01,
                                facecolor="lightblue", edgecolor="black")
    ax.add_collection(collection)
    plt.scatter(T[:, :, 0], T[:, :, 1], color='lightblue', s=1)  # 将点的颜色设置为填充颜色

    # 保存 obj 文件
    obj_dir_png = './test/png'
    if not os.path.exists(obj_dir_png):  # 检查目录是否存在，如果不存在则创建
        os.makedirs(obj_dir_png)
    png_name_png = f'{obj_dir_png}/frame{i:06d}.png'
    plt.savefig(png_name_png, dpi=100)
    plt.close()

def img2video(image_dir, video_dir):
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
        
    cmd = f"ffmpeg -r 30 -i {image_dir}/frame%06d.png -vb 20M -vcodec mpeg4 -y {video_dir}/output.mp4 -loglevel quiet"
    os.system(cmd)

def smpl2jpg():
    # STAR模型初始化
    star = STAR(gender='male')

    # 读取 pickle 文件
    with open('./motion/ep000010/gBR_sBM_cAll_d04_mBR0_ch01.json.pkl', 'rb') as f:
        data = pickle.load(f)

    # 定义结果列表
    results_list = []

    # 循环遍历 smpl_poses 中的每一组数据
    for pose_group in data['smpl_poses']:
        # 将每一组数据转换为 NumPy 数组并添加到结果列表中
        results_list.append(np.array(pose_group))

    # 将结果列表转换为 NumPy 数组
    results_array = np.array(results_list)

    # 遍历结果数组中的每一组数据
    for i, poses in tqdm(enumerate(results_array), total=len(results_array)):
        # 调整形状为（1，72）
        poses = poses.reshape(1, 72)
        
        # 体型参数beta：10
        betas = torch.cuda.FloatTensor([[0]*10])
        betas = Variable(betas, requires_grad=False)

        # 相机参数trans：3
        trans = torch.cuda.FloatTensor([1,1,1]).view(1,3)
        trans = Variable(trans, requires_grad=False)

        # 转换为 PyTorch 的 Variable
        poses = torch.cuda.FloatTensor(poses)
        poses = Variable(poses, requires_grad=False)

        # 计算 STAR 模型的输出
        d1 = star(poses, betas, trans)

        # 保存 obj 文件
        obj_dir = './eval/obj'
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)
        outmesh_path = f'{obj_dir}/{i}.obj'  # 文件名中添加索引以区分不同组的结果
        d1_np = d1.cpu().detach().numpy()
        # print('d1_np shape:', d1_np.shape) : d1_np shape: (1, 6890, 3)

        with open(outmesh_path, 'w') as fp:
            for vertices in d1_np:
                for v in vertices:
                    fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in star.f:  # 迭代每个面的索引
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))  # 加1以匹配.obj文件格式
        dir = f'{obj_dir}/{i}.obj'
        obj2png(dir, scale_factor=1.8, i=i)
        
    image_dir = './test/png'  # 设置保存图片的目录
    video_dir = './test/videos'  # 设置生成视频的目录
    img2video(image_dir, video_dir)

if __name__ == "__main__":
    smpl2jpg()

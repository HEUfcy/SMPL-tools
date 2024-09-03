import os
from star.pytorch.star import STAR
import torch
import numpy as np 
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import axes3d, Axes3D
import pickle
import open3d as o3d
from tqdm import tqdm
from PIL import Image


def create_smpl():
    # 导入包
    # Now read the smpl model.
    with open('./smpl/SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl', 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')
        Vertices = data['v_template']  ##  Loaded vertices of size (6890, 3)
        X,Y,Z = [Vertices[:,0], Vertices[:,1],Vertices[:,2]]
        
    data_matrix = np.column_stack((X, Y, Z))
    # 指定保存的文件路径
    save_path = "./picture/file.txt"
    # 使用 np.savetxt() 函数保存数据到文本文件
    np.savetxt(save_path, data_matrix, fmt='%.6f', delimiter=' ')
    print("Data saved successfully.")

    # 创建 Open3D 三角网格对象
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(data_matrix)
    # 保存为 OBJ 文件
    o3d.io.write_triangle_mesh("./test.obj", mesh)

    fig = plt.figure(figsize=[150,30]) # 创建窗口

    ax = fig.add_subplot(141, projection='3d')
    ax.scatter(Z, X, Y, s=0.02, c='k')
    smpl_view_set_axis_full_body(ax)

    ax = fig.add_subplot(142, projection='3d')
    ax.scatter(Z,X,Y,s=0.02,c='k')
    smpl_view_set_axis_full_body(ax,45)

    ax = fig.add_subplot(143, projection='3d')
    ax.scatter(Z,X,Y,s=0.02,c='k')
    smpl_view_set_axis_full_body(ax,90)

    ax = fig.add_subplot(144, projection='3d')
    ax.scatter(Z,X,Y, s=1, c='k')
    smpl_view_set_axis_full_body(ax,0)

    plt.savefig('./picture/create_smpl.png')

    return Vertices #

def smpl_view_set_axis_full_body(ax, azimuth=0):
    ## Manually set axis
    ax.view_init(0, azimuth)
    max_range = 0.55
    # ax.set_xlim(- max_range, max_range)
    # ax.set_ylim(- max_range, max_range)
    # ax.set_zlim(-0.2 - max_range, -0.2 + max_range)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-0.2 - 2, -0.2 + 2)
    ax.axis('off')

def smpl_view_set_axis_face(ax, azimuth=0):
    ## Manually set axis
    ax.view_init(0, azimuth)
    max_range = 0.1
    # ax.set_xlim(- max_range, max_range)
    # ax.set_ylim(- max_range, max_range)
    # ax.set_zlim(0.45 - max_range, 0.45 + max_range)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-0.2 - 2, -0.2 + 2)
    ax.axis('off')

def color(save_dir):
    # 绘制三维模型并上色
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 添加三角面片
    poly3d = [[vertices[vert_id - 1] for vert_id in face] for face in faces]
    collection = Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.9)
    ax.add_collection3d(collection)

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置坐标轴刻度
    ax.set_xticks(np.arange(-1, 1, 0.5))
    ax.set_yticks(np.arange(-1, 1, 0.5))
    ax.set_zticks(np.arange(-1, 1, 0.5))

    plt.savefig(f'{save_dir}/3d_model.png')

def plot_points_t1(begin, end, b, e, save_dir, obj_file_path, body_name = 'eyes'): #save_demo_dir = './picture/demo'
    X, Y, Z = create_smpl_t1(save_dir, obj_file_path) # 这里为读取music生成的动作
    Z_ft, X_ft, Y_ft = Z[begin:end], X[begin:end], Y[begin:end]
    fig = plt.figure(figsize=[10,10])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z_ft, X_ft, Y_ft, s=1, c='k')
    ax.scatter(Z_ft[b:e], X_ft[b:e], Y_ft[b:e], s=15, c='r')
    smpl_view_set_axis_face(ax, 0)
    plt.savefig(f'{save_dir}/{body_name}.png')
def t1():
    star = STAR(gender='male')
    # 定义结果列表
    results_list = [0.026966, 2.041709, -0.159421,
    -0.140865, -0.034384, 0.053507,
    -0.213289, 0.031218, -0.130430,
    0.017957, 0.102952, -0.147074,
    0.038848, -0.001821, -0.046600,
    0.154280, -0.003599, 0.021681,
    0.127517, 0.067442, 0.005631,
    0.000164, 0.003691, -0.003332,
    0.000011, -0.003593, 0.000043,
    0.156306, 0.064241, 0.105878,
    0.001053, 0.000934, 0.001774,
    0.001307, 0.002724, 0.001733,
    0.181365, -0.097459, 0.174611,
    -0.054207, -0.181539, -0.325583,
    -0.178651, 0.532536, 0.271599,
    -0.094676, 0.023696, 0.018853,
    -0.105677, -0.246210, -0.824831,
    -0.108671, 0.607504, 0.827493,
    -0.078190, -1.038523, -0.732346,
    -0.034359, 0.629879, 0.532948,
    0.002977, -0.001728, 0.000500,
    0.000501, 0.001126, -0.001865,
    -0.000483, 0.004100, -0.001239,
    -0.001866, -0.001803, 0.000429]
    # 转换为NumPy数组
    results_array = np.array(results_list)

    # 调整形状为（1，72）
    results_array = results_array.reshape(1, 72)

    # print("results_array.shape is : ",results_array)
    # 体型参数beta：10
    betas = torch.cuda.FloatTensor([[0]*10])
    betas = Variable(betas, requires_grad=False)

    # 相机参数trans：3
    trans = torch.cuda.FloatTensor([1,1,1]).view(1,3)
    trans = Variable(trans, requires_grad=False)

    poses = torch.cuda.FloatTensor(results_array)
    poses1 = Variable(poses, requires_grad=False)

    d1 = star(poses1, betas, trans)
    v = d1.cpu().detach().reshape(6890,3)
    print(v)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    # print(v1)
    # return x, y, z # x,y,z 共6890个浮点型数据
    v = v.numpy()
    return v
    # outmesh_path = './picture/hopeful.obj'  # 将整数转换为字符串
    # d1_np = d1.cpu().detach().numpy()
def create_smpl_t1(save_dir, obj_file_path):
    v = t1()
    X, Y, Z = v[:, 0] , v[:, 1], v[:, 2]

    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)
    x_mid = (xmin + xmax) / 2
    y_mid = (ymin + ymax) / 2

    X = X - x_mid
    Y = Y - y_mid

    X, Y = X * 3.6, Y * 4.3 # 使人物偏瘦偏高
    # 查看XYZ保存是否有误
    data_matrix = np.column_stack((X, Y, Z))
    # 指定保存的文件路径
    save_path = "./picture/demo_file.txt"
    # 使用 np.savetxt() 函数保存数据到文本文件
    np.savetxt(save_path, data_matrix, fmt='%.6f', delimiter=' ')
    print("Data saved successfully.")

    fig = plt.figure(figsize=[10,10]) # 创建窗口
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z, X, Y, s=1, c='k')
    smpl_view_set_axis_full_body(ax)

    # ax = fig.add_subplot(142, projection='3d')
    # ax.scatter(Z,X,Y,s=0.02,c='k')
    # smpl_view_set_axis_full_body(ax,45)


    # ax = fig.add_subplot(143, projection='3d')
    # ax.scatter(Z,X,Y,s=0.02,c='k')
    # smpl_view_set_axis_full_body(ax,90)

    # ax = fig.add_subplot(144, projection='3d')
    # ax.scatter(Z,X,Y, s=1, c='k')
    # smpl_view_set_axis_full_body(ax,0)
    plt.savefig(f'{save_dir}/create_smpl_t1.png')
    return X,Y,Z #

def plot_points_t2(begin, end, b, e, save_dir, obj_file_path, body_name = 'eyes'): #save_demo_dir = './picture/demo'
    X, Y, Z = create_smpl_t2(save_dir, obj_file_path) # 这里为读取music生成的动作
    Z_ft, X_ft, Y_ft = Z[begin:end], X[begin:end], Y[begin:end]
    fig = plt.figure(figsize=[100,30])

    ax = fig.add_subplot(141, projection='3d')
    ax.scatter(Z_ft, X_ft, Y_ft, s=1, c='k')
    ax.scatter(Z_ft[b:e], X_ft[b:e], Y_ft[b:e], s=15, c='r')
    smpl_view_set_axis_face(ax, 0)

    ax = fig.add_subplot(142, projection='3d')
    ax.scatter(Z_ft, X_ft, Y_ft, s=1, c='k')
    ax.scatter(Z_ft[b:e], X_ft[b:e], Y_ft[b:e], s=15, c='r')
    smpl_view_set_axis_full_body(ax,90)


    ax = fig.add_subplot(143, projection='3d')
    ax.scatter(Z_ft, X_ft, Y_ft, s=1, c='k')
    ax.scatter(Z_ft[b:e], X_ft[b:e], Y_ft[b:e], s=15, c='r')
    smpl_view_set_axis_full_body(ax,-90)

    ax = fig.add_subplot(144, projection='3d')
    ax.scatter(Z_ft, X_ft, Y_ft, s=1, c='k')
    ax.scatter(Z_ft[b:e], X_ft[b:e], Y_ft[b:e], s=15, c='r')
    smpl_view_set_axis_full_body(ax,180)

    plt.savefig(f'{save_dir}/{body_name}.png')
def t2():
    v = create_smpl() # 此处的v已经是numpy数组
    return v
def create_smpl_t2(save_dir, obj_file_path):
    v = t2()
    X, Y, Z = v[:, 0] , v[:, 1], v[:, 2]

    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)
    x_mid = (xmin + xmax) / 2
    y_mid = (ymin + ymax) / 2

    X = X - x_mid
    Y = Y - y_mid

    X, Y = X * 3.6, Y * 4.3 # 使人物偏瘦偏高
    Z = Z * 3.7
    # 查看XYZ保存是否有误
    data_matrix = np.column_stack((X, Y, Z))
    # 指定保存的文件路径
    save_path = "./picture/demo_file.txt"
    # 使用 np.savetxt() 函数保存数据到文本文件
    np.savetxt(save_path, data_matrix, fmt='%.6f', delimiter=' ')
    print("Data saved successfully.")

    fig = plt.figure(figsize=[10,10]) # 创建窗口
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z, X, Y, s=1, c='k')
    smpl_view_set_axis_full_body(ax)
    plt.savefig(f'{save_dir}/create_smpl_t1.png')
    return X,Y,Z #

if __name__ == "__main__":
    obj_file_path = './test/obj/0.obj'
    save_dir = './picture'

    obj_demo_path = './smpl_uv.obj'
    save_demo_dir = './picture/demo'
    if not os.path.exists(save_demo_dir):
        os.makedirs(save_demo_dir)


    # 找点的程序
    plot_points_t2(0, -1, 2778, 2800, save_demo_dir, obj_demo_path, 'left_eyes') # 2778-2800 左眼
    plot_points_t2(0, -1, 2800, 2822, save_demo_dir, obj_demo_path, 1) # 2778-2800 左眼
    # 2778-2800 左眼





    # read_obj_file(obj_file_path)
    # create_smpl()
    # create_smpl()
    # create_smpl_t1(save_dir, obj_file_path)
    # plot_points_t1(0, -1, 2778, 2800, save_dir, obj_file_path)













#     img2video(image_dir, video_dir)
# star = STAR(gender='male')
# # 定义结果列表
# results_list = [0.026966, 2.041709, -0.159421,
# -0.140865, -0.034384, 0.053507,
# -0.213289, 0.031218, -0.130430,
# 0.017957, 0.102952, -0.147074,
# 0.038848, -0.001821, -0.046600,
# 0.154280, -0.003599, 0.021681,
# 0.127517, 0.067442, 0.005631,
# 0.000164, 0.003691, -0.003332,
# 0.000011, -0.003593, 0.000043,
# 0.156306, 0.064241, 0.105878,
# 0.001053, 0.000934, 0.001774,
# 0.001307, 0.002724, 0.001733,
# 0.181365, -0.097459, 0.174611,
# -0.054207, -0.181539, -0.325583,
# -0.178651, 0.532536, 0.271599,
# -0.094676, 0.023696, 0.018853,
# -0.105677, -0.246210, -0.824831,
# -0.108671, 0.607504, 0.827493,
# -0.078190, -1.038523, -0.732346,
# -0.034359, 0.629879, 0.532948,
# 0.002977, -0.001728, 0.000500,
# 0.000501, 0.001126, -0.001865,
# -0.000483, 0.004100, -0.001239,
# -0.001866, -0.001803, 0.000429]
# # 转换为NumPy数组
# results_array = np.array(results_list)

# # 调整形状为（1，72）
# results_array = results_array.reshape(1, 72)

# # print("results_array.shape is : ",results_array)
# # 体型参数beta：10
# betas = torch.cuda.FloatTensor([[0]*10])
# betas = Variable(betas, requires_grad=False)

# # 相机参数trans：3
# trans = torch.cuda.FloatTensor([1,1,1]).view(1,3)
# trans = Variable(trans, requires_grad=False)

# poses = torch.cuda.FloatTensor(results_array)
# poses1 = Variable(poses, requires_grad=False)

# d1 = star(poses1, betas, trans)

# outmesh_path = './picture/hopeful.obj'  # 将整数转换为字符串
# d1_np = d1.cpu().detach().numpy()
# # print('d1_np shape:', d1_np.shape)

# with open(outmesh_path, 'w') as fp:
#     for vertices in d1_np:
#         for v in vertices:
#             fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
#     for f in star.f:  # 迭代每个面的索引
#         fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))  # 加1以匹配.obj文件格式

# color()
# create_smpl()
# plot_points(0, -1, 2778, 2800, False, True) 


def plot_points(begin, end, b, e, save_dir, obj_file_path, body_name = 'eyes' ,body=False, face=True):
    X, Y, Z = create_smpl_t1(save_dir, obj_file_path)
    Z_ft, X_ft, Y_ft = Z[begin:end], X[begin:end], Y[begin:end]
    fig = plt.figure(figsize=[10,10])
    
    if body:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Z_ft, X_ft, Y_ft, s=0.02, c='k')
        ax.scatter(Z_ft[b:e], X_ft[b:e], Y_ft[b:e], s=15, c='r')
        smpl_view_set_axis_full_body(ax)
        plt.show()
    
    if face:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Z_ft, X_ft, Y_ft, s=1, c='k')
        ax.scatter(Z_ft[b:e], X_ft[b:e], Y_ft[b:e], s=15, c='r')
        smpl_view_set_axis_face(ax,0)
    plt.savefig(f'./picture/eyes.png')

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
np.set_printoptions(precision=3)


def DLU(A):
    D = np.zeros(np.shape(A))
    L = np.zeros(np.shape(A))
    U = np.zeros(np.shape(A))
    for i in range(A.shape[0]):
        D[i, i] = A[i, i]
        for j in range(i):
            L[i, j] = -A[i, j]
        for k in range(i+1, A.shape[1]):
            U[i, k] = -A[i, k]
    return D, L, U


def SOR(D, L, U, b, x0, w, tol=1e-6, max_iter=1000):
    x = x0.copy()
    for i in range(max_iter):
        x_new = np.linalg.solve(D-w*L, np.dot((1-w)*D+w*U, x)+w*b)
        if np.linalg.norm(x_new-x) < tol:
            break
        x = x_new
    return i, x


def a1(y, x):  # a_(i,j-1),输入i,j
    return -sigmay[x]*dt


def a2(y, x):  # a_(i-1,j)
    return -sigmax[x]*dt


def a0(y, x):  # a_(i,j)
    return 2*S+(sigmax[x+1]+sigmax[x]+2*sigmay[x])*dt


def a3(y, x):  # a_(i+1,j)
    return -sigmax[x+1]*dt


def a4(y, x):  # a_(i,j+1)
    return -sigmay[x]*dt


def Lc(y, x):
    return -a1(y, x)*H_e[y, x+1]-a2(y, x)*H_e[y+1, x]-(a0(y, x)-2*S)*H_e[y+1, x+1]-a3(y, x)*H_e[y+1, x+2]-a4(y, x)*H_e[y+2, x+1]


def T_calcu():
    i = 0
    for j in range(3):
        while i < Tn_m[j]:
            Txx[i] = T_m[j]
            sigmay[i] = T_m[j]*alpha/dy/dy
            i = i+1
        j = j+1
    sigmax[0] = Txx[0]/dx/dx
    sigmax[-1] = Txx[-1]/dx/dx
    for i in range(1, Nx):
        # 第i个：T_(i-1/2, j), -1/2~Nx-1/2共Nx+1
        sigmax[i] = 2/dx/dx/(1/Txx[i-1]+1/Txx[i])
    return None


def extend():  # 根据边界条件扩展矩阵
    for i in range(Nx):
        for j in range(Ny):
            H_e[j+1, i+1] = H[j, i]
    # 上边界
    if bdry_m[0] == 1:
        for i in range(Nx):
            H_e[0, i+1] = 2*H1[0]-H[0, i]
    else:
        for i in range(Nx):
            H_e[0, i+1] = H[0, i]
    # 下边界
    if bdry_m[1] == 1:
        for i in range(Nx):
            H_e[-1, i+1] = 2*H1[1]-H[-1, i]
    else:
        for i in range(Nx):
            H_e[-1, i+1] = H[-1, i]
    # 左边界
    if bdry_m[2] == 1:
        for j in range(Ny):
            H_e[j+1, 0] = 2*H1[2]-H[j, 0]
    else:
        for j in range(Ny):
            H_e[j+1, 0] = H[j, 0]
    # 右边界
    if bdry_m[3] == 1:
        for j in range(Ny):
            H_e[j+1, -1] = 2*H1[3]-H[j, -1]
    else:
        for j in range(Ny):
            H_e[j+1, -1] = H[j, -1]
    return None


def fillA():  # 填充系数矩阵A
    # 角点填充
    # 左上
    A[0, 0] = a0(0, 0)+a1(0, 0)*(-1)**(bdry_m[0])+a2(0, 0) * \
        (-1)**(bdry_m[2])  # m[]=1为减，m[]=2为加
    A[0, 1] = a3(0, 0)
    A[0, Nx] = a4(0, 0)
    # 右上
    A[Nx-1, Nx-2] = a2(0, Nx-1)
    A[Nx-1, Nx-1] = a0(0, Nx-1)+a1(0, Nx-1)*(-1)**(bdry_m[0]) + \
        a3(0, Nx-1)*(-1)**(bdry_m[3])
    A[Nx-1, 2*Nx-1] = a4(0, Nx-1)
    # 左下
    A[(Ny-1)*Nx, (Ny-2)*Nx] = a1(Ny-1, 0)
    A[(Ny-1)*Nx, (Ny-1)*Nx] = a0(Ny-1, 0)+a2(Ny-1, 0) * \
        (-1)**(bdry_m[2])+a4(Ny-1, 0)*(-1)**(bdry_m[1])
    A[(Ny-1)*Nx, (Ny-1)*Nx+1] = a3(Ny-1, 0)
    # 右下
    A[Ny*Nx-1, Ny*Nx-Nx-1] = a1(Ny-1, Nx-1)
    A[Ny*Nx-1, Ny*Nx-2] = a2(Ny-1, Nx-1)
    A[Ny*Nx-1, Ny*Nx-1] = a0(Ny-1, Nx-1)+a3(Ny-1, Nx-1) * \
        (-1)**(bdry_m[3])+a4(Ny-1, Nx-1)*(-1)**(bdry_m[1])
    # 上边界（第零行除角点）填充
    for i in range(1, Nx-1):
        A[i, i-1] = a2(0, i)
        A[i, i] = a0(0, i)+a1(0, i)*(-1)**(bdry_m[0])
        A[i, i+1] = a3(0, i)
        A[i, i+Nx] = a4(0, i)
    # 下边界（第Ny-1行除角点）填充
    for i in range(1, Nx-1):
        A[-Nx+i, -2*Nx+i] = a1(Ny-1, i)
        A[-Nx+i, -Nx+i-1] = a2(Ny-1, i)
        A[-Nx+i, -Nx+i] = a0(Ny-1, i)+a4(Ny-1, i)*(-1)**(bdry_m[1])
        A[-Nx+i, -Nx+i+1] = a3(Ny-1, i)
    # 左边界（第零列除角点）填充
    for j in range(1, Ny-1):
        A[j*Nx, j*Nx-Nx] = a1(j, 0)
        A[j*Nx, j*Nx] = a0(j, 0)+a2(j, 0)*(-1)**(bdry_m[2])
        A[j*Nx, j*Nx+1] = a3(j, 0)
        A[j*Nx, j*Nx+Nx] = a4(j, 0)
    # 右边界（第Nx-1列除角点）填充
    for j in range(1, Ny-1):
        A[(j+1)*Nx-1, j*Nx-1] = a1(j, Nx-1)
        A[(j+1)*Nx-1, (j+1)*Nx-2] = a2(j, Nx-1)
        A[(j+1)*Nx-1, (j+1)*Nx-1] = a0(j, Nx-1)+a3(j, Nx-1)*(-1)**(bdry_m[3])
        A[(j+1)*Nx-1, (j+2)*Nx-1] = a4(j, Nx-1)
    # 内部填充
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            A[j*Nx+i, j*Nx+i] = a0(j, i)
            A[j*Nx+i, (j-1)*Nx+i] = a1(j, i)
            A[j*Nx+i, j*Nx+i-1] = a2(j, i)
            A[j*Nx+i, j*Nx+i+1] = a3(j, i)
            A[j*Nx+i, (j+1)*Nx+i] = a4(j, i)
    return None


def fillb():  # 填充b
    b[0] = 2*S*H[0, 0]+Lc(0, 0)+2*dt*WM[0, 0] + (bdry_m[0]-2) * \
        2*a1(0, 0)*H1[0]+(bdry_m[2]-2)*2*a2(0, 0)*H1[2]  # m[]=1时减去
    b[Nx-1] = 2*S*H[0, -1]+Lc(0, Nx-1)+2*dt*WM[0, -1] + (bdry_m[0]-2) * \
        2*a1(0, Nx-1)*H1[0]+(bdry_m[3]-2)*2*a3(0, 0)*H1[3]
    b[(Ny-1)*Nx] = 2*S*H[Ny-1, 0]+Lc(Ny-1, 0)+2*dt*WM[Ny-1, 0] + \
        (bdry_m[2]-2) * 2*a2(Ny-1, 0)*H1[2]+(bdry_m[1]-2)*2*a4(Ny-1, 0)*H1[1]
    b[Ny*Nx-1] = 2*S*H[Ny-1, Nx-1]+Lc(Ny-1, Nx-1)*+2*dt*WM[Ny-1, Nx-1] + \
        (bdry_m[3]-2) * 2*a2(Ny-1, Nx-1)*H1[3] + \
        (bdry_m[1]-2)*2*a4(Ny-1, Nx-1)*H1[1]
    for i in range(1, Nx-1):
        b[i] = 2*S*H[0, i]+Lc(0, i)+2*dt*WM[0, i] + \
            (bdry_m[0]-2)*2*a1(0, i)*H1[0]
        b[-Nx+i] = 2*S*H[-1, i]+Lc(Ny-1, i)+2 * \
            dt*WM[-1, i]+(bdry_m[1]-2)*2*a4(Ny-1, i)*H1[1]
    for j in range(1, Ny-1):
        b[j*Nx] = 2*S*H[j, 0]+Lc(j, 0)+2*dt*WM[j, 0] + \
            (bdry_m[2]-2)*2*a2(j, 0)*H1[2]
        b[(j+1)*Nx-1] = 2*S*H[j, Nx-1]+Lc(j, Nx-1)+2*dt*WM[j, Nx-1] + \
            (bdry_m[3]-2)*2*a3(j, Nx-1)*H1[3]
    for j in range(1, Ny-1):
        for i in range(1, Nx-1):
            b[j*Nx+i] = 2*S*H[j, i]+Lc(j, i)+2*dt*WM[j, i]
    return None


def isopleth():
    x_draw = np.linspace(0, Length, Nx)
    y_draw = np.linspace(0, Width, Ny)
    X, Y = np.meshgrid(x_draw, y_draw)
    plt.figure(figsize=(8, 6))
    level = 15  # 等值线数量
    plt.contour(
        X, Y, H, levels=level, colors='black')
    plt.contourf(X, Y, H, levels=level, cmap=plt.cm.Blues)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()  # 竖直翻转，使图像与实际网格对应
    plt.title('Isopleth of Waterhead', fontsize=18)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.colorbar(label='H')
    plt.show()
    return None


def scatter():
    x_grid = np.array([i*dx for i in range(Nx)])
    row_str = ', '.join([str(i) for i in output_row])  # 标题含参
    row_legend = []
    for k in range(output_num):
        row_legend.append('row = '+str(output_row[k]))  # 图例含参
    title = 'Waterhead at row '+row_str
    plt.figure(figsize=(8, 6))
    plt.title(title, fontsize=18)
    plt.xlabel('x/m', fontsize=18)
    plt.ylabel('H/m', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    colormap = plt.get_cmap('Blues')(np.linspace(
        1, (4/5)**(output_num-1), output_num))  # 根据输出线条数量确定线条颜色
    for i, color in zip(output_row, colormap):
        plt.plot(x_grid, H[i+1, :], color=color, marker='o')
    plt.legend(row_legend, frameon=False, fontsize=18)
    plt.show()
    return None


# 参数读取
dataframe = pd.read_excel(
    'D:\\Course File\\Porous Media\\Course Design\\input.xlsx')
x = dataframe.Index
Length = x[0]
Nx = int(x[1])
Width = x[2]
Ny = int(x[3])
Period = int(x[4])  # 时段数
S = x[5]  # 贮水系数
alpha = x[6]  # 沿列方向导水系数与沿行方向导水系数的比值
H0 = x[7]  # 初始水头
x1 = dataframe.time_s1
x2 = dataframe.time_s2
time_m1 = x1[0:Period].to_numpy()
time_m2 = x2[0:Period].to_numpy()
time_m = np.column_stack((time_m1, time_m2))  # 行数=时段数，[0]时段长度，[1]步数
x = dataframe.T_s
T_m = x[0:3].to_numpy()  # 行方向上三个参数区域内的导水系数
x = dataframe.Tn_s
Tn_m = x[0:3].to_numpy()  # 各参数区域最后一列的编号
x = dataframe.bdry_s
bdry_m = x[0:4].to_numpy()  # 分别为上、下、左、右边界类型
x = dataframe.H1
H1 = x[0:4].to_numpy()  # 第一类边界水头，分别对应上下左右边界
x = dataframe.sf_source_s
sf_source_m = x[0:Period].to_numpy()  # 单位面积面源作用
x1 = dataframe.well1
x2 = dataframe.well2
x3 = dataframe.well3
well1 = x1[0:1+int(x1[0])].to_numpy()
well2 = x2[0:1+int(x1[0])].to_numpy()
well3 = x3[0:1+int(x1[0])].to_numpy()
# (0,0)井数；(1,0)第一口井行数；(1,1)第一口井列数；(1,2)第一口井流量
well = np.column_stack((well1, well2, well3))
x = dataframe.output_num
output_num = int(x[0])
x = dataframe.output_row
output_row = x[0:output_num].to_numpy()
output_row = output_row.astype(int)

# 空间域的离散化
dx = float(Length)/float(Nx)
dy = float(Width)/float(Ny)
x0 = np.full((Nx*Ny), 50)  # 迭代初始

# 待用矩阵初始化
H = np.full((Ny, Nx), H0)  # 初始化水头矩阵
H_e = np.zeros((Ny+2, Nx+2))  # 边界扩展的水头矩阵
Txx = np.zeros(Nx)  # 导水系数矩阵
sigmax = np.zeros(Nx+1)
sigmay = np.zeros(Nx)
T_calcu()  # 网格间导水系数计算

# 逐时步计算
for i_p in range(Period):
    T = time_m[i_p, 0]
    Nt = int(time_m[i_p, 1])
    dt = T/float(Nt)
    # 读取源汇条件
    sf_source = sf_source_m[i_p]
    WM = np.full((Ny, Nx), sf_source)
    for i in range(int(well[0, 0])):
        WM[int(well[i+1, 1])-1, int(well[i+1, 0])-1] += well[i+1, 2]/dx/dy
    A = np.zeros((Nx*Ny, Nx*Ny))
    b = np.zeros(Nx*Ny)
    fillA()  # 计算系数矩阵
    D, L, U = DLU(A)
    num = 0  # 该时步内最大迭代次数
    for step in range(Nt):
        extend()  # 由H结合边界条件扩展得到H_e
        fillb()
        num_new, x = SOR(D, L, U, b, x0, 1.5)
        x0 = x.copy()
        H_new = x.reshape(Ny, Nx)
        H = H_new.copy()
        if num_new > num:
            num = num_new


# 绘制水头等值线
isopleth()
# 绘制某行的水头变化
scatter()

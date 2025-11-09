import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.ticker as ticker

def add_arrow(line, position=None, direction='right', start_ind=10,  size=20, color=None):
    """
    添加箭头到线上
    :param line: Line2D对象
    :param position: 箭头的位置
    :param direction: 'left'或'right'
    :param size: 箭头的大小
    :param color: 箭头的颜色，如果没有给出，则使用线的颜色
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    # 将start_ind设置为第100个点（注意：数组索引从0开始，所以第100个点是索引99）
    start_ind = start_ind
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),  # 使用起始位置作为箭头的起始位置
                       xy=(xdata[end_ind], ydata[end_ind]),  # 箭头指向的点
                       arrowprops=dict(arrowstyle="->", color=color, linewidth=2.3),  # 加粗的箭头
                       size=size,
                       color=color
                       )


def f(x, y):
    return -0.5 * y**2 + 2 * x * y - 0.5 * 2**2 * x**2

def df_dx(x, y):
    return 2 * y - 2**2 * x

def df_dy(x, y, L):
    return L * x - y


def grad(x, y, scale1=0.001, scale2=0.001):
    dx = 2 * y - 2**2 * x + np.random.normal(loc=0, scale=scale1)
    dy = 2 * x - y + np.random.normal(loc=0, scale=scale2)
    return np.array([dx, dy])




#
def TiAda(x, y, lr_x, lr_y, alpha, beta,  x_log, y_log, true_grad_x, epoch, grad_func, df_dx_func):
    cache = np.zeros_like([x, y])
    for _ in range(epoch):
        g = grad_func(x_log[-1], y_log[-1])
        cache += g ** 2
        x_log.append(x_log[-1] - lr_x / max(cache[0], cache[1]) ** alpha * g[0])
        y_log.append(y_log[-1] + lr_y / cache[1] ** beta * g[1])
        true_grad_x.append(abs(df_dx_func(x_log[-1], y_log[-1])))

# def ours(lr_x, lr_y, x_log, y_log, true_grad_x, epoch, grad_func, df_dx_func):
#     g = grad_func(x_log[-1], y_log[-1])
#     v_list = [g[0]]
#     w_list = [g[1]]
#     eta_x_item = 0
#     eta_y_item = 0
#     sum_x_grad = g[0] ** 2
#     sum_y_grad = g[1] ** 2
#     for _ in range(epoch):
#         beta_t = 1 / (1 + max(sum_x_grad, sum_y_grad)) ** (2/3)
#         eta_x_item += v_list[-1] ** 2 / beta_t
#         eta_y_item += w_list[-1] ** 2 / beta_t
#         eta_x = 1 / eta_x_item ** (0.3)
#         eta_y = 1 / eta_y_item ** (1e-4)
#
#         x_log.append(x_log[-1] - lr_x * max(eta_x, eta_y) * v_list[-1])
#         y_log.append(y_log[-1] + lr_y * eta_y * w_list[-1])
#         g_t = grad_func(x_log[-2], y_log[-2])
#         g_t_1 = grad_func(x_log[-1], y_log[-1])
#
#         v_list.append(g_t_1[0] + (1 - beta_t) * (v_list[-1] - g_t[0]))
#         w_list.append(g_t_1[1] + (1 - beta_t) * (w_list[-1] - g_t[1]))
#         sum_x_grad += g_t_1[0] ** 2
#         sum_y_grad += g_t_1[1] ** 2
#         true_grad_x.append(abs(df_dx_func(x_log[-1], y_log[-1])))


def ours(lr_x, lr_y, x_log, y_log, true_grad_x, epoch, grad_func, df_dx_func):
    g = grad_func(x_log[-1], y_log[-1])
    v_list = [g[0]]
    w_list = [g[1]]
    eta_x_item = 0
    eta_y_item = 0
    sum_x_grad = g[0] ** 2
    sum_y_grad = g[1] ** 2
    for _ in range(epoch):
        beta_t = 1 / (1 + max(sum_x_grad, sum_y_grad)) ** (2/3)
        eta_x_item += v_list[-1] ** 2 / beta_t
        eta_y_item += w_list[-1] ** 2 / beta_t

        x_log.append(x_log[-1] - lr_x / max(eta_x_item, eta_y_item) ** (0.3) * v_list[-1])
        y_log.append(y_log[-1] + lr_y / eta_y_item ** (0.15) * w_list[-1])
        g_t = grad_func(x_log[-2], y_log[-2])
        g_t_1 = grad_func(x_log[-1], y_log[-1])

        v_list.append(g_t_1[0] + (1 - beta_t) * (v_list[-1] - g_t[0]))
        w_list.append(g_t_1[1] + (1 - beta_t) * (w_list[-1] - g_t[1]))
        sum_x_grad += g_t_1[0] ** 2
        sum_y_grad += g_t_1[1] ** 2
        true_grad_x.append(abs(df_dx_func(x_log[-1], y_log[-1])))



def RSGDA(lr_x, lr_y, beta_x, beta_y, x_log, y_log, true_grad_x, epoch, grad_func, df_dx_func):
    g = grad_func(x_log[-1], y_log[-1])
    v_list = [g[0]]
    w_list = [g[1]]
    for i in range(epoch):
        eta_t_x = lr_x / (8+i) ** (1/3)
        eta_t_y = lr_y / (8+i) ** (1/3)
        x_log.append(x_log[-1] - eta_t_x * v_list[-1])
        y_log.append(y_log[-1] + eta_t_y * w_list[-1])

        g_t = grad_func(x_log[-2], y_log[-2])
        g_t_1 = grad_func(x_log[-1], y_log[-1])
        beta_x_t = beta_x * eta_t_x ** 2
        beta_y_t = beta_y * eta_t_y ** 2
        v_list.append(g_t_1[0] + (1 - beta_x_t) * (v_list[-1] - g_t[0]))
        w_list.append(g_t_1[1] + (1 - beta_y_t) * (w_list[-1] - g_t[1]))
        true_grad_x.append(abs(df_dx_func(x_log[-1], y_log[-1])))

def VRAdaGDA(lr_x, lr_y, beta_x, beta_y, x_log, y_log, true_grad_x, epoch, grad_func, df_dx_func):

    g = grad_func(x_log[-1], y_log[-1])
    v_list = [g[0]]
    w_list = [g[1]]
    a_x = g[0] ** 2 * 0.1
    a_y = g[1] ** 2 * 0.1
    for i in range(epoch):
        eta_t_x = lr_x / (8+i) ** (1/3)
        eta_t_y = lr_y / (8+i) ** (1/3)
        x_log.append(x_log[-1] - eta_t_x * (np.sqrt(a_x) + 1e-10) * v_list[-1])
        y_log.append(y_log[-1] + eta_t_y * (np.sqrt(a_y) + 1e-10) * w_list[-1])

        g_t = grad_func(x_log[-2], y_log[-2])
        g_t_1 = grad_func(x_log[-1], y_log[-1])

        beta_x_t = beta_x * eta_t_x ** 2
        beta_y_t = beta_y * eta_t_y ** 2

        v_list.append(g_t_1[0] + (1 - beta_x_t) * (v_list[-1] - g_t[0]))
        w_list.append(g_t_1[1] + (1 - beta_y_t) * (w_list[-1] - g_t[1]))

        a_x = 0.9 * a_x + 0.1 * g_t_1[0] ** 2
        a_y = 0.9 * a_y + 0.1 * g_t_1[1] ** 2
        true_grad_x.append(abs(df_dx_func(x_log[-1], y_log[-1])))








# 起始位置

xs = np.array([3.0])
ys = np.array([3.0])
# 记录轨迹

TiAda_x_log = [xs]
TiAda_y_log = [ys]


ours_x_log = [xs]
ours_y_log = [ys]

RSGDA_x_log = [xs]
RSGDA_y_log = [ys]

VRAdaGDA_x_log = [xs]
VRAdaGDA_y_log = [ys]

TiAda_true_grad_x = []
ours_true_grad_x = []
RSGDA_true_grad_x = []
VRAda_true_grad_x = []

RSGDA_x_logs = []
RSGDA_y_logs = []
RSGDA_true_grad_xs = []

VRAdaGDA_x_logs = []
VRAdaGDA_y_logs = []
VRAdaGDA_true_grad_xs = []



epoch = 1000
TiAda(xs, ys, 4, 0.8, 0.6, 0.4, TiAda_x_log, TiAda_y_log, TiAda_true_grad_x, epoch, grad, df_dx)
ours(0.6, 0.12, ours_x_log, ours_y_log, ours_true_grad_x,  epoch, grad, df_dx)
# RSGDA(0.01, 0.002, 512, 512, RSGDA_x_log, RSGDA_y_log, RSGDA_true_grad_x,   epoch, grad, df_dx)



VRAdaGDA(0.001, 0.0002, 512, 512, VRAdaGDA_x_log, VRAdaGDA_y_log, VRAda_true_grad_x, epoch, grad, df_dx)

# # 可视化
max_x = 1.1 * max(abs(np.concatenate([TiAda_x_log,  ours_x_log, RSGDA_x_log, VRAdaGDA_x_log ])))
max_y = 1.1 * max(abs(np.concatenate([TiAda_y_log,  ours_y_log, VRAdaGDA_y_log, RSGDA_y_log])))
#
x = np.linspace(-10, 100, 100)
y = np.linspace(-10, 150, 100)
#
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
plt.contourf(X, Y, Z, 1000, alpha=0.75, cmap='viridis', )
plt.colorbar()



linewidth = 3

sta_x = np.linspace(0, 80, 100)
sta_y = 2 * sta_x


# 绘制线


line1, = plt.plot(RSGDA_x_log, RSGDA_y_log, marker='', label='RSGDA', linewidth=linewidth)
line2, = plt.plot(VRAdaGDA_x_log, VRAdaGDA_y_log, marker='', label='VRAdaGDA', linewidth=linewidth)

line3, = plt.plot(TiAda_x_log, TiAda_y_log, marker='', label='TiAda', linewidth=linewidth)
line4, = plt.plot(ours_x_log, ours_y_log, marker='', label='AdaCM', linewidth=linewidth)
# # plt.scatter(1, 1, marker='*', color='red', s=50, label='Stationary point')
plt.plot(sta_x, sta_y, linestyle='dashed', color='r', label='Stationary points', linewidth=linewidth)
plt.scatter(3, 3, marker='*', color='green', s=100, label='Initial point')



# # 为每条线添加箭头

# add_arrow(line1, start_ind=500)
# add_arrow(line2, start_ind=210)
# add_arrow(line3, start_ind=80)
# add_arrow(line4, start_ind=300)
plt.xlabel('X')
plt.ylabel('Y')

plt.xlim(-5, 100)
plt.ylim(-5, 140)
# plt.title('Gradient Descent On Function')
plt.legend(loc='best', fontsize=11)
plt.savefig('test_function2_trajectory_1.png', bbox_inches='tight')
plt.show()
#


#
# epoch = 3000
# TiAda(xs, ys, 0.0001, 0.01, 0.65, 0.35, TiAda_x_log, TiAda_y_log, TiAda_true_grad_x, epoch=epoch)
# TiAda(xs, ys,  0.001, 0.01, 0.6, 0.4, TiAda_x_log_1, TiAda_y_log_1, TiAda_true_grad_x_1, epoch, grad1, df1_dx)
# TiAda(xs, ys,  0.001, 0.01, 0.55, 0.45, TiAda_x_log_2, TiAda_y_log_2, TiAda_true_grad_x_2,  epoch, grad1, df1_dx)
# TiAda(xs, ys, 0.001, 0.01, 0.5, 0.5, TiAda_x_log_3, TiAda_y_log_3, TiAda_true_grad_x_3,  epoch, grad1, df1_dx)
# ours(0.001, 0.01, ours_x_log, ours_y_log, ours_true_grad_x, epoch, grad1, df1_dx)
# RSGDA(0.0001, 0.001, 0.5, RSGDA_x_log, RSGDA_y_log, RSGDA_true_grad_x, epoch, grad1, df1_dx)
# VRAdaGDA(0.0001, 0.001, 0.5, VRAdaGDA_x_log, VRAdaGDA_y_log, VRAda_true_grad_x, epoch, grad1, df1_dx)
# linewidth = 3
#
#
#
# plt.figure()
# # plt.plot(TiAda_true_grad_x, label=r'TiAda($\alpha$ = 0.65, $\beta$ = 0.35)')
# # plt.plot(TiAda_true_grad_x_1, label=r'TiAda($\rho^x$ = 0.6, $\rho^y$ = 0.4)', linewidth=linewidth)
# # plt.plot(TiAda_true_grad_x_2, label=r'TiAda($\rho^x$ = 0.55, $\rho^y$ = 0.45)', linewidth=linewidth)
# plt.plot(RSGDA_true_grad_x, label='RSGDA', linewidth=linewidth)
# plt.plot(VRAda_true_grad_x, label='VRAdaGDA', linewidth=linewidth)
# plt.plot(TiAda_true_grad_x_3, label=r'TiAda', linewidth=linewidth)
# plt.plot(ours_true_grad_x, label='VRAda', linewidth=linewidth)
# plt.xlabel('#gradient calls')
# plt.ylabel(r'$\|\|\nabla_xf(x,y)\|\|$')
# plt.legend(loc='best', fontsize=11)
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# plt.show()
#
#



# epoch = 1000
# # #
# # #
# # # TiAda 和 ours 只执行一次
# # TiAda(xs, ys, 9, 1.8, 0.6, 0.4, TiAda_x_log, TiAda_y_log, TiAda_true_grad_x, epoch, grad, df_dx)
# # ours(1, 0.2, ours_x_log, ours_y_log, ours_true_grad_x, epoch, grad, df_dx)
# #
# # RSGDA 和 VRAdaGDA 从 0.0001 到 0.001，一百次迭代
# step_size = (0.0001 - 0.000000001) / 40
# for i in range(41):
#     first_value = 0.000000001 + i * step_size

#     # 每次迭代之前初始化 log 列表，并添加初始值 xs 和 ys
#     RSGDA_x_log, RSGDA_y_log, RSGDA_true_grad_x = [xs], [ys], []
#     VRAdaGDA_x_log, VRAdaGDA_y_log, VRAda_true_grad_x = [xs], [ys], []

#     # 执行函数
#     RSGDA(first_value, 0.0014, 512, 512, RSGDA_x_log, RSGDA_y_log, RSGDA_true_grad_x, epoch, grad, df_dx)
#     VRAdaGDA(first_value, 0.0014, 512, 512, VRAdaGDA_x_log, VRAdaGDA_y_log, VRAda_true_grad_x, epoch, grad, df_dx)

#     # 保存每次迭代的结果
#     RSGDA_x_logs.append(RSGDA_x_log)
#     RSGDA_y_logs.append(RSGDA_y_log)
#     RSGDA_true_grad_xs.append(RSGDA_true_grad_x)

#     VRAdaGDA_x_logs.append(VRAdaGDA_x_log)
#     VRAdaGDA_y_logs.append(VRAdaGDA_y_log)
#     VRAdaGDA_true_grad_xs.append(VRAda_true_grad_x)


# # y轴的值，即每次迭代的 true_grad_x 的最后一个元素
# RSGDA_y_values = [log[-1] for log in RSGDA_true_grad_xs]
# VRAdaGDA_y_values = [log[-1] for log in VRAdaGDA_true_grad_xs]
# #
# # 获取TiAda和AdaCM的true_grad_x的最后一个元素
# # TiAda_last_grad = TiAda_true_grad_x[-1]
# # AdaCM_last_grad = ours_true_grad_x[-1]
# # print(TiAda_last_grad)
# # print(AdaCM_last_grad)
# # 可视化
# x_values = [0.0000001 + i * step_size for i in range(41)]

# # y轴的值，即每次迭代的 true_grad_x 的最后一个元素
# RSGDA_y_values = [log[-1] for log in RSGDA_true_grad_xs]
# VRAdaGDA_y_values = [log[-1] for log in VRAdaGDA_true_grad_xs]
# #
# x_start = min(x_values)
# x_end = max(x_values)
# # 绘制图表
# # plt.figure(figsize=(10, 6))

# # 绘制线条
# plt.plot(x_values, RSGDA_y_values, label='RSGDA', marker='o', linewidth=3)
# plt.plot(x_values, VRAdaGDA_y_values, label='VRAdaGDA', marker='o', linewidth=3)
# plt.hlines(y=[3.1317822], xmin=x_start, xmax=x_end * 1.01, color='g', linestyle='-', label='TiAda', linewidth=3)
# plt.hlines(y=[2.24574033], xmin=x_start, xmax=x_end * 1.01, color='r', linestyle='-', label='AdaFM', linewidth=3)

# # 设置坐标轴标签
# plt.xlabel('Value of  $\eta^x$', fontsize=15)
# plt.ylabel(r'$\|\nabla_xf(x,y)\|$', fontsize=15)

# # 设置x轴和y轴格式化
# plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
# plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
# plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

# # 设置x轴和y轴范围
# plt.ylim(2.1, 6)  # 设置y轴范围
# plt.xlim(-1e-6, max(x_values) * 1.01)  # 设置x轴范围

# # 其他设置
# plt.grid(True)
# plt.legend(loc='best', fontsize=15)

# # 显示或保存图像
# plt.savefig('test_function2_change_x_1.png', bbox_inches='tight')
# plt.show()





# 定义一个列表来保存每次迭代的数据
# RSGDA_x_logs = []
# RSGDA_y_logs = []
# RSGDA_true_grad_xs = []
#
# VRAdaGDA_x_logs = []
# VRAdaGDA_y_logs = []
# VRAdaGDA_true_grad_xs = []
#
# # TiAda 和 ours 只执行一次
# # TiAda(xs, ys, 0.005, 0.005, 0.6, 0.4, TiAda_x_log, TiAda_y_log, TiAda_true_grad_x, epoch, grad, df_dx)
# # ours(0.00013, 0.00013, ours_x_log, ours_y_log, ours_true_grad_x, epoch, grad, df_dx)
#
# # RSGDA 和 VRAdaGDA 从 0.0001 到 0.001，一百次迭代
# step_size = (0.0035 - 0.00001) / 40
# for i in range(41):
#     second_value =  0.00001 + i * step_size
#
#     # 每次迭代之前初始化 log 列表，并添加初始值 xs 和 ys
#     RSGDA_x_log, RSGDA_y_log, RSGDA_true_grad_x = [xs], [ys], []
#     VRAdaGDA_x_log, VRAdaGDA_y_log, VRAda_true_grad_x = [xs], [ys], []
#
#     # 执行函数，保持第一个值固定，改变第二个值
#     RSGDA(0.00008, second_value,  512, 512, RSGDA_x_log, RSGDA_y_log, RSGDA_true_grad_x, epoch, grad, df_dx)
#     VRAdaGDA(0.00008, second_value,  512, 512, VRAdaGDA_x_log, VRAdaGDA_y_log, VRAda_true_grad_x, epoch, grad, df_dx)
#
#     # 保存每次迭代的结果
#     RSGDA_x_logs.append(RSGDA_x_log)
#     RSGDA_y_logs.append(RSGDA_y_log)
#     RSGDA_true_grad_xs.append(RSGDA_true_grad_x)
#
#     VRAdaGDA_x_logs.append(VRAdaGDA_x_log)
#     VRAdaGDA_y_logs.append(VRAdaGDA_y_log)
#     VRAdaGDA_true_grad_xs.append(VRAda_true_grad_x)
#
#
# x_values = [ 0.00001 + i * step_size for i in range(41)]
#
# # y轴的值，即每次迭代的 true_grad_x 的最后一个元素
# RSGDA_y_values = [log[-1] for log in RSGDA_true_grad_xs]
# VRAdaGDA_y_values = [log[-1] for log in VRAdaGDA_true_grad_xs]
#
# # 获取TiAda和AdaCM的true_grad_x的最后一个元素
# # TiAda_last_grad = TiAda_true_grad_x_3[-1]
# # AdaCM_last_grad = ours_true_grad_x[-1]
#
# x_start = min(x_values)
# x_end = max(x_values)
#
#
# # 绘制 RSGDA 和 VRAdaGDA 的结果
# plt.plot(x_values, RSGDA_y_values, label='RSGDA', marker='o', linewidth=3)
# plt.plot(x_values, VRAdaGDA_y_values, label='VRAdaGDA', marker='o', linewidth=3)
#
# # 绘制 TiAda 和 AdaCM 的水平直线
# plt.hlines(y=[3.1317822], xmin=x_start, xmax=x_end * 1.01, color='g', linestyle='-', label='TiAda', linewidth=3)
# plt.hlines(y=[2.24574033], xmin=x_start, xmax=x_end * 1.01, color='r', linestyle='-', label='AdaFM', linewidth=3)
#
#
# # 设置 x 轴和 y 轴标签
# plt.xlabel('Value of $\eta^y$', fontsize=15)
# plt.ylabel(r'$\|\nabla_xf(x,y)\|$', fontsize=15)
#
#
# # 使用科学计数法格式化坐标轴
# plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
# plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
# plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
# plt.ylim(2.1, 6)  # 设置y轴范围
# plt.xlim(1e-4 * 2, max(x_values) * 1.01)  # 设置x轴范围
# # 添加图例和网格
# plt.legend()
# plt.grid(True)
# plt.legend(loc='best', fontsize=15)
# # 保存图表到文件
# plt.savefig('C:\\Users\\48301\Desktop\\result\\test_function2_change_y_2.png', bbox_inches='tight')
# # 显示图表
# plt.show()





# RSGDA_x_logs = []
# RSGDA_y_logs = []
# RSGDA_true_grad_xs = []
#
# VRAdaGDA_x_logs = []
# VRAdaGDA_y_logs = []
# VRAdaGDA_true_grad_xs = []
#
# # TiAda 和 ours 只执行一次
# # TiAda(xs, ys, 0.005, 0.005, 0.6, 0.4, TiAda_x_log, TiAda_y_log, TiAda_true_grad_x, epoch, grad, df_dx)
# # ours(0.00013, 0.00013, ours_x_log, ours_y_log, ours_true_grad_x, epoch, grad, df_dx)
# #
# # 固定第二个参数
# #
# #
# # 改变 beta 值，从 0.1 到 0.9
# for beta_x in np.arange(1, 1000, 25):
#     # 每次迭代之前初始化 log 列表，并添加初始值 xs 和 ys
#     RSGDA_x_log, RSGDA_y_log, RSGDA_true_grad_x = [xs], [ys], []
#     VRAdaGDA_x_log, VRAdaGDA_y_log, VRAda_true_grad_x = [xs], [ys], []
#
#     # 执行函数，使用当前的 beta 值
#     RSGDA(0.001, 0.0078, beta_x, 512,  RSGDA_x_log, RSGDA_y_log, RSGDA_true_grad_x, epoch, grad, df_dx)
#     VRAdaGDA(0.001 * 4, 0.0084 * 4, beta_x, 512,  VRAdaGDA_x_log, VRAdaGDA_y_log, VRAda_true_grad_x, epoch, grad, df_dx)
#
#     # 保存每次迭代的结果
#     RSGDA_x_logs.append(RSGDA_x_log)
#     RSGDA_y_logs.append(RSGDA_y_log)
#     RSGDA_true_grad_xs.append(RSGDA_true_grad_x)
#
#     VRAdaGDA_x_logs.append(VRAdaGDA_x_log)
#     VRAdaGDA_y_logs.append(VRAdaGDA_y_log)
#     VRAdaGDA_true_grad_xs.append(VRAda_true_grad_x)
#
# # 新的 x_values，对应不同的 beta 值
# beta_values = np.arange(1, 1000, 25)
#
# RSGDA_y_values = [log[-1] for log in RSGDA_true_grad_xs]
# VRAdaGDA_y_values = [log[-1] for log in VRAdaGDA_true_grad_xs]
#
# # TiAda_last_grad = TiAda_true_grad_x[-1]
# # AdaCM_last_grad = ours_true_grad_x[-1]
#
#
# # 可视化
# x_start = min(beta_values)
# x_end = max(beta_values)
#
# # 绘制 RSGDA 和 VRAdaGDA 的结果
# plt.plot(beta_values, RSGDA_y_values, label='RSGDA', marker='o', linewidth=3)
# plt.plot(beta_values, VRAdaGDA_y_values, label='VRAdaGDA', marker='o', linewidth=3)
#
# # 绘制 TiAda 和 AdaCM 的水平直线
# plt.hlines(y=[3.1317822], xmin=x_start, xmax=x_end * 1.01, color='g', linestyle='-', label='TiAda', linewidth=3)
# plt.hlines(y=[2.24574033], xmin=x_start, xmax=x_end * 1.01, color='r', linestyle='-', label='AdaFM', linewidth=3)
#
# # 设置 x 轴和 y 轴标签
# plt.xlabel(r'Value of $\beta_x$', fontsize=15)
# plt.ylabel(r'$\|\nabla_xf(x,y)\|$', fontsize=15)
#
# # 使用科学计数法格式化坐标轴
# plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
# plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
# # plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
#
# # 添加图例和网格
# plt.legend()
# plt.grid(True)
# plt.legend(loc='best', fontsize=15)
# plt.ylim(2.1, 6)  # 设置y轴范围
# plt.xlim(-5, max(beta_values) * 1.01)  # 设置x轴范围
# # 保存图表到文件
# plt.savefig('C:\\Users\\48301\Desktop\\result\\test_function2_change_beta_x_1.png', bbox_inches='tight')
# # 显示图表
# plt.show()



# RSGDA_x_logs = []
# RSGDA_y_logs = []
# RSGDA_true_grad_xs = []
#
# VRAdaGDA_x_logs = []
# VRAdaGDA_y_logs = []
# VRAdaGDA_true_grad_xs = []
# #
# # # TiAda 和 ours 只执行一次
# # TiAda(xs, ys, 0.005, 0.005, 0.6, 0.4, TiAda_x_log, TiAda_y_log, TiAda_true_grad_x_3, epoch, grad1, df1_dx)
# # ours(0.00013, 0.00013, ours_x_log, ours_y_log, ours_true_grad_x, epoch, grad1, df1_dx)
# #
# # # 固定第二个参数
# #
# #
# # # 改变 beta 值，从 0.1 到 0.9
# for beta_y in np.arange(1, 1000, 25):
# #     # 每次迭代之前初始化 log 列表，并添加初始值 xs 和 ys
#     RSGDA_x_log, RSGDA_y_log, RSGDA_true_grad_x = [xs], [ys], []
#     VRAdaGDA_x_log, VRAdaGDA_y_log, VRAda_true_grad_x = [xs], [ys], []
#
#     # 执行函数，使用当前的 beta 值
#     RSGDA(0.001, 0.0078, 512, beta_y,  RSGDA_x_log, RSGDA_y_log, RSGDA_true_grad_x, epoch, grad, df_dx)
#     VRAdaGDA(0.001 * 4, 0.0084 * 4, 512, beta_y,  VRAdaGDA_x_log, VRAdaGDA_y_log, VRAda_true_grad_x, epoch, grad, df_dx)
#
#     # 保存每次迭代的结果
#     RSGDA_x_logs.append(RSGDA_x_log)
#     RSGDA_y_logs.append(RSGDA_y_log)
#     RSGDA_true_grad_xs.append(RSGDA_true_grad_x)
#
#     VRAdaGDA_x_logs.append(VRAdaGDA_x_log)
#     VRAdaGDA_y_logs.append(VRAdaGDA_y_log)
#     VRAdaGDA_true_grad_xs.append(VRAda_true_grad_x)
#
# # 新的 x_values，对应不同的 beta 值
# beta_values = np.arange(1, 1000, 25)
#
# RSGDA_y_values = [log[-1] for log in RSGDA_true_grad_xs]
# VRAdaGDA_y_values = [log[-1] for log in VRAdaGDA_true_grad_xs]
# #
# # TiAda_last_grad = TiAda_true_grad_x_3[-1]
# # AdaCM_last_grad = ours_true_grad_x[-1]
# #
# #
# # # 可视化
# x_start = min(beta_values)
# x_end = max(beta_values)
# #
# # # 绘制 RSGDA 和 VRAdaGDA 的结果
# plt.plot(beta_values, RSGDA_y_values, label='RSGDA', marker='o', linewidth=3)
# plt.plot(beta_values, VRAdaGDA_y_values, label='VRAdaGDA', marker='o', linewidth=3)
# #
# # # 绘制 TiAda 和 AdaCM 的水平直线
# plt.hlines(y=[3.1317822], xmin=x_start, xmax=x_end * 1.01, color='g', linestyle='-', label='TiAda', linewidth=3)
# plt.hlines(y=[2.24574033], xmin=x_start, xmax=x_end * 1.01, color='r', linestyle='-', label='AdaFM', linewidth=3)
# # # 设置 x 轴和 y 轴标签
# plt.xlabel(r'Value of $\beta_y$', fontsize=15)
# plt.ylabel(r'$\|\|\nabla_xf(x,y)\|\|$', fontsize=15)
#
# # 使用科学计数法格式化坐标轴
# plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
# plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
# # plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
#
# # 添加图例和网格1
# plt.legend()
# plt.grid(True)
# plt.legend(loc='best', fontsize=15)
# plt.ylim(2.1, 6)  # 设置y轴范围
# # plt.xlim(-10, max(beta_values) * 1.01)  # 设置x轴范围
# # 保存图表到文件
# plt.savefig('C:\\Users\\48301\Desktop\\result\\test_function2_change_beta_y_2.png', bbox_inches='tight')
# # 显示图表
# plt.show()








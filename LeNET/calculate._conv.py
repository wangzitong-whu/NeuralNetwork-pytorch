import math


# 若图像为正方形：设输入图像尺寸为WxW，卷积核尺寸为FxF，步幅为S，Padding使用P,经过该卷积层后输出的图像尺寸为NxN：
def calculate_N(W, F, S, P):
    return ((W - F + 2 * P) / S) + 1


def calculate_S(W, F, P, N):
    return (W - F + 2 * P) / (N - 1)


def calculate_P(W, F, S, N):
    return ((N - 1) * S - W + F) * 0.5

def calculate_F(W, S, P, N):
    return W+2*P-(N-1)*S

if __name__ == '__main__':
    print(f'经过该卷积层后输出的图像尺寸为{calculate_N(224,3,1,1)}')
    #print(f'所需步长为{calculate_S(5, 3, 1, 5)}')
    #print(f'所需填充为{calculate_P(5, 3, 1, 5)}')
    #print(f'卷积核大小为{calculate_F(5, 1, 1, 1)}')

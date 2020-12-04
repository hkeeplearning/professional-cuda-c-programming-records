

## 一维 DCT 变换

假设输入信号为$f(i)$，长度是$N$，那么第$u$个位置的离散余弦变换为：
$$
DCT(u) = c(u)\sum_{i=0}^{N-1}f(i)cos\left [ {\pi }\frac{(i+0.5)}{N}u  \right ]
$$
其中，
$$
c(u)=\begin{cases} 
  \sqrt{\frac{1}{N} } & \text{ if } u=0 \\
  \sqrt{\frac{2}{N} } & \text{ if } u\ne 0 
\end{cases}
$$

```matlab
function y = dct1(x)

num = length(x);
y = zeros(1, num);
for u = 0: num-1
    c = sqrt(2 / num);
    if u == 0
        c = sqrt(1 / num);
    end
    sum = 0;
    for i = 0: num-1
        sum = sum + x(i + 1) * cos((i + 0.5) * u * pi / num);
    end
    y(u + 1) = c * sum;
end

end
```

## 一维 IDCT 变换

$$
IDCT(i) =  \sum_{u=0}^{N-1}c(u)F(u)cos\left [ \pi \frac{(i + 0.5)}{N}u \right ]
$$

```matlab
function y = idct1(x)

num = length(x);
y = zeros(1, num);
for i = 0: num-1
    sum = 0;
    for u = 0: num-1
        c = sqrt(2 / num);
        if u == 0
            c = sqrt(1 / num);
        end
        sum = sum + x(i + 1) * cos((i + 0.5) * u * pi / num);
    end
    y(u + 1) = sum;
end

end
```



## 二维 DCT 变换

二维 DCT 变换其实是在一维 DCT 变换的基础上再做一次 DCT 变换。
$$
DCT(u, v) = c(u)c(v)\sum_{i=0}^{N-1}\sum_{j=0}^{N-1}f(i, j)cos\left [ {\pi }\frac{(i+0.5)}{N}u  \right ]cos\left [ {\pi }\frac{(j+0.5)}{N}v  \right ]
$$
其中，
$$
c(u)=\begin{cases} 
  \sqrt{\frac{1}{N} } & \text{ if } u=0 \\
  \sqrt{\frac{2}{N} } & \text{ if } u\ne 0 
\end{cases}
$$

$$
c(v)=\begin{cases} 
  \sqrt{\frac{1}{N} } & \text{ if } v=0 \\
  \sqrt{\frac{2}{N} } & \text{ if } v\ne 0 
\end{cases}
$$

如果输入不是方阵，我们可以将矩阵补全重新构造成一个新的方阵，所以通常我们只考虑方阵的二维 DCT 变换。上式可以简化为矩阵的形式：
$$
F = AfA^T
$$
其中，
$$
A(i, j) = c(i)cos\left [\pi  \frac{(j + 0.5)}{N} i \right ]
$$

## 二维 IDCT 变换
$$
IDCT(i, j) = \sum_{i=0}^{N-1}\sum_{j=0}^{N-1}c(u)c(v)F(u, v)cos\left [ {\pi }\frac{(i+0.5)}{N}u  \right ]cos\left [ {\pi }\frac{(j+0.5)}{N}v  \right ]
$$
其中，
$$
c(u)=\begin{cases} 
  \sqrt{\frac{1}{N} } & \text{ if } u=0 \\
  \sqrt{\frac{2}{N} } & \text{ if } u\ne 0 
\end{cases}
$$

$$
c(v)=\begin{cases} 
  \sqrt{\frac{1}{N} } & \text{ if } v=0 \\
  \sqrt{\frac{2}{N} } & \text{ if } v\ne 0 
\end{cases}
$$

矩阵的形式：
$$
f = A^TFA
$$

## DCT 变换的可分离性

DCT 变换是可分离的变换。通常根据可分离性，二维 DCT 可用两次一维 DCT 变换来完成。
import torch, math

frac_bits = {8:3, 16: 7, 32: 8}

def get_minq_maxq(bits: int, sym: bool):
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = torch.tensor(-maxq -1)
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = torch.tensor(0)

    return minq, maxq

def asym_quantize(x: torch.Tensor, bits: int):
    minq, maxq = get_minq_maxq(bits=bits, sym=False)
    xmax = torch.amax(x, dim=-1, keepdim=True)
    xmin = torch.zeros_like(xmax)
    # print("xmax, xmin", xmax, xmin, x.max(), x.min(), maxq)
    # print("sub", xmax - xmin, "max", (xmax - xmin).max())
    # print("clamp", ((xmax - xmin)*0.9).clamp(min=1e-5))
    print("clamp result", ((xmax - xmin)*0.9).max())
    scale = (xmax - xmin)*0.9
    print("scale in this, ", scale, scale.max())
    zero = torch.zeros_like(xmax)
    q = torch.clamp(torch.round((x + zero) / scale), -xmax, xmax)

    return q, scale, zero

def asym_dequantize(q, scale, zero):
    return q * scale - zero

def frac_mult(x, y, bw):
    x=(x*(2**(bw-1))).to(torch.int64)
    y=(y*(2**(bw-1))).to(torch.int64)
    if torch.isnan(x).any():
        print('frac_mult overflow', x.dtype)
    ans = (x * y).to(torch.int64)

    result = (ans/(2**(bw-1))).to(torch.int64)
    return result/(2**(bw-1))

def frac_exp2(x, bw, term):
    # q, scale, zero = asym_quantize(x, bw)
    # result = torch.zeros_like(x)
    # factorial = 1
    ln2 = torch.log(torch.tensor(2))
    scale1 = ln2
    q1 = 1.5 / scale1
    scale2 = scale1 ** 2 / 6
    q2 = 0.625 / scale2
    scale3 = scale2 * ln2
    q3 = 1.0 / scale3
    if term == 3:
        tmp1 = frac_add(x, q1, bw)
        tmp2 = frac_mult(tmp1, tmp1, bw)
        tmp3 = frac_add(tmp2, q2, bw)
        tmp4 = frac_mult(tmp3, x, bw)
        result = frac_add(tmp4, q3, bw)
    else:
        assert False

    return result, scale3

def custom_int_tanh(x, bw, term):
    x = x.to(dtype=torch.float64)
    exp_2x = custom_int_exp(frac_mult(frac_mult(torch.tensor(-2.0), x, bw), x, bw), bw, term)
    #exp_2x = custom_int_exp(x*(-2)*x, 32, 3)
    tanh_x = frac_add(torch.tensor(1.0), -exp_2x, bw) / frac_add(torch.tensor(1.0), exp_2x, bw)
    #tanh_x = (1-exp_2x)/(1+exp_2x)
    return tanh_x

def custom_int_gelu(x, bw, term):
    x = x.to(dtype=torch.float64)
    x_3 = frac_mult(frac_mult(frac_mult(x, x, bw), x, bw), torch.tensor(0.044715), bw)
    #x_3=x*x*x
    # print(x, frac_mult(x, x, bw), x**3)
    tanh = custom_int_tanh(x_3, bw, term)
    tanh_plus1 = frac_add(torch.tensor(1.0), tanh, bw)
    #tanh_plus1 = 1+ tanh
    return frac_mult(frac_mult(torch.tensor(0.5), x, bw), tanh_plus1, bw)
    #return 0.5*x*tanh_plus1

def custom_int_exp(x, bw, term):
    q, scale, zero = asym_quantize(x, bw)
    if scale.max() in [float('inf'), float('-inf')]:
        print('scale overflow', scale, scale.max(), x.max(), x.min())
    fp_x = asym_dequantize(q, scale, zero)
    #print(fp_x)
    input = fp_x*torch.tensor(1.44238)
    int_part = torch.floor(input)
    frac_part = input - int_part
    #print(frac_part)
    #print(int_part)
    max_int_scale = 2 ** int(scale.max() * 0.9)
    #print(max_int_scale)
    q, scale = frac_exp2(frac_part, bw, term)
    q = q * torch.pow(2, int_part) / max_int_scale
    return q, scale * max_int_scale
    
def frac_add(x, y, bw):
    #print(x)
    scale = frac_bits[bw]
    tmp_x=(x*(2**(scale-1))).to(torch.int64)
    #print('x: ', x)
    #print(y)
    tmp_y=(y*(2**(scale-1))).to(torch.int64)
    #print('y: ', y)
    ans = tmp_x + tmp_y
    if(ans >=2 **(bw-1)).any():
      print('addition overflow')
    ans[ans >= 2 ** (bw - 1)] = (2 ** (bw - 1)) - 1
    result = ans.to(torch.int64)
    return result/(2**(scale-1))

def frac_div(x, y, bw):
    #print(x)
    scale = frac_bits[bw]
    tmp_x=(x*(2**(scale-1))).to(torch.int64)
    #print('x: ', x)
    #print(y)
    tmp_y=(y*(2**(scale-1))).to(torch.int64)
    # print(x, y, x/y)
    #print('y: ', y)
    return tmp_x / tmp_y

def custom_int_softmax(x, bw, term):
    x_max = torch.max(x)
    x_norm = x - x_max
    print("before exp", x_max, x.max(), x.min())
    x_exp, s = custom_int_exp(x_norm, bw, term)
    ln2 = torch.log(torch.tensor(2))
    #print("sum should be", x_exp.sum(), x_exp.max(), s)
    x_sum = torch.tensor(0)
    for x_i in x_exp:
        # print(x_i)
        x_sum = frac_add(x_sum, x_i, bw)
        # print(x_i)
        # print(x_exp, x.exp(), x_sum)
        #print("sum actual be", x_sum)

    return frac_div(x_exp, x_sum, bw)
    # return x_exp / x_sum

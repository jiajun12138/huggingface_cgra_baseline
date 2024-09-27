import torch, math

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
    xmin = torch.amin(x, dim=-1, keepdim=True)
    scale = (((xmax - xmin)*0.9).clamp(min=1e-5) / maxq)
    zero = -xmin
    q = torch.clamp(torch.round((x + zero) / scale), 0, maxq)

    return q, scale, zero

def custom_int_gelu(x, bw, term):
    raise NotImplementedError

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
    result = torch.zeros_like(x)
    factorial = 1
    ln2 = torch.log(torch.tensor(2))
    ln2 = (ln2*(2**(bw-1))).to(torch.int)/(2**(bw-1))
    if torch.isnan(ln2).any():
        print('ln2 overflow', ln2.dtype)    
    power = torch.ones_like(x)

    for n in range(term):
        result += power / factorial
        power = frac_mult(power, x, bw)
        power = frac_mult(power, ln2, bw)

        factorial *= (n + 1)

    return result

def custom_int_exp(x, bw, term):
    fp_x = x.to(torch.float64)

    input = fp_x*torch.tensor(1.44238)
    int_part = torch.floor(input)
    frac_part = input - int_part

    result = torch.pow(2, int_part)*frac_exp2(frac_part, bw, term)
    if torch.isnan(frac_exp2(frac_part, bw, term)).any():
        print('2 exp frac overflow')
    if torch.isnan(torch.pow(2, int_part)).any():
        print('2 exp of int overflow', result.dtype)
    return result
  
def frac_add(x, y, bw):
    #x=(x*(2**(bw-1))).to(torch.int64)
    #y=(y*(2**(bw-1))).to(torch.int64)

    x=torch.round(x*(2**(bw-1)))/(2**(bw-1))
    y=torch.round(y*(2**(bw-1)))/(2**(bw-1))
    
    ans = x + y
    if (ans >= 2 ** (2 * bw - 1)).any():
        print('addition overflow', x, y)
    ans[ans >= 2 ** (2 * bw - 1)] = (2 ** (2 * bw - 1)) - 1
    result = torch.round(ans*(2**(bw-1)))/(2**(bw-1))
    #return result/(2**(bw-1))
    return result

def frac_div(x, y, bw):
    #x=(x*(2**(bw-1))).to(torch.int64)
    #y=(y*(2**(bw-1))).to(torch.int64)
    x=torch.round(x*(2**(bw-1)))
    y=torch.round(y*(2**(bw-1)))
    return x / y

def custom_int_softmax(x, bw, term):
    x_max = torch.max(x, -1, keepdim=True)[0]
    x = x - x_max
    x_exp = custom_int_exp(x.to(dtype=torch.float64), bw, term)
    #x_exp = torch.exp(x)
    
    #x_sum = torch.tensor(0)
    #for x_i in x_exp:
    #    x_sum = frac_add(x_sum, x_i, bw)

    if torch.isnan(x_exp).any():
        print('x_exp overflow', x_exp.dtype)
            
    x_exp = torch.round(x_exp*(2**(bw-1)))/(2**(bw-1))
    if torch.isnan(x_exp).any():
        print('x_exp round overflow', x_exp.dtype)
    x_exp = torch.clamp(x_exp, max=(2 ** (2 * bw - 1)) - 1)
    x_sum = torch.sum(x_exp,dim=-1,keepdim=True)

    
    #return x_exp / x_sum
    return frac_div(x_exp, x_sum, bw)
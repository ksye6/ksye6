import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output

w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

def fn(x, y):
    z = torch.matmul(x, w)+b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
    return loss

loss = fn(x, y)

loss.backward()
print(w.grad)
print(b.grad)

print(f"Gradient function for loss = {loss.grad_fn}")

xp = torch.randn(5)  # input tensor
yp = torch.randint(0, 1, (3, )).float()  # expected output

loss = fn(xp, yp)

loss.backward()
print(w.grad)
print(b.grad)

w.grad=None
b.grad=None

loss = fn(xp, yp)

loss.backward()
print(w.grad)
print(b.grad)

# can't backward twice
#loss.backward() # gives error

w.grad=None
b.grad=None

loss = fn(xp, yp)

loss.backward(retain_graph=True)
print(w.grad)
print(b.grad)
loss.backward()
print(w.grad)
print(b.grad)

# second-order

wp = torch.randn(5, 5)
xp = torch.randn(1, 5, requires_grad=True)

def second(x):
    z = xp @ wp @ xp.T
    return z

z = second(xp)
pzpx = torch.autograd.grad(z, xp, create_graph=True)[0]
p2zpx2 = torch.autograd.grad(pzpx.sum(), xp)[0]


# stop gradient

with torch.no_grad():
    z = second(x)
    print(z.requires_grad)

# when to track

x1 = torch.randn(3, 10)
w1 = torch.randn(10, 8)
b1 = torch.randn(8)
w2 = torch.randn(8, 5)
b2 = torch.randn(5).requires_grad_()
w3 = torch.randn(5, 1)
b3 = torch.randn(1)

lin1 = x1 @ w1 + b1
lin2 = lin1 @ w2 + b2
lin3 = lin2 @ w3 + b3

print(lin1.requires_grad)
print(lin2.requires_grad)
print(lin3.requires_grad)

#lin3.numpy() # gives error
lin3 = lin3.detach()
print(lin3.requires_grad)
lin3.numpy()

# item gives the native type

print(lin3[0].item())

# jacobian

x = torch.randn(5)  # input tensor

w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

def fn(x):
    z = torch.matmul(x, w) + b
    return z

jac = torch.autograd.functional.jacobian(fn, x).T

print(torch.allclose(jac, w))



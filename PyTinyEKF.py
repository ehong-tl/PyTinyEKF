import math

def choldc1(a, p, n):
    for i in range(n):
        for j in range(i,n):
            summ = a[i*n+j]
            for k in range(i-1,-1,-1):
                summ -= a[i*n+k] * a[j*n+k]
            if i == j:
                if summ <= 0:
                    return 1, a, p
                p[i] = math.sqrt(summ)
            else:
                a[j*n+i] = summ / p[i]
    return 0, a, p

def choldcs1(A, a, p, n):
    for i in range(n):
        for j in range(n):
            a[i*n+j] = A[i*n+j]
    rtn, a, p = choldc1(a,p,n)
    if rtn:
        return 1, A, a, p
    for i in range(n):
        a[i*n+i] = 1 / p[i]
        for j in range(i+1,n):
            summ = 0.0
            for k in range(i,j):
                summ -= a[j*n+k] * a[k*n+i]
            a[j*n+i] = summ / p[j]
    return 0, A, a, p

def cholsl(A, a, p, n):
    rtn, A, a, p = choldcs1(A, a, p, n)
    if rtn:
        return 1, a
    for i in range(n):
        for j in range(i+1,n):
            a[i*n+j] = 0.0
    for i in range(n):
        a[i*n+i] *= a[i*n+i]
        for k in range(i+1,n):
            a[i*n+i] += a[k*n+i] * a[k*n+i]
        for j in range(i+1,n):
            for k in range(j,n):
                a[i*n+j] += a[k*n+i] * a[k*n+j]
    for i in range(n):
        for j in range(i):
            a[i*n+j] = a[j*n+i]
    return 0, a

def zeros(m, n):
    a = []
    for j in range(m*n):
        a.append(0.0)
    return a

def mulmat(a, b, c, arows, acols, bcols):
    for i in range(arows):
        for j in range(bcols):
            c[i*bcols+j] = 0.0
            for l in range(acols):
                c[i*bcols+j] += a[i*acols+l] * b[l*bcols+j]
    return c

def mulvec(a, x, y, m, n):
    for i in range(m):
        y[i] = 0.0
        for j in range(n):
            y[i] += x[j] * a[i*n+j]
    return y

def transpose(a, at, m, n):
    for i in range(m):
        for j in range(n):
            at[j*m+i] = a[i*n+j]
    return at

def accum(a, b, m, n):
    for i in range(m):
        for j in range(n):
            a[i*n+j] += b[i*n+j]
    return a

def add(a, b, c, n):
    for j in range(n):
        c[j] = a[j] + b[j]
    return c

def sub(a, b, c, n):
    for j in range(n):
        c[j] = a[j] - b[j]
    return c

def negate(a, m, n):
    for i in range(m):
        for j in range(n):
            a[i*n+j] = -a[i*n+j]
    return a

def mat_addeye(a, n):
    for i in range(n):
        a[i*n+i] += 1
    return a

class model:

    def __init__(self, n, m):
        self.x = []
        self.fx = []
        self.F = []
        self.hx = []
        self.H = []
        self.P = []
        self.Q = []
        self.R = []
        for i in range(n):
            self.x.append(0.0)
            self.fx.append(0.0)
            self.F.append([])
            self.P.append([])
            self.Q.append([])
            for j in range(n):
                self.F[i].append(0.0)
                self.P[i].append(0.0)
                self.Q[i].append(0.0)
        for i in range(m):
            self.hx.append(0.0)
            self.H.append([])
            self.R.append([])
            for j in range(n):
                self.H[i].append(0.0)
            for k in range(m):
                self.R[i].append(0.0)

    def setfx(self, i, val):
        self.fx[i] = val

    def setF(self, i, j, val):
        self.F[i][j] = val

    def sethx(self, i, val):
        self.hx[i] = val

    def setH(self, i, j, val):
        self.H[i][j] = val

    def setP(self, i, j, val):
        self.P[i][j] = val

    def setQ(self, i, j, val):
        self.Q[i][j] = val

    def setR(self, i, j, val):
        self.R[i][j] = val

    def flatten(self):
        self.F_f = sum(self.F, [])
        self.H_f = sum(self.H, [])
        self.Q_f = sum(self.Q, [])
        self.R_f = sum(self.R, [])

    def flattenP(self):
        self.P_f = sum(self.P, [])

class ekf(model):

    def __init__(self, n, m):
        model.__init__(self, n, m)
        self.n = n
        self.m = m
        self.G = zeros(n, m)
        self.Ht = zeros(n, m)
        self.Ft = zeros(n, n)
        self.Pp = zeros(n, n)
        self.tmp0 = zeros(n, n)
        self.tmp1 = zeros(n, m)
        self.tmp2 = zeros(m, n)
        self.tmp3 = zeros(m, m)
        self.tmp4 = zeros(m, m)
        self.tmp5 = zeros(m, m)

    def predict(self):
        self.tmp0 = mulmat(self.F_f, self.P_f, self.tmp0, self.n, self.n, self.n)
        self.Ft = transpose(self.F_f, self.Ft, self.n, self.n)
        self.Pp = mulmat(self.tmp0, self.Ft, self.Pp, self.n, self.n, self.n)
        self.Pp = accum(self.Pp, self.Q_f, self.n, self.n)

    def correct(self):
        self.Ht = transpose(self.H_f, self.Ht, self.m, self.n)
        self.tmp1 = mulmat(self.Pp, self.Ht, self.tmp1, self.n, self.n, self.m)
        self.tmp2 = mulmat(self.H_f, self.Pp, self.tmp2, self.m, self.n, self.n)
        self.tmp3 = mulmat(self.tmp2, self.Ht, self.tmp3, self.m, self.n, self.m)
        self.tmp3 = accum(self.tmp3, self.R_f, self.m, self.m)
        rtn, self.tmp4 = cholsl(self.tmp3, self.tmp4, self.tmp5, self.m)
        if rtn:
            return 1
        self.G = mulmat(self.tmp1, self.tmp4, self.G, self.n, self.m, self.m)
        
        self.tmp5 = sub(self.z, self.hx, self.tmp5, self.m)
        self.tmp2 = mulvec(self.G, self.tmp5, self.tmp2, self.n, self.m)
        self.x = add(self.fx, self.tmp2, self.x, self.n)
        
        self.tmp0 = mulmat(self.G, self.H_f, self.tmp0, self.n, self.m, self.n)
        self.tmp0 = negate(self.tmp0, self.n, self.n)
        self.tmp0 = mat_addeye(self.tmp0, self.n)
        self.P = mulmat(self.tmp0, self.Pp, self.P_f, self.n, self.n, self.n)

    def step(self, z):
        self.z = z
        self.predict()
        self.correct()
        return 0

    def getX(self):
        return self.x

    def getP(self):
        return self.P

    def getG(self):
        return self.G

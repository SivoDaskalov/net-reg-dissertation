import matlab.engine
import numpy as np

eng = matlab.engine.start_matlab("-nodesktop")
X = np.random.rand(16, 8)
y = np.random.rand(16, 1)
m_X = matlab.double(X.tolist())
m_y = matlab.double(y.tolist())
coef = eng.testcvxfunc(m_X, m_y)
print(coef)
eng.exit()

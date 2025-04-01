import numpy as np
import matplotlib.pyplot as plt

data = np.load('log1d.npz')
log50 = data['log50']  # LoG 필터 (σ=50)
gauss50 = data['gauss50']  # Gaussian 필터 (σ=50)
gauss53 = data['gauss53']  # Gaussian 필터 (σ=53)

# 시각화
plt.figure(figsize=(10, 5))
plt.plot(log50, label="LoG (σ=50)", linestyle='dashed')
plt.plot(gauss50, label="Gaussian (σ=50)")
plt.plot(gauss53, label="Gaussian (σ=53)")
plt.plot(gauss53 - gauss50, label="DoG (σ=50, 53)", linestyle='dotted')
plt.legend()
plt.title("Comparison of LoG and DoG")
plt.show()
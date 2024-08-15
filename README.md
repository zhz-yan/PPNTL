# PPNTL

代码和数据已上传，正在准备说明文档。

The code and data have been uploaded, and the README is being prepared...

# 准备 Requirement
```bash
pip install xgboost
pip install scikit-learn
pip install numpy
pip install mpi4py
```

# 说明
本代码的运行基于mpi4py模块，请运行前安装该模块。安装教程参考: [Linux环境下配置mpi4py
](https://blog.csdn.net/monster7777777/article/details/124001248)

# 数据
本文数据位于 'data/' 文件中。

# 运行

### 运行基于秘密共享的分布式XGBoost算法，请执行
'''bash
mpiexec -n 5 python VerticalXGBoost.py
'''

### 运行经典XGBoost算法，请执行
'''bash
python XGBoost.py
'''

## 感谢
在实现过程中，我们的代码主要基于Lunchen Xie的[MP-FedXGB](https://github.com/HikariX/MP-FedXGB),非常感谢这些作者的工作.

During the implementation we base our code mostly on the [MP-FedXGB](https://github.com/HikariX/MP-FedXGB) by Lunchen Xie. Many thanks to these authors for their great work!

1.Spark“数字人体”AI挑战赛——脊柱疾病智能诊断大赛
2.本项目依赖于nn_tools，可以在github本人（wolaituodiban）项目中中自行搜索下载  
3.文件结构  
--code  
&nbsp;|--core  
&nbsp;|--disease        症状分类相关代码（因为是baseline，椎间盘默认预测v1，锥体默认预测v2)  
&nbsp;|--key_point      定位相关代码  
&nbsp;|--static_files   定义输出格式的静态文件  
&nbsp;|--structure      封装了DICOM的类，包含基础的自动寻找T2序列和中间帧的功能，拓展了一些简单空间几何变换函数  
&nbsp;|--data_utils.py  数据处理的常用函数  
&nbsp;|--dicom_utils.py 读取dicom文件的常用函数  
&nbsp;|--visiliation.py 简单的可视化函数  
&nbsp;|--main.py            入口，请在项目根目录下运行python -m code.main  
--data  
&nbsp;|--lumbar_train150    训练集  
&nbsp;|--train              校验集  
&nbsp;|--lumbar_testA50     A榜测试集  
--models                  存放模型文件的目录  
--predictions             存放预测结果的目录  
--requirements.txt        项目的依赖，请自行安装      
4.关于环境安装的建议  
conda create -n py37torch15 python=3.7  
conda activate py37torch15  
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch  
pip install -r requirements.txt
import numpy as np
import pandas as pd
import jieba
import jieba.analyse
import codecs
 
import sys
import json
import re
# 目标 转化成
"""
JSON example:
{
    "doc_label": ["Computer--MachineLearning--DeepLearning", "Neuro--ComputationalNeuro"],
    "doc_token": ["I", "love", "deep", "learning"],
    "doc_keyword": ["deep learning"],
    "doc_topic": ["AI", "Machine learning"]
}
"""
 
# 读入数据
input_path ="../data/"
file_name1 = input_path+"train_set.csv"
df1 = pd.read_csv(file_name1,header=None)
df1.columns = ['doc_label','doc_token']
df1.shape
 
df1
 
# 将df转换成两份  并且把数据打乱
df2 = df1.sample(frac=1)
df2
 
# 一部分作为训练街
train_df = df2[:20000]
# 一部分作为验证集
valid_df = df2[20000:]
 
train_df.shape
 
valid_df.shape
 
# 停用词
def stop_words(path):
    with open(path) as f:
        return [l.strip() for l in f]
 
output_path = '../data/train_set.json'
 
with open(output_path,"w+",encoding='utf-8') as f:
#with open(output_path, "w") as fw:
    for indexs in train_df.index:
        dict1 = {}
        dict1['doc_label'] = [str(train_df.loc[indexs].values[0])]
        doc_token = train_df.loc[indexs].values[1]
        # 只保留中文、大小写字母和阿拉伯数字
        reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
        doc_token =re.sub(reg, '', doc_token)
        print(doc_token)
        # 中文分词
        seg_list  = jieba.cut(doc_token, cut_all=False)
        # 去除停用词
        content = [x for x in seg_list if x not in stop_words('../data/stop_words.txt')]
        dict1['doc_token'] = content
        dict1['doc_keyword'] = []
        dict1['doc_topic'] = []
        # 组合成字典
        print(dict1)
        # 将字典转化成字符串
        json_str = json.dumps(dict1, ensure_ascii=False)
        f.write('%s\n' % json_str)
        # 将字符串 转换为 字典
        #new_dict = json.loads(json_str)
        #将数据写入json文件中
        #f.write('%s\n' % new_dict)
        #json.dump(new_dict,f,ensure_ascii=False,sort_keys=True, indent=4)
        #json.dump(new_dict,fw)
 
output_path = '../data/valida_set.json'
 
with open(output_path,"w+",encoding='utf-8') as f:
#with open(output_path, "w") as fw:
    for indexs in valid_df.index:
        dict1 = {}
        dict1['doc_label'] = str(valid_df.loc[indexs].values[0])
        doc_token = valid_df.loc[indexs].values[1]
        # 去除空格
        reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
        doc_token =re.sub(reg, '', doc_token)
        print(doc_token)
        # 中文分词
        seg_list  = jieba.cut(doc_token, cut_all=False)
        # 去除停用词
        content = [x for x in seg_list if x not in stop_words('../data/stop_words.txt')]
        dict1['doc_token'] = content
        dict1['doc_keyword'] = []
        dict1['doc_topic'] = []
        # 组合成字典
        print(dict1)
        # 将字典转化成字符串
        json_str = json.dumps(dict1, ensure_ascii=False)
        f.write('%s\n' % json_str)
        # 将字符串 转换为 字典
        #new_dict = json.loads(json_str)
        #将数据写入json文件中
        #f.write('%s\n' % new_dict)
        #json.dump(new_dict,f,ensure_ascii=False,sort_keys=True, indent=4)
        #json.dump(new_dict,fw)
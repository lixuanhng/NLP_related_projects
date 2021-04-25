# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   test.py
 
@Time    :   2019-10-31 20:54
 
@Desc    :
 
'''

from Chatbot_Retrieval_model.Bert_sim import predicts


sentences = [['长的清新是什么意思', '小清新的意思是什么']]

for sentence in sentences:
    dic = predicts(sentence)
    print(dic)

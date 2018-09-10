import numpy as np
import tensorflow as tf

import os
import pickle
import random


features = pickle.load(open('features.p', 'rb'))
target_values = pickle.load(open('target.p', 'rb'))
title_length, title_set, genres2int, features, target_values,ratings, users,\
 movies, data, movies_orig, users_orig = pickle.load(open('params.p',mode='rb'))
#电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第五行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i,val in enumerate(movies.values)}
sentences_size = title_length #16
load_dir = './save_model/'
movie_feature_size = user_feature_size = 512
movie_matrix_path = 'movie_matrix.p'
user_matrix_path = 'user_matrix.p'


#获取 Tensors
def get_tensors(loaded_graph):
	uid = loaded_graph.get_tensor_by_name('uid:0')
	user_gender = loaded_graph.get_tensor_by_name('user_gender:0')
	user_age = loaded_graph.get_tensor_by_name('user_age:0')
	user_job = loaded_graph.get_tensor_by_name('user_job:0')
	movie_id = loaded_graph.get_tensor_by_name('movie_id:0')
	movie_categories = loaded_graph.get_tensor_by_name('movie_categories:0')
	movie_titles = loaded_graph.get_tensor_by_name('movie_titles:0')
	targets = loaded_graph.get_tensor_by_name('targets:0')
	dropout_keep_prob = loaded_graph.get_tensor_by_name('dropout_keep_prob:0')

	inference = loaded_graph.get_tensor_by_name('inference/MatMul:0')
	movie_combine_layer_flat = loaded_graph.get_tensor_by_name('movie_fc/Reshape:0')
	user_combine_layer_flat = loaded_graph.get_tensor_by_name('user_fc/Reshape:0')
	return uid, user_gender, user_age, user_job, movie_id, movie_categories,movie_titles, targets,\
		dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat

#预测指定用户对指定电影的评分
#这部分就是对网络做正向传播，计算得到预测的评分
def rating_movie(user_id, movie_id_val):
	loaded_graph = tf.Graph()
	with tf.Session(graph=loaded_graph) as sess:
		#load save model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)
		#get tensors from loaded model
		uid, user_gender, user_age, user_job, movie_id, movie_categories,movie_titles, targets,\
		dropout_keep_prob, inference,_,__ = get_tensors(loaded_graph)
		categories = np.zeros([1, 19])
		categories[0] = movies.values[movieid2idx[movie_id_val]][2]
		titles = np.zeros([1, sentences_size])
		titles[0] = movies.values[movieid2idx[movie_id_val]][1]
		feed = {
			uid: np.reshape(users.values[user_id-1][0], [1, 1]),
			user_gender: np.reshape(users.values[user_id-1][1], [1, 1]),
			user_age: np.reshape(users.values[user_id-1][2], [1, 1]),
			user_job: np.reshape(users.values[user_id-1][3], [1, 1]),
			movie_id: np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
			movie_categories: categories,  #x.take(6,1)
			movie_titles: titles,  #x.take(5,1)
			dropout_keep_prob: 1
		}
		#get prediction
		inference_val = sess.run([inference], feed)
		return (inference_val)

#生成movie特征矩阵，将训练好的电影特征组合成电影特征矩阵并保存到本地
#对每个电影进行正向传播
def save_movie_feature_matrix():
	loaded_graph = tf.Graph()
	movie_matrics = []
	with tf.Session(graph=loaded_graph) as sess:
		#load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		#get tensor from loaded model
		uid, user_gender, user_age, user_job, movie_id, \
		movie_categories, movie_titles, targets, dropout_keep_prob,\
		_, movie_combine_layer_flat, __ = get_tensors(loaded_graph)

		for item in movies.values:
			categories = np.zeros([1, 19])
			categories[0] = item.take(2)
			titles = np.zeros([1, sentences_size])
			titles[0] = item.take(1)
			feed = {
				movie_id: np.reshape(item.take(0), [1, 1]),
				movie_categories:categories,#x.take(6,1)
				movie_titles:titles, #x.take(5, 1)
				dropout_keep_prob: 1,
			}
			movie_representation = sess.run([
				movie_combine_layer_flat], feed)
			movie_matrics.append(movie_representation)
	movie_matrics = np.array(movie_matrics).reshape(-1, movie_feature_size)
	pickle.dump(movie_matrics, open(movie_matrix_path,'wb'))


#生成user特征矩阵
#将训练好的用户特征组合成用户特征矩阵并保存到本地
#对每个用户进行正向传播
def save_user_feature_matrix():
	loaded_graph = tf.Graph()
	users_matrics = []
	with tf.Session(graph=loaded_graph) as sess:
		#load saved model
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)
		uid, user_gender, user_age, user_job, movie_id, \
		movie_categories, movie_titles, targets, dropout_keep_prob,\
		_, __, user_combine_layer_flat = get_tensors(loaded_graph)
		for item in users.values:
			feed = {
				uid:np.reshape(item.take(0), [1, 1]),
				user_gender: np.reshape(item.take(1), [1, 1]),
				user_age: np.reshape(item.take(2), [1, 1]),
				user_job: np.reshape(item.take(3), [1, 1]),
				dropout_keep_prob: 1
			}
			user_representation = sess.run([user_combine_layer_flat], feed)
			users_matrics.append(user_representation)
	users_matrics = np.array(users_matrics).reshape(-1, user_feature_size)
	pickle.dump(users_matrics, open(user_matrix_path, 'wb'))



def load_feature_matrix(path):
	if(os.path.exists(path)):
		pass
	elif path == movie_matrix_path:
		save_movie_feature_matrix()
	else:
		save_user_feature_matrix()
	return pickle.load(open(path, 'rb'))



#使用电影特征矩阵推荐同类型的电影
#思路是计算指定电影的特征向量与整个电影特征矩阵的余弦相似度，
#取相似度最大的top_k个，
#ToDo: 加入随机选择，保证每次的推荐稍微不同
def recommend_same_type_movie(movie_id, top_k=5):
	loaded_graph = tf.Graph()
	movie_matrics = load_feature_matrix(movie_matrix_path)
	movie_feature = movie_matrics[movieid2idx[movie_id]].reshape([1, movie_feature_size])#给定电影的representation

	with tf.Session(graph=loaded_graph) as sess:
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		# 计算余弦相似度
		norm_movie_matrics = tf.sqrt(tf.reduce_sum(
			tf.square(movie_matrics), 1, keep_dims=True)) #计算每个representation的长度 ||x||
		normalized_movie_matrics = movie_matrics / (norm_movie_matrics*norm_movie_matrics[movie_id])
		probs_similarity = tf.matmul(movie_feature, tf.transpose(normalized_movie_matrics))
		#得到对于给定的movie id，所有电影对它的余弦相似值
		sim = probs_similarity.eval()

	print('和电影：{} 相似的电影有：\n'.format(movies_orig[movieid2idx[movie_id]]))

	sim = np.squeeze(sim)#将二维sim转为一维
	res_list = np.argsort(-sim)[:top_k] #获取余弦相似度最大的前top k个movie信息
	results = list()
	for res in res_list:
		movie_info = movies_orig[res]
		results.append(movie_info)
	print(results)
	return results


#给定指定用户，推荐其喜欢的电影
#思路是使用用户特征向量与电影特征矩阵计算所有电影的评分，
#取评分最高的top_k个，
# ToDo 加入随机选择
def recommend_your_favorite_movie(user_id, top_k=5):
	loaded_graph = tf.Graph()
	movie_matrics = load_feature_matrix(movie_matrix_path)
	users_matrics = load_feature_matrix(user_matrix_path)
	user_feature = users_matrics[user_id-1].reshape([1, user_feature_size])#是否需要减一

	with tf.Session(graph=loaded_graph) as sess:
		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		#获取图中的 inference，然后用sess运行
		probs_similarity = tf.matmul(user_feature, tf.transpose(movie_matrics))
		sim = (probs_similarity.eval())
		sim = np.squeeze(sim)
		res_list = np.argsort(-sim)[:top_k] #获取该用户对所有电影可能评分最高的top k
		results = []
		for res in res_list:
			moive_info = movies_orig[res]
			results.append(moive_info)
		print('以下是给您的推荐：', results)
		return results




#看过这个电影的人还可能（喜欢）哪些电影
#首先选出喜欢某个电影的top_k个人，得到这几个人的用户特征向量
#然后计算这几个人对所有电影的评分
#选择每个人评分最高的电影作为推荐
# ToDo 加入随机选择
def recommend_other_favorite_movie(movie_id, top_k=5):
	loaded_graph = tf.Graph()
	movie_matrics = load_feature_matrix(movie_matrix_path)
	users_matrics = load_feature_matrix(user_matrix_path)
	movie_feature = (movie_matrics[movieid2idx[movie_id]]).reshape([1, movie_feature_size])
	print('您看的电影是：{}'.format(movies_orig[movieid2idx[movie_id]]))

	with tf.Session(graph=loaded_graph) as sess:

		loader = tf.train.import_meta_graph(load_dir + '.meta')
		loader.restore(sess, load_dir)

		#计算对给定movie，所有用户对其可能的评分
		users_inference = tf.matmul(movie_feature, tf.transpose(users_matrics))
		favorite_users_id = np.argsort(users_inference.eval())[0][-top_k:]
		print('喜欢看这个电影的人是：{}'.format(users_orig[favorite_users_id-1])) #user_id 处理时是否需要减一

		results = []
		for user in favorite_users_id:
			movies = recommend_your_favorite_movie(user, top_k=2)
			results.extend(movies)
		# print('喜欢这个电影的人还喜欢：', results)
		return results









#test every recommendation functions here

#预测给定user对给定movie的评分
#prediction_rating = rating_movie(user_id=123, movie_id=1234)
#print('for user:123, predicting the rating for movie:1234', prediction_rating)

#生成user和movie的特征矩阵，并存储到本地
# save_movie_feature_matrix()
#save_user_feature_matrix()

#对给定的电影，推荐相同类型的其他top k 个电影
#results = recommend_same_type_movie(movie_id=666, top_k=5)

#对给定用户，推荐其可能喜欢的top k个电影
#results = recommend_your_favorite_movie(user_id=222, top_k=5)

#看过这个电影的人还可能喜欢看那些电影
recommend_other_favorite_movie(movie_id=666, top_k=5)















# 加载数据并保存到本地
# title_length：Title字段的长度（16）
# title_set：Title文本的集合
# genres2int：电影类型转数字的字典
# features：是输入X
# targets_values：是学习目标y
# ratings：评分数据集的Pandas对象
# users：用户数据集的Pandas对象
# movies：电影数据的Pandas对象
# data：三个数据集组合在一起的Pandas对象
# movies_orig：没有做数据处理的原始电影数据
# users_orig：没有做数据处理的原始用户数据

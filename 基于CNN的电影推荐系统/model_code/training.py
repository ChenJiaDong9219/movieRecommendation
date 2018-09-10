import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import pickle

import matplotlib.pyplot as plt
import time
import datetime

import model.movie_nn as movie_nn
import model.user_nn as user_nn



tf.reset_default_graph()
train_graph = tf.Graph()

features = pickle.load(open('features.p', 'rb'))
target_values = pickle.load(open('target.p', 'rb'))
# title_length, title_set, genres2int, features, target_values,ratings, users,\
#  movies, data, movies_orig, users_orig = pickle.load(open('params.p', 'rb'))


#超参
num_epochs = 1 #  = 5
batch_size = 256
dropout_keep = 0.5
learning_rate = 0.0001
show_every_n_batches = 40#show stats for every n number of batches
save_dir = './save_model/'


#电影名单词长度
title_length = 16



def get_targets():
	targets = tf.placeholder(tf.int32, [None, 1], name="targets")
	return targets


def get_batches(Xs, ys, batch_size):
	for start in range(0, len(Xs), batch_size):
		end = min(start + batch_size, len(Xs))
		yield Xs[start:end], ys[start:end]



with train_graph.as_default():
	global_step = tf.Variable(0, name='global_step', trainable=True)
	targets = get_targets()

	#获取user和movie的input placeholders
	uid, user_gender, user_age, user_job = user_nn.get_inputs()
	movie_id, movie_categories, movie_titles,dropout_keep_prob = movie_nn.get_inputs()

	# 获取User的4个嵌入向量和user feature representation
	uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = \
		user_nn.get_user_embedding(uid, user_gender, user_age, user_job)
	_, user_combine_layer_flat = \
		user_nn.get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer)

	#获取movie id embedding, genres embedding, title representation，然后得到整个movie的representation
	movie_id_embed_layer = movie_nn.get_movie_id_embed_layer(movie_id)
	movie_categories_embed_layer = movie_nn.get_movie_categories_embed_layer(movie_categories)
	_, dropout_layer = movie_nn.get_movie_cnn_layer(movie_titles,dropout_keep_prob)
	_, movie_combine_layer_flat = movie_nn.get_movie_feature_layer(movie_id_embed_layer,
															movie_categories_embed_layer, dropout_layer)

	with tf.name_scope('inference'):
		# 将用户特征和电影特征作为输入，进行矩阵乘法，得到一个值,即为预测的ranking。（考虑继续使用一个神经网络？）
		inference = tf.matmul(user_combine_layer_flat,
			tf.transpose(movie_combine_layer_flat))

	with tf.name_scope('loss'):
		# 使用平方损失函数定义整个模型的损失值
		cost = tf.losses.mean_squared_error(targets, inference)
		loss = tf.reduce_mean(cost)

	# optimizer = tf.train.AdamOptimizer(learning_rate)
	# gradients = optimizer.compute_gradients(loss)#return a list of (gradients, variable) pairs
	# train_op = optimizer.apply_gradients(gradients, global_step=global_step)

	train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss)



#训练网络
losses = {'train':[],'test':[]}
with tf.Session(graph=train_graph) as sess:

	#tensorboard记录数据
	#output directory for models and summaries
	timestamp = str(int(time.time()))
	out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
	print('writing to {}\n'.format(out_dir))

	#summaries for loss
	loss_summary = tf.summary.scalar('loss',loss)

	#train summaries
	train_summary_op = tf.summary.merge([loss_summary])
	train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
	train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

	#test inference summaries
	inference_summary_op = tf.summary.merge([loss_summary])
	inference_summary_dir = os.path.join(out_dir, 'summaries', 'inference')
	inference_summary_writer = tf.summary.FileWriter(inference_summary_dir, sess.graph)


	#开始training
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	for epoch_i in range(num_epochs):#5
		#将数据集分为训练集和测试集，随机种子不固定
		train_X, test_X, train_y, test_y = train_test_split(\
			features, target_values, test_size=0.2, random_state=0)
		train_batches = get_batches(train_X, train_y, batch_size)
		test_batches = get_batches(test_X, test_y, batch_size)

		#训练的迭代， 保存训练损失for tensorboard
		for batch_i in range(len(train_X) // batch_size):
			x,y = next(train_batches)#返回迭代器的下一个batch。
			categories = np.zeros([batch_size, 19])
			for i in range(batch_size):
				categories[i] = x.take(6,1)[i]#取得batch中每个电影的分类 for feed
			titles = np.zeros([batch_size, title_length])
			for i in range(batch_size):
				titles[i] = x.take(5,1)[i]#取得batch中每个电影的title for feed
			feed = {
				uid: np.reshape(x.take(0,1), [batch_size, 1]),
				user_gender: np.reshape(x.take(2,1), [batch_size, 1]),
				user_age: np.reshape(x.take(3,1), [batch_size, 1]),
				user_job: np.reshape(x.take(4,1), [batch_size, 1]),
			#	user_job: np.zeros([batch_size, 1]),
				movie_id: np.reshape(x.take(1,1), [batch_size, 1]),
				movie_categories: categories,  #x.take(6,1)
				movie_titles: titles,  #x.take(5,1)
				targets: np.reshape(y, [batch_size, 1]),
				dropout_keep_prob: dropout_keep, #dropout_keep
			}

			step, train_loss, summaries, _ = sess.run([
				global_step, loss, train_summary_op, train_op], feed_dict=feed)
			losses['train'].append(train_loss)
			train_summary_writer.add_summary(summaries, step)

			#show loss every n batches
			if batch_i % show_every_n_batches == 0:

				prediction = inference.eval(feed)
				print('training prediction: %.2f,  expection: %d'%(prediction[0][0], y[0][0]))

				time_str = datetime.datetime.now().isoformat()
				print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(time_str,
				epoch_i+1, batch_i, (len(train_X) // batch_size), train_loss))

		#使用测试数据集对本次的epoch训练的模型进行测试
		for batch_i in range(len(test_X) // batch_size):
			x, y = next(test_batches)
			categories = np.zeros([batch_size, 19])
			for i in range(batch_size):
				categories[i] = x.take(6,1)[i]
			titles = np.zeros([batch_size, title_length])
			for i in range(batch_size):
				titles[i] = x.take(5,1)[i]
			feed = {
				uid: np.reshape(x.take(0,1), [batch_size, 1]),
				user_gender: np.reshape(x.take(2,1), [batch_size, 1]),
				user_age: np.reshape(x.take(3,1), [batch_size, 1]),
				user_job: np.reshape(x.take(4,1), [batch_size, 1]),
				movie_id: np.reshape(x.take(1,1), [batch_size, 1]),
				movie_categories: categories,  #x.take(6,1)
				movie_titles: titles,  #x.take(5,1)
				targets: np.reshape(y, [batch_size, 1]),
				dropout_keep_prob: 1,
			}
			step, test_loss, summaries = sess.run([
				global_step, loss, inference_summary_op],feed)

			#保存测试损失
			losses['test'].append(test_loss)
			inference_summary_writer.add_summary(summaries, step)
			time_str = datetime.datetime.now().isoformat()
			if batch_i % show_every_n_batches == 0:
				prediction = inference.eval(feed)
				print('test prediction: %.2f,  expection: %d' % (prediction[0][0], y[0][0]))
				print('{}: Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
					time_str,epoch_i+1, batch_i, (len(test_X) // batch_size),test_loss))


	#save model
	saver.save(sess, save_dir)
	print('model trained and saved')




# plt.plot(losses['train'], label='training loss')
# plt.legend()
# _ = plt.ylim()


# plt.plot(losses['test'], label='test loss')
# plt.legend()
# _ = plt.ylim()

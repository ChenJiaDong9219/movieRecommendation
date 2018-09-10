import tensorflow as tf
import pickle


features = pickle.load(open('features.p', 'rb'))
#feature info: ['UserID' 'MovieID' 'Gender' 'Age' 'JobID' 'Title' 'Genres']
title_length, title_set, genres2int, features, target_values,ratings, users,\
 movies, data, movies_orig, users_orig = pickle.load(open('params.p', 'rb'))

movie_id_num, title_length, title_vocb_num, genres_num = pickle.load(open('argument.p', 'rb'))

MOVIE_CATEGORAY_LENGTH = 19
MOVIE_TITLE_LENGTH = 16

embed_dim = 64
#电影ID个数
movie_id_max = movie_id_num #3953
#电影类型个数
movies_categories_max = genres_num #18
#电影名单词个数 = title_vocb_num = 5216


#电影名长度  title_length = 16

#文本卷积数量
filter_num = 8 #* 2  #为保证输出len(window_sizes) * filter_num = 64 进行的修改

#电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第五行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i,val in enumerate(movies.values)}








def get_inputs():
	'''
	获取movie所有特征的input
	:return:
	'''
	movie_id = tf.placeholder(tf.int32, [None,1], name='movie_id')
	movie_categories = tf.placeholder(tf.int32, [None, MOVIE_CATEGORAY_LENGTH], name='movie_categories')
	movie_titles = tf.placeholder(tf.int32, [None,MOVIE_TITLE_LENGTH], name='movie_titles')
	dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
	return movie_id, movie_categories, movie_titles, dropout_keep_prob


def get_movie_id_embed_layer(movie_id):
	'''
	获取movie id 的embedding
	'''
	with tf.name_scope('movie_embedding'):
		movie_id_embed_matrix = tf.Variable(tf.random_uniform([
			movie_id_max, embed_dim], -1, 1), name='movie_id_embed_matrix')
		movie_id_embed_layer = tf.nn.embedding_lookup(
			movie_id_embed_matrix, movie_id, name='movie_id_embed_layer')
	return movie_id_embed_layer

#对电影类型的多个嵌入向量做加和
def get_movie_categories_embed_layer(movie_categories, combiner = 'sum'):
	'''
	定义对movie类型的embedding，同时对于一个movie的所有类型，进行combiner的组合。
	目前仅考虑combiner为sum的情况，即将该电影所有的类型进行sum求和
	'''
	with tf.name_scope('movie_categories_layer'):
		movie_categories_embed_matrix = tf.Variable(tf.random_uniform([
			movies_categories_max, embed_dim], -1, 1),
			name='movie_categories_embed_matrix')
		movie_categories_embed_layer = tf.nn.embedding_lookup(
			movie_categories_embed_matrix, movie_categories,
			name='movie_categories_embed_layer')
		if combiner == 'sum':
			movie_categories_embed_layer = tf.reduce_sum(
				movie_categories_embed_layer, axis=1, keep_dims=True)
	return movie_categories_embed_layer


def get_movie_cnn_layer(movie_titles, dropout_keep_prob, window_sizes = [3,4,5,6]):
	'''
	对movie的title，进行卷积神经网络实现
	window_sizes:  文本卷积滑动窗口，分别滑动3,4,5, 6个单词
	'''
	#从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
	with tf.name_scope('movie_embedding'):
		movie_title_embed_matrix = tf.Variable(tf.random_uniform([
			title_vocb_num, embed_dim], -1, 1),
			name='movie_title_embed_matrix')
		movie_title_embed_layer = tf.nn.embedding_lookup(
			movie_title_embed_matrix, movie_titles, name='movie_title_embed_layer')
		movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)#title的二维representation矩阵

	#对文本嵌入层使用不同尺寸的卷积核做卷积核最大池化
	pool_layer_lst = []
	for window_size in window_sizes:
		with tf.name_scope('movie_txt_conv_maxpool_{}'.format(window_size)):
			filter_weights = tf.Variable(tf.truncated_normal([
				window_size, embed_dim, 1, filter_num], stddev=0.1), name='filter_weights')#修改卷积核大小
			filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num], name='filter_bias'))

			conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand,
				filter_weights, [1,1,1,1], padding='VALID', name='conv_layer')
			relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name='relu_layer')

			maxpool_layer = tf.nn.max_pool(relu_layer,
				[1, title_length - window_size, 1, 1],
				[1,1,1,1], padding='VALID', name='maxpool_layer')

			pool_layer_lst.append(maxpool_layer)

	#dropout layer
	with tf.name_scope('pool_dropout'):
		pool_layer = tf.concat(pool_layer_lst, 3, name='pool_layer')
		max_num = len(window_sizes) * filter_num * 2 #为了让max_num = 64 修改
		pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name='pool_layer_flat')
		dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name='dropout_layer')

	return pool_layer_flat, dropout_layer


def get_movie_feature_layer(
	movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
	'''
	将movie id，movie genres， movie title的representations分别连入一个小型的神经网络
	然后将每个神经网络的输出拼接在一起，组成movie feature representation
	'''
	# print(movie_id_embed_layer.get_shape())   (? 1 64)
	# print(movie_categories_embed_layer.get_shape())同上
	# print(dropout_layer.get_shape())同上
	with tf.name_scope('movie_fc'):
		# 首先将movie的id和genres分别连入一个小型神经网络
		movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer,
			embed_dim, name='movie_id_fc_layer', activation=tf.nn.relu)
		movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer,
			embed_dim, name='movie_categories_fc_layer', activation=tf.nn.relu)

		#将id和genres的神经网络输出和经过cnn、dropout的titile feature拼接到一起，组成movie的representation
		movie_combine_layer = tf.concat([
			movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)#(?,1,96)
		movie_combine_layer = tf.contrib.layers.fully_connected(
			movie_combine_layer, 512, tf.tanh) #(?,1,200)
		movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 512])

		print(movie_combine_layer_flat.get_shape())
	return movie_combine_layer, movie_combine_layer_flat







if __name__ == '__main__':
	movie_id, movie_categories, movie_titles, dropout_keep_prob = get_inputs()
	movie_id_embed_layer = get_movie_id_embed_layer(movie_id)
	movie_categories_embed_layer = get_movie_categories_embed_layer(movie_categories)
	pool_layer_flat, dropout_layer = get_movie_cnn_layer(movie_titles, dropout_keep_prob)
	movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(
		movie_id_embed_layer, movie_categories_embed_layer, dropout_layer)

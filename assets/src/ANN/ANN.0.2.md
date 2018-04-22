### Deeplearning Algorithms tutorial
谷歌的人工智能位于全球前列，在图像识别、语音识别、无人驾驶等技术上都已经落地。而百度实质意义上扛起了国内的人工智能的大旗，覆盖无人驾驶、智能助手、图像识别等许多层面。苹果业已开始全面拥抱机器学习，新产品进军家庭智能音箱并打造工作站级别Mac。另外，腾讯的深度学习平台Mariana已支持了微信语音识别的语音输入法、语音开放平台、长按语音消息转文本等产品，在微信图像识别中开始应用。全球前十大科技公司全部发力人工智能理论研究和应用的实现，虽然入门艰难，但是一旦入门，高手也就在你的不远处！

机器学习主要有三种方式：监督学习，无监督学习与半监督学习。

（1）监督学习：从给定的训练数据集中学习出一个函数，当新的数据输入时，可以根据函数预测相应的结果。监督学习的训练集要求是包括输入和输出，也就是特征和目标。训练集中的目标是有标注的。如今机器学习已固有的监督学习算法有可以进行分类的，例如贝叶斯分类，SVM，ID3，C4.5以及分类决策树，以及现在最火热的人工神经网络，例如BP神经网络，RBF神经网络，Hopfield神经网络、深度信念网络和卷积神经网络等。人工神经网络是模拟人大脑的思考方式来进行分析，在人工神经网络中有显层，隐层以及输出层，而每一层都会有神经元，神经元的状态或开启或关闭，这取决于大数据。同样监督机器学习算法也可以作回归，最常用便是逻辑回归。

（2）无监督学习：与有监督学习相比，无监督学习的训练集的类标号是未知的，并且要学习的类的个数或集合可能事先不知道。常见的无监督学习算法包括聚类和关联，例如K均值法、Apriori算法。

（3）半监督学习：介于监督学习和无监督学习之间,例如EM算法。

如今的机器学习领域主要的研究工作在三个方面进行：1）面向任务的研究，研究和分析改进一组预定任务的执行性能的学习系统；2）认知模型，研究人类学习过程并进行计算模拟；3）理论的分析，从理论的层面探索可能的算法和独立的应用领域算法。

#### 自动编码器(Autoencoder)
自动编码器(Autoencoder)是一种无监督的学习算法，主要用于数据的降维或者特征的抽取，在深度学习中，自动编码器(Autoencoder)可用于在训练阶段开始前，确定权重矩阵的初始值。

神经网络中的权重矩阵可看作是对输入的数据进行特征转换，即先将数据编码为另一种形式，然后在此基础上进行一系列学习。然而，在对权重初始化时，我们并不知道初始的权重值在训练时会起到怎样的作用，也不知道在训练过程中权重会怎样的变化。因此一种较好的思路是，利用初始化生成的权重矩阵进行编码时，我们希望编码后的数据能够较好的保留原始数据的主要特征。那么，如何衡量码后的数据是否保留了较完整的信息呢？答案是：如果编码后的数据能够较为容易地通过解码恢复成原始数据，我们则认为较好的保留了数据信息。

自动编码器(Autoencoder)中：原始input（设为x）经过加权（W、b)、映射（Sigmoid）之后得到y，再对y反向加权映射回来成为z。

通过反复迭代训练两组（W、b），使得误差函数最小，即尽可能保证z近似于x，即完美重构了x。

那么可以说正向第一组权（W、b）是成功的，很好的学习了input中的关键特征，不然也不会重构得如此完美.
<p align="center">
<img width="300" align="center" src="../../images/252.jpg" />
</p>

这个过程很有趣，首先，它没有使用数据标签来计算误差update参数，所以是无监督学习。

其次，利用类似神经网络的双隐层的方式，简单粗暴地提取了样本的特征。 

这个双隐层是有争议的，最初的编码器确实使用了两组（W，b），但是Vincent在2010年的论文中做了研究，发现只要单组W就可以了。

即W'=W^T, W和W'称为Tied Weights。实验证明，W'真的只是在打酱油，完全没有必要去做训练。

逆向重构矩阵让人想起了逆矩阵，若W^-1=W^T的话，W就是个正交矩阵了，即W是可以训成近似正交阵的。

由于W'就是个酱油，训练完之后就没它事了。正向传播用W即可，相当于为input预先编个码，再导入到下一layer去。所以叫自动编码器，而不叫自动编码解码器。

自动编码器相当于创建了一个隐层，一个简单想法就是加在深度网络的开头，作为原始信号的初级filter，起到降维、提取特征的效果。
当然，这种做法就有一个问题，AutoEncoder可以看作是PCA的非线性补丁加强版，PCA的取得的效果是建立在降维基础上的。

仔细想想CNN这种结构，随着layer的推进，每层的神经元个数在递增，如果用了AutoEncoder去预训练，岂不是增维了？真的没问题？

相关论文中给出的实验结果认为AutoEncoder的增维效果还不赖，原因可能是非线性网络能力很强，尽管神经元个数增多，但是每个神经元的效果在衰减。

同时，随机梯度算法给了后续监督学习一个良好的开端。整体上，增维是利大于弊的。

#### 应用示例
```python
from __future__ import division

import tensorflow as tf
import numpy as np
import logging
import json
import os


class TextAutoencoder(object):
    """
    Class that encapsulates the encoder-decoder architecture to
    reconstruct pieces of text.
    """

    def __init__(self, lstm_units, embeddings, go, train=True,
                 train_embeddings=False, bidirectional=True):
        """
        Initialize the encoder/decoder and creates Tensor objects
        :param lstm_units: number of LSTM units
        :param embeddings: numpy array with initial embeddings
        :param go: index of the GO symbol in the embedding matrix
        :param train_embeddings: whether to adjust embeddings during training
        :param bidirectional: whether to create a bidirectional autoencoder
            (if False, a simple linear LSTM is used)
        """
        # EOS and GO share the same symbol. Only GO needs to be embedded, and
        # only EOS exists as a possible network output
        self.go = go
        self.eos = go

        self.bidirectional = bidirectional
        self.vocab_size = embeddings.shape[0]
        self.embedding_size = embeddings.shape[1]
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # the sentence is the object to be memorized
        self.sentence = tf.placeholder(tf.int32, [None, None], 'sentence')
        self.sentence_size = tf.placeholder(tf.int32, [None],
                                            'sentence_size')
        self.l2_constant = tf.placeholder(tf.float32, name='l2_constant')
        self.clip_value = tf.placeholder(tf.float32, name='clip')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.dropout_keep = tf.placeholder(tf.float32, name='dropout_keep')

        self.decoder_step_input = tf.placeholder(tf.int32,
                                                 [None],
                                                 'prediction_step')

        name = 'decoder_fw_step_state_c'
        self.decoder_fw_step_c = tf.placeholder(tf.float32,
                                                [None, lstm_units], name)
        name = 'decoder_fw_step_state_h'
        self.decoder_fw_step_h = tf.placeholder(tf.float32,
                                                [None, lstm_units], name)
        self.decoder_bw_step_c = tf.placeholder(tf.float32,
                                                [None, lstm_units],
                                                'decoder_bw_step_state_c')
        self.decoder_bw_step_h = tf.placeholder(tf.float32,
                                                [None, lstm_units],
                                                'decoder_bw_step_state_h')

        with tf.variable_scope('autoencoder') as self.scope:
            self.embeddings = tf.Variable(embeddings, name='embeddings',
                                          trainable=train_embeddings)

            initializer = tf.glorot_normal_initializer()
            self.lstm_fw = tf.nn.rnn_cell.LSTMCell(lstm_units,
                                                   initializer=initializer)
            self.lstm_bw = tf.nn.rnn_cell.LSTMCell(lstm_units,
                                                   initializer=initializer)

            embedded = tf.nn.embedding_lookup(self.embeddings, self.sentence)
            embedded = tf.nn.dropout(embedded, self.dropout_keep)

            # encoding step
            if bidirectional:
                bdr = tf.nn.bidirectional_dynamic_rnn
                ret = bdr(self.lstm_fw, self.lstm_bw,
                          embedded, dtype=tf.float32,
                          sequence_length=self.sentence_size,
                          scope=self.scope)
            else:
                ret = tf.nn.dynamic_rnn(self.lstm_fw, embedded,
                                        dtype=tf.float32,
                                        sequence_length=self.sentence_size,
                                        scope=self.scope)
            _, self.encoded_state = ret
            if bidirectional:
                encoded_state_fw, encoded_state_bw = self.encoded_state

                # set the scope name used inside the decoder.
                # maybe there's a more elegant way to do it?
                fw_scope_name = self.scope.name + '/fw'
                bw_scope_name = self.scope.name + '/bw'
            else:
                encoded_state_fw = self.encoded_state
                fw_scope_name = self.scope

            self.scope.reuse_variables()

            # generate a batch of embedded GO
            # sentence_size has the batch dimension
            go_batch = self._generate_batch_go(self.sentence_size)
            embedded_eos = tf.nn.embedding_lookup(self.embeddings,
                                                  go_batch)
            embedded_eos = tf.reshape(embedded_eos,
                                      [-1, 1, self.embedding_size])
            decoder_input = tf.concat([embedded_eos, embedded], axis=1)

            # decoding step

            # We give the same inputs to the forward and backward LSTMs,
            # but each one has its own hidden state
            # their outputs are concatenated and fed to the softmax layer
            if bidirectional:
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    self.lstm_fw, self.lstm_bw, decoder_input,
                    self.sentence_size, encoded_state_fw, encoded_state_bw)

                # concat fw and bw outputs
                outputs = tf.concat(outputs, -1)
            else:
                outputs, _ = tf.nn.dynamic_rnn(
                    self.lstm_fw, decoder_input, self.sentence_size,
                    encoded_state_fw)

            self.decoder_outputs = outputs

        # now project the outputs to the vocabulary
        with tf.variable_scope('projection') as self.projection_scope:
            # decoder_outputs has shape (batch, max_sentence_size, vocab_size)
            self.logits = tf.layers.dense(outputs, self.vocab_size)

        # tensors for running a model
        embedded_step = tf.nn.embedding_lookup(self.embeddings,
                                               self.decoder_step_input)
        state_fw = tf.nn.rnn_cell.LSTMStateTuple(self.decoder_fw_step_c,
                                                 self.decoder_fw_step_h)
        state_bw = tf.nn.rnn_cell.LSTMStateTuple(self.decoder_bw_step_c,
                                                 self.decoder_bw_step_h)
        with tf.variable_scope(fw_scope_name, reuse=True):
            ret_fw = self.lstm_fw(embedded_step, state_fw)
        step_output_fw, self.decoder_fw_step_state = ret_fw

        if bidirectional:
            with tf.variable_scope(bw_scope_name, reuse=True):
                ret_bw = self.lstm_bw(embedded_step, state_bw)
                step_output_bw, self.decoder_bw_step_state = ret_bw
                step_output = tf.concat(axis=1, values=[step_output_fw,
                                                        step_output_bw])
        else:
            step_output = step_output_fw

        with tf.variable_scope(self.projection_scope, reuse=True):
            self.projected_step_output = tf.layers.dense(step_output,
                                                         self.vocab_size)

        if train:
            self._create_training_tensors()

    def _create_training_tensors(self):
        """
        Create member variables related to training.
        """
        eos_batch = self._generate_batch_go(self.sentence_size)
        eos_batch = tf.reshape(eos_batch, [-1, 1])
        decoder_labels = tf.concat([self.sentence, eos_batch], -1)

        projection_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope=self.projection_scope.name)
        # a bit ugly, maybe we should improve this?
        projection_w = [var for var in projection_vars
                        if 'kernel' in var.name][0]
        projection_b = [var for var in projection_vars
                        if 'bias' in var.name][0]

        # set the importance of each time step
        # 1 if before sentence end or EOS itself; 0 otherwise
        max_len = tf.shape(self.sentence)[1]
        mask = tf.sequence_mask(self.sentence_size + 1, max_len + 1, tf.float32)
        num_actual_labels = tf.reduce_sum(mask)
        projection_w_t = tf.transpose(projection_w)

        # reshape to have batch and time steps in the same dimension
        decoder_outputs2d = tf.reshape(self.decoder_outputs,
                                       [-1, tf.shape(self.decoder_outputs)[-1]])
        labels = tf.reshape(decoder_labels, [-1, 1])
        sampled_loss = tf.nn.sampled_softmax_loss(
            projection_w_t, projection_b, labels, decoder_outputs2d, 100,
            self.vocab_size)

        masked_loss = tf.reshape(mask, [-1]) * sampled_loss
        self.loss = tf.reduce_sum(masked_loss) / num_actual_labels

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)

        self.train_op = optimizer.apply_gradients(zip(gradients, v),
                                                  global_step=self.global_step)

    def get_trainable_variables(self):
        """
        Return all trainable variables inside the model
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def train(self, session, save_path, train_data, valid_data,
              batch_size, epochs, learning_rate, dropout_keep,
              clip_value, report_interval):
        """
        Train the model
        :param session: tensorflow session
        :param train_data: Dataset object with training data
        :param valid_data: Dataset object with validation data
        :param batch_size: batch size
        :param learning_rate: initial learning rate
        :param dropout_keep: the probability that each LSTM input/output is kept
        :param epochs: how many epochs to train for
        :param clip_value: value to clip tensor norm during training
        :param save_path: folder to save the model
        :param report_interval: report after that many batches
        """
        saver = tf.train.Saver(self.get_trainable_variables(),
                               max_to_keep=1)

        best_loss = 10000
        accumulated_loss = 0
        batch_counter = 0
        num_sents = 0

        # get all data at once. we need all matrices with the same size,
        # or else they don't fit the placeholders
        # train_sents, train_sizes = train_data.join_all(self.go,
        #                                                self.num_time_steps,
        #                                                shuffle=True)

        # del train_data  # save memory...
        valid_sents, valid_sizes = valid_data.join_all(self.go,
                                                       shuffle=True)
        train_data.reset_epoch_counter()
        feeds = {self.clip_value: clip_value,
                 self.dropout_keep: dropout_keep,
                 self.learning_rate: learning_rate}

        while train_data.epoch_counter < epochs:
            batch_counter += 1
            train_sents, train_sizes = train_data.next_batch(batch_size)
            feeds[self.sentence] = train_sents
            feeds[self.sentence_size] = train_sizes

            _, loss = session.run([self.train_op, self.loss], feeds)

            # multiply by len because some batches may be smaller
            # (due to bucketing), then take the average
            accumulated_loss += loss * len(train_sents)
            num_sents += len(train_sents)

            if batch_counter % report_interval == 0:
                avg_loss = accumulated_loss / num_sents
                accumulated_loss = 0
                num_sents = 0

                # we can't use all the validation at once, since it would
                # take too much memory. running many small batches would
                # instead take too much time. So let's just sample it.
                sample_indices = np.random.randint(0, len(valid_data),
                                                   5000)
                validation_feeds = {
                    self.sentence: valid_sents[sample_indices],
                    self.sentence_size: valid_sizes[sample_indices],
                    self.dropout_keep: 1}

                loss = session.run(self.loss, validation_feeds)
                msg = '%d epochs, %d batches\t' % (train_data.epoch_counter,
                                                   batch_counter)
                msg += 'Avg batch loss: %f\t' % avg_loss
                msg += 'Validation loss: %f' % loss
                if loss < best_loss:
                    best_loss = loss
                    self.save(saver, session, save_path)
                    msg += '\t(saved model)'

                logging.info(msg)

    def save(self, saver, session, directory):
        """
        Save the autoencoder model and metadata to the specified
        directory.
        """
        model_path = os.path.join(directory, 'model')
        saver.save(session, model_path)
        metadata = {'vocab_size': self.vocab_size,
                    'embedding_size': self.embedding_size,
                    'num_units': self.lstm_fw.output_size,
                    'go': self.go,
                    'bidirectional': self.bidirectional
                    }
        metadata_path = os.path.join(directory, 'metadata.json')
        with open(metadata_path, 'wb') as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, directory, session, train=False):
        """
        Load an instance of this class from a previously saved one.
        :param directory: directory with the model files
        :param session: tensorflow session
        :param train: if True, also create training tensors
        :return: a TextAutoencoder instance
        """
        model_path = os.path.join(directory, 'model')
        metadata_path = os.path.join(directory, 'metadata.json')
        with open(metadata_path, 'rb') as f:
            metadata = json.load(f)
        dummy_embeddings = np.empty((metadata['vocab_size'],
                                     metadata['embedding_size'],),
                                    dtype=np.float32)

        ae = TextAutoencoder(metadata['num_units'], dummy_embeddings,
                             metadata['go'], train=train,
                             bidirectional=metadata['bidirectional'])
        vars_to_load = ae.get_trainable_variables()
        if not train:
            # if not flagged for training, the embeddings won't be in
            # the list
            vars_to_load.append(ae.embeddings)

        saver = tf.train.Saver(vars_to_load)
        saver.restore(session, model_path)
        return ae

    def encode(self, session, inputs, sizes):
        """
        Run the encoder to obtain the encoded hidden state
        :param session: tensorflow session
        :param inputs: 2-d array with the word indices
        :param sizes: 1-d array with size of each sentence
        :return: a 2-d numpy array with the hidden state
        """
        feeds = {self.sentence: inputs,
                 self.sentence_size: sizes,
                 self.dropout_keep: 1}
        state = session.run(self.encoded_state, feeds)
        if self.bidirectional:
            state_fw, state_bw = state
            return np.hstack((state_fw.c, state_bw.c))
        return state.c

    def run(self, session, inputs, sizes):
        """
        Run the autoencoder with the given data
        :param session: tensorflow session
        :param inputs: 2-d array with the word indices
        :param sizes: 1-d array with size of each sentence
        :return: a 2-d array (batch, output_length) with the answer
            produced by the autoencoder. The output length is not
            fixed; it stops after producing EOS for all items in the
            batch or reaching two times the maximum number of time
            steps in the inputs.
        """
        feeds = {self.sentence: inputs,
                 self.sentence_size: sizes,
                 self.dropout_keep: 1}
        state = session.run(self.encoded_state, feeds)
        if self.bidirectional:
            state_fw, state_bw = state
        else:
            state_fw = state

        time_steps = 0
        max_time_steps = 2 * len(inputs[0])
        answer = []
        input_symbol = self.go * np.ones_like(sizes, dtype=np.int32)

        # this array control which sequences have already been finished by the
        # decoder, i.e., for which ones it already produced the END symbol
        sequences_done = np.zeros_like(sizes, dtype=np.bool)

        while True:
            # we could use tensorflow's rnn_decoder, but this gives us
            # finer control

            feeds = {self.decoder_fw_step_c: state_fw.c,
                     self.decoder_fw_step_h: state_fw.h,
                     self.decoder_step_input: input_symbol,
                     self.dropout_keep: 1}
            if self.bidirectional:
                feeds[self.decoder_bw_step_c] = state_bw.c
                feeds[self.decoder_bw_step_h] = state_bw.h

                ops = [self.projected_step_output,
                       self.decoder_fw_step_state,
                       self.decoder_bw_step_state]
                outputs, state_fw, state_bw = session.run(ops, feeds)
            else:
                ops = [self.projected_step_output,
                       self.decoder_fw_step_state]
                outputs, state_fw = session.run(ops, feeds)

            input_symbol = outputs.argmax(1)
            answer.append(input_symbol)

            # use an "additive" or in order to avoid infinite loops
            sequences_done |= (input_symbol == self.eos)

            if sequences_done.all() or time_steps > max_time_steps:
                break
            else:
                time_steps += 1

        return np.hstack(answer)

    def _generate_batch_go(self, like):
        """
        Generate a 1-d tensor with copies of EOS as big as the batch size,
        :param like: a tensor whose shape the returned embeddings should match
        :return: a tensor with shape as `like`
        """
        ones = tf.ones_like(like)
        return ones * self.go
```

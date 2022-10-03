import tensorflow as tf
import os, time
import numpy as np
from Network import ResNet
from ImageUtils import parse_record
from tqdm import tqdm
from scipy.special import softmax

"""This script defines the training, validation and testing process.
"""


class ResNetModel(object):

    def __init__(self, sess, configs):
        self.sess = sess
        self.configs = configs
        self.network = ResNet(configs)

    def setup(self, training):
        print('---Setup input interfaces...')
        self.inputs = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        self.labels = tf.placeholder(tf.int32)
        # Note: this placeholder allows us to set the learning rate for each epoch
        self.learning_rate = tf.placeholder(tf.float32)

        print('---Setup the network...')
        network = ResNet(self.configs)

        if training:
            print('---Setup training components...')
            # compute logits
            self.logits = network(self.inputs, True)

            # predictions for validation
            self.preds = tf.argmax(self.logits, axis=-1)

            # weight decay
            l2_loss = self.configs['weight_decay'] * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables()
                 if 'kernel' in v.name])

            # cross entropy
            params = tf.constant(np.eye(27))
            targets = tf.gather(params, self.labels)
            ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=self.logits))

            # final loss function
            self.losses = ce_loss + l2_loss
            ### END CODE HERE

            # momentum optimizer with momentum=0.9
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=self.learning_rate, momentum=0.9)

            # train_op
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.losses)

            print('---Setup the Saver for saving models...')
            self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=0)

        else:
            print('---Setup testing components...')
            # compute predictions
            self.logits = network(self.inputs, False)
            self.preds = tf.argmax(self.logits, axis=-1)

            print('---Setup the Saver for loading models...')
            self.loader = tf.train.Saver(var_list=tf.global_variables())

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        print('###Train###')

        self.setup(True)
        self.sess.run(tf.global_variables_initializer())

        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = int(num_samples / configs['batch_size'])

        learning_rate = 0.1
        print('---Run...')
        for epoch in range(1, configs['max_epoch'] + 1):

            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            # # Set the learning rate for this epoch
            # if epoch < 10:
            #     learning_rate = 0.1
            # elif epoch < 20:
            #     learning_rate = 0.05
            # else:
            #     learning_rate = 0.01

            loss_value = []
            for i in range(num_batches):
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                x_batch = curr_x_train[i * configs['batch_size']: (i + 1) * configs['batch_size']]
                y_batch = curr_y_train[i * configs['batch_size']: (i + 1) * configs['batch_size']]
                for idx in range(configs['batch_size']):
                    x_batch[idx] = parse_record(x_batch[idx], training=1).reshape(-1)
                x_batch = x_batch.reshape(configs['batch_size'], 28, 28, 1)

                # Run
                feed_dict = {self.inputs: x_batch,
                             self.labels: y_batch,
                             self.learning_rate: learning_rate}
                loss, _ = self.sess.run(
                    [self.losses, self.train_op], feed_dict=feed_dict)

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss),
                      end='\r', flush=True)

            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(
                epoch, loss, duration))

            if epoch % configs['save_interval'] == 0:
                self.save(self.saver, epoch)

    def evaluate(self, x, y, checkpoint_num_list):
        print('###Test or Validation###')

        self.setup(False)
        self.sess.run(tf.global_variables_initializer())

        # load checkpoint
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = self.configs['modeldir'] + '/model.ckpt-' + str(checkpoint_num)
            self.load(self.loader, checkpointfile)

            preds = []
            for i in tqdm(range(x.shape[0])):
                inputs = parse_record(x[i], training=0).reshape(-1)
                inputs = inputs.reshape(1, 28, 28, 1)
                preds.append(self.sess.run(self.preds, feed_dict={self.inputs: inputs}))

            preds = np.array(preds).reshape(y.shape)
            print('Test accuracy: {:.4f}'.format(np.sum(preds == y) / y.shape[0]))

    def predict_prob(self, x, checkpoint_num_list):
        self.setup(False)
        self.sess.run(tf.global_variables_initializer())

        # load checkpoint
        for checkpoint_num in checkpoint_num_list:
            checkpointfile = self.configs['modeldir'] + '/model.ckpt-' + str(checkpoint_num)
            self.load(self.loader, checkpointfile)

            preds = []
            for i in tqdm(range(x.shape[0])):
                inputs = parse_record(x[i], training=0).reshape(-1)
                inputs = inputs.reshape(1, 28, 28, 1)
                preds.append(self.sess.run(self.logits, feed_dict={self.inputs: inputs}))

            preds = np.array(preds).reshape((-1, 27))
            preds = softmax(preds, axis=1)
            # preds = np.array(preds).reshape((-1, 1))
        return preds

    def save(self, saver, step):
        '''Save weights.'''
        model_name = 'model.ckpt'
        checkpoint_path = os.path.join(self.configs['modeldir'], model_name)
        if not os.path.exists(self.configs['modeldir']):
            os.makedirs(self.configs['modeldir'])
        saver.save(self.sess, checkpoint_path, global_step=step)
        print('The checkpoint has been created.')

    def load(self, loader, filename):
        '''Load trained weights.'''
        loader.restore(self.sess, filename)
        print("Restored model parameters from {}".format(filename))

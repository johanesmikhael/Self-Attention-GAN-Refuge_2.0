import time
import tensorflow as tf
from ops import *
from utils import *
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
from glob import glob

class SAGAN(object):

    def __init__(self, sess, **kwargs):
        self.sess = sess
        self.model_name = "SAGAN"  # name for checkpoint
        self.dataset_name = kwargs.get('dataset')
        self.checkpoint_dir = kwargs.get('checkpoint_dir')
        self.sample_dir = kwargs.get('sample_dir')
        self.result_dir = kwargs.get('result_dir')
        self.log_dir = kwargs.get('log_dir')

        self.epoch = kwargs.get('epoch', 10)
        self.iteration = kwargs.get('iteration', 10000)
        self.batch_size = kwargs.get('batch_size', 16)
        self.print_freq = kwargs.get('print_freq', 500)
        self.save_freq = kwargs.get('save_freq', 500)
        self.img_size = kwargs.get('img_size', 128)

        """ Generator """
        self.layer_num = int(np.log2(self.img_size)) - 3
        self.z_dim = kwargs.get('z_dim', 128)  # dimension of noise-vector
        self.gan_type = kwargs.get('gan_type', 'hinge')

        """ Discriminator """
        self.n_critic = kwargs.get('n_critic', 1)
        self.sn = kwargs.get('sn', True)
        self.ld = kwargs.get('ld', 10.0)


        self.sample_num = kwargs.get('sample_num', 64)  # number of generated images to be saved
        self.test_num = kwargs.get('test_num', 1)

        """ Augmentation """
        self.crop_pos = kwargs.get('crop_pos', 'center')
        self.zoom_range = kwargs.get('zoom_range', 0.0)

        # train
        self.g_learning_rate = kwargs.get('g_lr', 0.0001)
        self.d_learning_rate = kwargs.get('d_lr', 0.0004)
        self.beta1 = kwargs.get('beta1', 0.0)
        self.beta2 = kwargs.get('beta2', 0.9)

        self.c_dim = 3
        self.data = load_data(dataset_name=self.dataset_name, size=self.img_size)
        self.custom_dataset = True


        self.dataset_num = len(self.data)

        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        print()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# generator layer : ", self.layer_num)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.layer_num)
        print("# the number of critic : ", self.n_critic)
        print("# spectral normalization : ", self.sn)

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, z, is_training=True, reuse=False):
        with tf.compat.v1.variable_scope("generator", reuse=reuse):
            ch = 1024
            x = fully_connected(z, units=4 * 4 * ch, sn=self.sn, scope='fc')
            x = tf.reshape(x, [-1, 4, 4, ch])

            x = up_resblock(x, channels=ch, is_training=is_training, sn=self.sn, scope='front_resblock_0')

            for i in range(self.layer_num // 2) :
                x = up_resblock(x, channels=ch // 2, is_training=is_training, sn=self.sn, scope='middle_resblock_' + str(i))
                ch = ch // 2

            x = self.google_attention(x, channels=ch, scope='self_attention')

            for i in range(self.layer_num // 2, self.layer_num) :
                x = up_resblock(x, channels=ch // 2, is_training=is_training, sn=self.sn, scope='back_resblock_' + str(i))
                ch = ch // 2

            x = batch_norm(x, is_training)
            x = relu(x)

            x = conv(x, channels=self.c_dim, kernel=3, stride=1, pad=1, pad_type='reflect', scope='g_logit')
            x = tanh(x)

            return x

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x, reuse=False):
        with tf.compat.v1.variable_scope("discriminator", reuse=reuse):
            ch = 64

            x = init_down_resblock(x, channels=ch, sn=self.sn, scope='init_resblock')

            x = down_resblock(x, channels=ch * 2, sn=self.sn, scope='front_down_resblock')
            x = self.google_attention(x, channels=ch * 2, scope='self_attention')

            ch = ch * 2

            for i in range(self.layer_num) :
                if i == self.layer_num - 1 :
                    x = down_resblock(x, channels=ch, sn=self.sn, to_down=False, scope='middle_down_resblock_' + str(i))
                else :
                    x = down_resblock(x, channels=ch * 2, sn=self.sn, scope='middle_down_resblock_' + str(i))

                ch = ch * 2

            x = lrelu(x, 0.2)

            x = global_sum_pooling(x)

            x = fully_connected(x, units=1, sn=self.sn, scope='d_logit')

            return x

    def attention(self, x, channels, scope='attention'):
        with tf.compat.v1.variable_scope(scope):
            f = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv') # [bs, h, w, c']
            g = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv') # [bs, h, w, c']
            h = conv(x, channels, kernel=1, stride=1, sn=self.sn, scope='h_conv') # [bs, h, w, c]

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
            gamma = tf.compat.v1.get_variable("gamma", [1], initializer=tf.compat.v1.constant_initializer(0.0))

            o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
            o = conv(o, channels, kernel=1, stride=1, sn=self.sn, scope='attn_conv')

            x = gamma * o + x

        return x

    def google_attention(self, x, channels, scope='attention'):
        with tf.compat.v1.variable_scope(scope):
            batch_size, height, width, num_channels = x.get_shape().as_list()
            f = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv')  # [bs, h, w, c']
            f = max_pooling(f)

            g = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv')  # [bs, h, w, c']

            h = conv(x, channels // 2, kernel=1, stride=1, sn=self.sn, scope='h_conv')  # [bs, h, w, c]
            h = max_pooling(h)

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
            gamma = tf.compat.v1.get_variable("gamma", [1], initializer=tf.compat.v1.constant_initializer(0.0))

            o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
            o = conv(o, channels, kernel=1, stride=1, sn=self.sn, scope='attn_conv')
            x = gamma * o + x

        return x

    def gradient_penalty(self, real, fake):
        if self.gan_type == 'dragan' :
            shape = tf.shape(input=real)
            eps = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(x=real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper

            # Author suggested U[0,1] in original paper, but he admitted it is bug in github
            # (https://github.com/kodalinaveen3/DRAGAN). It should be two-sided.

            alpha = tf.random.uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X

        else :
            alpha = tf.random.uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
            interpolated = alpha*real + (1. - alpha)*fake

        logit = self.discriminator(interpolated, reuse=True)

        grad = tf.gradients(ys=logit, xs=interpolated)[0]  # gradient of D(interpolated)
        grad_norm = tf.norm(tensor=flatten(grad), axis=1)  # l2 norm

        GP = 0

        # WGAN - LP
        if self.gan_type == 'wgan-lp':
            GP = self.ld * tf.reduce_mean(input_tensor=tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
            GP = self.ld * tf.reduce_mean(input_tensor=tf.square(grad_norm - 1.))

        return GP

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ Graph Input """
        # images
        if self.custom_dataset :
            Image_Data_Class = ImageData(self.img_size, self.c_dim, crop_pos=self.crop_pos, zoom_range=self.zoom_range)
            inputs = tf.data.Dataset.from_tensor_slices(self.data)
            gpu_device = '/gpu:0'
            inputs = inputs.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))

            inputs_iterator = tf.compat.v1.data.make_one_shot_iterator(inputs)

            self.inputs = inputs_iterator.get_next()

        else :
            self.inputs = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='real_images')

        # noises
        self.z = tf.compat.v1.placeholder(tf.float32, [self.batch_size, 1, 1, self.z_dim], name='z')

        """ Loss Function """
        # output of D for real images
        real_logits = self.discriminator(self.inputs)

        # output of D for fake images
        fake_images = self.generator(self.z)
        fake_logits = self.discriminator(fake_images, reuse=True)

        if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan' :
            GP = self.gradient_penalty(real=self.inputs, fake=fake_images)
        else :
            GP = 0

        # get loss for discriminator
        self.d_loss = discriminator_loss(self.gan_type, real=real_logits, fake=fake_logits) + GP

        # get loss for generator
        self.g_loss = generator_loss(self.gan_type, fake=fake_logits)

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.compat.v1.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        # optimizers
        self.d_optim = tf.compat.v1.train.AdamOptimizer(self.d_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf.compat.v1.train.AdamOptimizer(self.g_learning_rate, beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)

        """ Summary """
        self.d_sum = tf.compat.v1.summary.scalar("d_loss", self.d_loss)
        self.g_sum = tf.compat.v1.summary.scalar("g_loss", self.g_loss)

    ##################################################################################
    # Train
    ##################################################################################

    def train(self, z_noise=None):

        # initialize all variables
        tf.compat.v1.global_variables_initializer().run(session=self.sess)

        # graph inputs for visualize training results
        sample_num, self.sample_z = self.get_z_samples(z_noise)

        # saver to save model
        self.saver = tf.compat.v1.train.Saver(max_to_keep=10)

        # summary writer
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load_models(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        past_g_loss = -1.
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, self.iteration):
                batch_z = np.random.normal(0, 1, [self.batch_size, 1, 1, self.z_dim])

                if self.custom_dataset :

                    train_feed_dict = {
                        self.z: batch_z
                    }

                else :
                    random_index = np.random.choice(self.dataset_num, size=self.batch_size, replace=False)
                    # batch_images = self.data[idx*self.batch_size : (idx+1)*self.batch_size]
                    batch_images = self.data[random_index]

                    train_feed_dict = {
                        self.inputs : batch_images,
                        self.z : batch_z
                    }

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # update G network
                g_loss = None
                if (counter - 1) % self.n_critic == 0:
                    _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict=train_feed_dict)
                    self.writer.add_summary(summary_str, counter)
                    past_g_loss = g_loss

                # display training status
                counter += 1
                if g_loss == None :
                    g_loss = past_g_loss
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                # save training results for every n steps
                if np.mod(idx+1, self.print_freq) == 0:
                    sample_list = []
                    for sample_z in self.sample_z:
                        sample = self.sess.run(self.fake_images, feed_dict={self.z: sample_z})
                        sample_list.append(sample)
                    if len(sample_list) == 1:
                        samples = sample
                    else:
                        samples = np.concatenate(sample_list)
                    tot_num_samples = sample_num
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images_plt(samples[:manifold_h * manifold_w, :, :, :],
                                [manifold_h, manifold_w],
                                self.sample_dir + '/' + self.model_name + '_train_{:02d}_{:05d}.png'.format(epoch, idx+1), mode='sample')

                if np.mod(idx+1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            # self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        dataset_name = self.dataset_name.split('/')[-1]
        return "{}_{}_{}_{}_{}_{}".format(
            self.model_name, dataset_name, self.gan_type, self.img_size, self.z_dim, self.sn)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load_models(self, checkpoint_dir):
        import re
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        models = glob(f'{checkpoint_dir}/*.meta')
        model_names = []
        if len(models) > 0:
            for i, model in enumerate(models):
                model_name = '.'.join(os.path.basename(model).split('.')[:-1])
                model_names.append(model_name)
                print(f'[{i}] {model_name}')
            selected_index = -1
            while not selected_index in range(i+1):
                try:
                    selected_index = int(input('select model to load: '))
                except:
                    print('select the model with model index')
            print( model_names[selected_index])
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, model_names[selected_index]))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",model_names[selected_index])).group(0))

            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print(ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, 1, 1, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    self.sample_dir + '/' + self.model_name + '_epoch%02d' % epoch + '_visualize.png')

    def get_z_samples(self, z_noise):
        z_samples = []
        if type(z_noise) is np.ndarray:
            if z_noise.shape[3] != self.z_dim:          # mismatch of z_dim
                print(f'dimension of z_noise mismatch : {self.z_dim}')
                return None, None
            else:
                sample_num = z_noise.shape[0]
                if sample_num == self.batch_size:
                    z_samples.append(z_noise)
                elif sample_num < self.batch_size:
                    z_padding = np.random.uniform(-1, 1, size=(self.batch_size-sample_num, 1, 1, self.z_dim))
                    z_extended = np.concatenate([z_noise, z_padding])
                    z_samples.append(z_extended)
                elif sample_num > self.batch_size:
                    full_batch_num = sample_num // self.batch_size
                    remaining = sample_num % self.batch_size
                    for batch_num in range(full_batch_num):
                        z_noise_per_batch = z_noise[self.batch_size*batch_num:self.batch_size*(batch_num+1)]
                        z_samples.append(z_noise_per_batch)
                    if remaining > 0:
                        z_padding = np.random.uniform(-1, 1, size=(self.batch_size-remaining, 1, 1, self.z_dim))
                        z_remaining = z_noise[self.batch_size*(batch_num+1):]
                        z_extended = np.concatenate([z_remaining, z_padding])
                        z_samples.append(z_extended)
        else:
            sample_num = self.sample_num
            sample_num_rounded = int(np.ceil(sample_num/self.batch_size))
            for i in range(sample_num_rounded):
                z_samples.append(np.random.normal(0, 1, size=(self.batch_size, 1, 1, self.z_dim)))
        return sample_num, z_samples

    def load_pretrained_model(self):
        tf.compat.v1.global_variables_initializer().run(session=self.sess)
        self.saver = tf.compat.v1.train.Saver()
        could_load, checkpoint_counter = self.load_models(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

    def generate_single_images(self, name='result', z_noise=None):
        if type(z_noise) is not np.ndarray:
            print('no z_noise provided')
            return
        result_dir = os.path.join(self.result_dir, self.model_dir)
        result_dir += f'/{name}'
        check_folder(result_dir)

        samples= self.predict(z_noise)
        print(samples.shape)

        for i in range(samples.shape[0]):
            img = samples[i]
            save_path = f'{result_dir}/{i:06d}.png'
            save_image(img, save_path)


    def predict(self, z_noise):
        sample_num, z_samples = self.get_z_samples(z_noise)

        sample_list = []
        for z_sample in z_samples:
            sample = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})
            sample_list.append(sample)
        if len(sample_list) == 1:
            samples = sample
        else:
            samples = np.concatenate(sample_list)

        samples = samples[:sample_num]

        return samples


    def test(self, name='test', nrows=None, ncols=None, z_noise=None):
        # tf.compat.v1.global_variables_initializer().run(session=self.sess)
        # self.saver = tf.compat.v1.train.Saver()
        # could_load, checkpoint_counter = self.load_models(self.checkpoint_dir)
        result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(result_dir)

        if type(z_noise) is not np.ndarray:
            sample_num = self.sample_num
        else:
            sample_num = z_noise[0]

        result = self.predict(z_noise)


        if nrows == None or ncols == None:
            manifold_h = int(np.floor(np.sqrt(sample_num)))
            manifold_w = int(np.floor(np.sqrt(sample_num)))
        else:
            manifold_h = nrows
            manifold_w = ncols

        save_images_plt(result,
                    [manifold_w, manifold_h],
                    result_dir + '/' + self.model_name + '_{}.png'.format(name))

import os
import tensorflow as tf
import source.layers as lay

class CNN(object):

    def __init__(self, height, width, channel, num_class, ksize, learning_rate=1e-3, ckpt_dir='./Checkpoint'):

        print("\nInitializing Short-ResNet...")
        self.height, self.width, self.channel, self.num_class = height, width, channel, num_class
        self.ksize, self.learning_rate = ksize, learning_rate
        self.ckpt_dir = ckpt_dir

        self.customlayers = lay.Layers()
        self.model(tf.zeros([1, self.height, self.width, self.channel]), verbose=True)

        self.optimizer = tf.optimizers.Adam(self.learning_rate)

        self.summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

    def step(self, x, y, iteration=0, train=False):

        with tf.GradientTape() as tape:
            logits = self.model(x, verbose=False)
            smce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.math.reduce_mean(smce)

        score = self.customlayers.softmax(logits)
        pred = tf.argmax(score, 1)
        correct_pred = tf.equal(pred, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        if(train):
            gradients = tape.gradient(loss, self.customlayers.params_trainable)
            self.optimizer.apply_gradients(zip(gradients, self.customlayers.params_trainable))

            with self.summary_writer.as_default():
                tf.summary.scalar('Network-in-Network/loss', loss, step=iteration)
                tf.summary.scalar('Network-in-Network/accuracy', accuracy, step=iteration)

        return loss, accuracy, score

    def save_params(self):

        vars_to_save = {}
        for idx, name in enumerate(self.customlayers.name_bank):
            vars_to_save[self.customlayers.name_bank[idx]] = self.customlayers.params_trainable[idx]
        vars_to_save["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_save)
        ckptman = tf.train.CheckpointManager(ckpt, directory=self.ckpt_dir, max_to_keep=3)
        ckptman.save()

    def load_params(self):

        vars_to_load = {}
        for idx, name in enumerate(self.customlayers.name_bank):
            vars_to_load[self.customlayers.name_bank[idx]] = self.customlayers.params_trainable[idx]
        vars_to_load["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        status = ckpt.restore(latest_ckpt)
        status.expect_partial()

    def model(self, x, verbose=False):

        if(verbose): print("input", x.shape)

        conv1 = self.customlayers.conv2d(x, \
            self.customlayers.get_weight(vshape=[3, 3, self.channel, 16], name="%s" %("conv1")), \
            stride_size=1, padding='SAME')
        conv1_bn = self.customlayers.batch_norm(conv1, name="%s_bn" %("conv1"))
        conv1_act = self.customlayers.elu(conv1_bn)
        conv1_pool = self.customlayers.maxpool(conv1_act, pool_size=2, stride_size=2)

        conv2_1 = self.residual(conv1_pool, \
            ksize=self.ksize, inchannel=16, outchannel=32, expansion=True, name="conv2_1", verbose=verbose)
        conv2_2 = self.residual(conv2_1, \
            ksize=self.ksize, inchannel=32, outchannel=32, expansion=True, name="conv2_2", verbose=verbose)
        conv2_pool = self.customlayers.maxpool(conv2_2, pool_size=2, stride_size=2)

        conv3_1 = self.residual(conv2_pool, \
            ksize=self.ksize, inchannel=32, outchannel=64, expansion=False, name="conv3_1", verbose=verbose)
        conv3_2 = self.residual(conv3_1, \
            ksize=self.ksize, inchannel=64, outchannel=self.num_class, expansion=False, name="conv3_2", verbose=verbose)

        output = tf.reduce_mean(conv3_2, axis=(1, 2))

        if(verbose):
            print("GAP", output.shape)
            print("\nNum Parameter")
            print("Total             : %d" %(self.customlayers.num_params))

        return output

    def ninconv(self, input, ksize, channels, name=""):

        convtmp_1 = self.customlayers.conv2d(input, \
            self.customlayers.get_weight(vshape=[ksize, ksize, channels[0], channels[1]], name="nin%s_1" %(name)), \
            stride_size=1, padding='SAME')
        convtmp_1bn = self.customlayers.batch_norm(convtmp_1, name="%s_bn" %("conv1"))
        convtmp_1act = self.customlayers.elu(convtmp_1bn)

        convtmp_2 = self.customlayers.conv2d(convtmp_1act, \
            self.customlayers.get_weight(vshape=[1, 1, channels[1], channels[2]], name="nin%s_2" %(name)), \
            stride_size=1, padding='SAME')
        convtmp_2bn = self.customlayers.batch_norm(convtmp_2, name="%s_bn" %("conv1"))
        convtmp_2act = self.customlayers.elu(convtmp_2bn)

        convtmp_3 = self.customlayers.conv2d(convtmp_2act, \
            self.customlayers.get_weight(vshape=[1, 1, channels[2], channels[3]], name="nin%s_3" %(name)), \
            stride_size=1, padding='SAME')
        convtmp_3bn = self.customlayers.batch_norm(convtmp_3, name="%s_bn" %("conv1"))
        output = self.customlayers.elu(convtmp_3bn)

        return output

    def residual(self, input, ksize, inchannel, outchannel, expansion=False, name="", verbose=False):

        channels = [inchannel, outchannel*2, outchannel*2, outchannel]
        convtmp_1 = self.ninconv(input=input, ksize=ksize, channels=channels, name="%s_1" %(name))
        channels = [outchannel, outchannel*2, outchannel*2, outchannel]
        convtmp_2 = self.ninconv(input=convtmp_1, ksize=ksize, channels=channels, name="%s_2" %(name))

        if(input.shape[-1] != convtmp_2.shape[-1]):
            convtmp_sc = self.customlayers.conv2d(input, \
                self.customlayers.get_weight(vshape=[1, 1, inchannel, outchannel], name="%s_sc" %(name)), \
                stride_size=1, padding='SAME')
            convtmp_scbn = self.customlayers.batch_norm(convtmp_sc, name="%s_scbn" %(name))
            convtmp_scact = self.customlayers.elu(convtmp_scbn)
            input = convtmp_scact

        output = input + convtmp_2

        if(verbose): print(name, output.shape)
        return output

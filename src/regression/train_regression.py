
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
from features import extract_features_omniglot, extract_features_mini_imagenet, sin_function
from inference import infer_classifier, inference_block
from utilities import *  # sample_normal, multinoulli_log_density, print_and_log, get_log_files
import os
from data_generator import DataGenerator
import matplotlib.pyplot as plt
import os.path as osp
"""
parse_command_line: command line parser
"""




def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", choices=["Omniglot", "miniImageNet", 'cifarfs'],
                        default="cifarfs", help="Dataset to use")
    parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                        help="Whether to run traing only, testing only, or both training and testing.")
    parser.add_argument("--seed", type=int, default=13,
                        help="dataset seeds")
    parser.add_argument("--d_theta", type=int, default=40,
                        help="Size of the feature extractor output.")
    parser.add_argument("--d_rn_f", type=int, default=128,
                        help="Size of the random feature base.")
    parser.add_argument("--shot", type=int, default=3,
                        help="Number of training examples.")
    parser.add_argument("--way", type=int, default=10,
                        help="Number of classes.")
    parser.add_argument("--test_shot", type=int, default=None,
                        help="Shot to be used at evaluation time. If not specified 'shot' will be used.")
    parser.add_argument("--test_way", type=int, default=None,
                        help="Way to be used at evaluation time. If not specified 'way' will be used.")
    parser.add_argument("--tasks_per_batch", type=int, default=32,
                        help="Number of tasks per batch.")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples from q.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--test_iterations", type=int, default=10,
                        help="test_iterations.")
    parser.add_argument("--iterations", type=int, default=20000,
                        help="Number of training iterations.")
    parser.add_argument("--checkpoint_dir", "-c", default='./checkpoint',
                        help="Directory to save trained models.")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout keep probability.")
    parser.add_argument("--test_model_path", "-m", default='./checkpoint/best_validation',
                        help="Model to load and test.")
    parser.add_argument("--print_freq", type=int, default=200,
                        help="Frequency of summary results (in iterations).")
    parser.add_argument("--load_dir", "-lc", default='',
                        help="Directory to save trained models.")

    ## hyper_params
    parser.add_argument("--zeta", type=float, default=-1.0,
                        help="hyper param for kernel_align_loss")
    parser.add_argument("--tau", type=float, default=0.0001,
                        help="hyper param for kl_loss")
    parser.add_argument("--lmd", type=float, default=0.1,
                        help="the init of lmb")

    args = parser.parse_args()

    # adjust test_shot and test_way if necessary
    if args.test_shot is None:
        args.test_shot = args.shot
    if args.test_way is None:
        args.test_way = args.way

    return args


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.ERROR)

    args = parse_command_line()

    logfile, checkpoint_path_validation, checkpoint_path_final = get_log_files(args.checkpoint_dir, args.mode, args.shot)

    if not osp.exists('./sin_figure'):
        os.makedirs('./sin_figure')

    # tf placeholders
    train_images = tf.placeholder(tf.float32, [args.tasks_per_batch,  # tasks per batch
                                               None,  # shot
                                               1],
                                  name='train_images')
    test_images = tf.placeholder(tf.float32, [args.tasks_per_batch,  # tasks per batch
                                              None,  # num test images
                                              1],
                                 name='test_images')
    train_labels = tf.placeholder(tf.float32, [args.tasks_per_batch,  # tasks per batch
                                               None,  # shot
                                               1],
                                  name='train_labels')
    test_labels = tf.placeholder(tf.float32, [args.tasks_per_batch,  # tasks per batch
                                              None,  # num test images
                                              1],
                                 name='test_labels')
    data_generator = DataGenerator(args.shot + args.way, args.tasks_per_batch)
    # initial_state_c = tf.placeholder(dtype=tf.float32, shape=[None, args.d_theta], name="initial_state_c")
    initial_state_fw_c = tf.placeholder(dtype=tf.float32, shape=[None, args.d_theta], name="initial_state_fw_c")
    initial_state_fw_h = tf.placeholder(dtype=tf.float32, shape=[None, args.d_theta], name="initial_state_fw_h")
    initial_state_bw_c = tf.placeholder(dtype=tf.float32, shape=[None, args.d_theta], name="initial_state_bw_c")
    initial_state_bw_h = tf.placeholder(dtype=tf.float32, shape=[None, args.d_theta], name="initial_state_bw_h")

    initial_state_fw = tf.nn.rnn_cell.LSTMStateTuple(initial_state_fw_c, initial_state_fw_h)
    initial_state_bw = tf.nn.rnn_cell.LSTMStateTuple(initial_state_bw_c, initial_state_bw_h)
    LSTM_cell = tf.nn.rnn_cell.LSTMCell(args.d_theta)
    zero_state = LSTM_cell.zero_state(batch_size=1, dtype=tf.float32)

    with tf.variable_scope('hyper_params'):
        # b_mu = init('b_mu', [1, args.d_theta], tf.zeros_initializer)  # base_mu
        # b_log_var = init('b_log_var', [1, args.d_theta], tf.zeros_initializer)  # base_sigma

        lmd = init('lambda', None, tf.constant([args.lmd]))  # regularization
        lmd_abs = tf.abs(lmd)
        gamma = init('gamma', None, tf.constant([1.0]))  # calibration params
        beta = init('beta', None, tf.constant([.0]))  # calibration params
        # zeta = init('zeta', None, tf.constant([0.0])) # kernel_align_loss

        eps = np.random.normal(0.0, 1.0, [args.d_rn_f, args.d_theta])  # eps for bases
        bias = np.random.uniform(0.0, 2 * np.pi, [args.d_rn_f, 1])  # bias for bases

    def compute_base_distri(inputs):
        train_inputs, test_inputs = inputs

        with tf.variable_scope('shared_features'):
            # extract features from train and test data
            features_train = sin_function(train_inputs, args.d_theta)
            features_test = sin_function(test_inputs, args.d_theta)
            features_train = normalize(features_train)
            features_test = normalize(features_test)


            support_mean_features = features_train
            support_all_mean_features = tf.expand_dims(tf.reduce_mean(support_mean_features, axis=0), axis=0)

        return [support_mean_features, support_all_mean_features, features_train, features_test]

    batch_base_d_output = tf.map_fn(fn=compute_base_distri,
                                    elems=(train_images, test_images),
                                    dtype=[tf.float32, tf.float32, tf.float32, tf.float32],
                                    parallel_iterations=args.tasks_per_batch)

    batch_mean_features, batch_mean_all_features, batch_features_train, batch_features_test = batch_base_d_output

    nus_sgms = tf.unstack(batch_mean_all_features, axis=0)
    outputs_l, output_state_fw, output_state_bw = bidirectionalLSTM(
        'bi-lstm',
        nus_sgms,
        layer_sizes=args.d_theta,
        initial_state_fw=initial_state_fw,
        initial_state_bw=initial_state_bw)

    # batch_mean_all_features_list = tf.unstack(batch_mean_all_features, axis=0)
    # outputs_l, states = rnn.static_rnn(GRU_cell, batch_mean_all_features_list, initial_state_h, dtype=tf.float32, scope='rnn')

    batch_representation = tf.stack(outputs_l, axis=0)

    def evaluate_task(inputs):
        features_train, train_outputs, features_test, test_outputs, representation, mean_features = inputs

        q_representation = multihead_attention(features_test, mean_features, mean_features)
        with tf.variable_scope('inference_network'):
            r_mu = inference_block(representation, args.d_theta, args.d_theta, 'infer_mean')
            r_log_var = inference_block(representation, args.d_theta, args.d_theta, 'infer_var')

            p_mu = inference_block(q_representation, args.d_theta, args.d_theta, 'p_mean')
            p_log_var = inference_block(q_representation, args.d_theta, args.d_theta, 'p_var')

        # Infer classification layer from q
        with tf.variable_scope('classifier'):
            # compute bases
            rs = tf.squeeze(sample_normal(r_mu, r_log_var, args.d_rn_f, eps_=eps))

            # compute the support kernel
            x_supp_phi_t = rand_features(rs, tf.transpose(features_train, [1, 0]), bias)  # (d_rn_f , w*s)
            support_kernel = dotp_kernel(tf.transpose(x_supp_phi_t), x_supp_phi_t)  # (w*s , w*s)

            # closed-form solution with trainable lmd
            alpha = tf.matmul(
                tf.matrix_inverse(support_kernel + (lmd_abs + 0.01) * tf.eye(tf.shape(support_kernel)[0])),
                train_outputs)
            x_que_phi_t = rand_features(rs, tf.transpose(features_test, [1, 0]),
                                        bias)  # tf.cos(tf.matmul(rs, tf.transpose(features_test, [1, 0])))

            # compute the cross kernel
            cross_kernel = dotp_kernel(tf.transpose(x_supp_phi_t), x_que_phi_t)

            # prediction with calibration params
            logits = gamma * tf.matmul(cross_kernel, alpha, transpose_a=True) + beta

            # align loss
            target_kernel = dotp_kernel(train_outputs, tf.transpose(train_outputs))
            target_kernel = 0.99 * (target_kernel + 0.01)
            kernel_align_loss = cosine_dist(target_kernel, support_kernel)

            # kl loss
            kl_loss = KL_divergence(r_mu, r_log_var, p_mu, p_log_var)

            # cross entry loss
            cross_entry_loss = mse(logits, test_outputs)

            task_loss = cross_entry_loss + args.zeta * kernel_align_loss + args.tau * kl_loss

        return [task_loss, cross_entry_loss, kernel_align_loss, kl_loss, logits]

    # tf mapping of batch to evaluation function
    batch_output = tf.map_fn(fn=evaluate_task,
                             elems=(batch_features_train, train_labels, batch_features_test, test_labels,
                                    batch_representation, batch_mean_features),
                             dtype=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                             parallel_iterations=args.tasks_per_batch)

    # average all values across batch
    batch_losses, batch_cross_entry_losses, batch_align_losses, batch_kl_losses, batch_logits = batch_output
    loss = tf.reduce_mean(batch_losses)
    loss_ce = tf.reduce_mean(batch_cross_entry_losses)
    loss_ka = tf.reduce_mean(batch_align_losses)
    loss_kl = tf.reduce_mean(batch_kl_losses)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print_and_log(logfile, "Options: %s\n" % args)
        saver = tf.train.Saver()

        state_fw, state_bw = sess.run(zero_state), sess.run(zero_state)

        if args.mode == 'train' or args.mode == 'train_test':
            # train the model

            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            train_step = optimizer.minimize(loss)  # , var_list=opt_list)

            validation_batches = 200
            iteration = 0
            best_validation_loss = np.inf
            train_iteration_accuracy = []
            sess.run(tf.global_variables_initializer())
            if args.load_dir:
                saver.restore(sess, save_path=args.load_dir)
            # Main training loop
            min_loss = 1000000.
            while iteration < args.iterations:
                batch_x, batch_y, amp, phase, omega = data_generator.generate()

                train_inputs = batch_x[:, :args.shot, :]
                train_outputs = batch_y[:, :args.shot, :]
                test_inputs = batch_x[:, args.shot:, :]  # b used for testing
                test_outputs = batch_y[:, args.shot:, :]

                feed_dict = {train_images: train_inputs, test_images: test_inputs,
                             train_labels: train_outputs, test_labels: test_outputs,
                             initial_state_fw: state_fw,
                             initial_state_bw: state_bw}
                _, curr_states_fw, curr_states_bw, \
                iteration_loss, iteration_loss_ce, iteration_loss_ka, iteration_loss_kl, \
                lmd_, gamma_, beta_ \
                    = sess.run([train_step, output_state_fw, output_state_bw,
                                loss, loss_ce, loss_ka, loss_kl,
                                lmd_abs, gamma, beta], feed_dict)

                state_fw, state_bw = curr_states_fw, curr_states_fw
                # np.save('./state_5/'+str(iteration)+'_state.npy', state_h)

                if iteration % args.print_freq == 0:
                    # compute accuracy on validation set

                    batch_x, batch_y, amp, phase, omega = data_generator.generate(train=False)

                    train_inputs = batch_x[:, :args.shot, :]
                    train_outputs = batch_y[:, :args.shot, :]

                    x = np.arange(-5., 5., 0.2)

                    test_x = np.tile(x.reshape(1, -1, 1), reps=[args.tasks_per_batch, 1, 1])
                    test_y_list = []
                    for i in range(args.tasks_per_batch):
                        y = amp[i] * np.sin(omega[i] * x - phase[i])
                        test_y_list.append(y)
                    test_y = np.expand_dims(np.array(test_y_list), axis=2)
                    feed_dict = {train_images: train_inputs, test_images: test_x,
                                 train_labels: train_outputs, test_labels: test_y, initial_state_fw: state_fw,
                                     initial_state_bw: state_bw}
                    test_predict, test_loss = sess.run([batch_logits, loss_ce], feed_dict)
                    if test_loss< min_loss:
                        min_loss = test_loss
                        saver.save(sess, save_path=checkpoint_path_validation)
                        np.save(checkpoint_path_validation + '_state_fw.npy', state_fw)
                        np.save(checkpoint_path_validation + '_state_bw.npy', state_bw)
                    for i in range(args.tasks_per_batch):
                        y = test_y_list[i]

                        fig, ax = plt.subplots()
                        ax.plot(x, y, color="#2c3e50", linewidth=0.8, label="Truth")
                        ax.scatter(train_inputs[i].reshape(-1), train_outputs[i].reshape(-1), color="#2c3e50",
                                   label="Training Set")
                        ax.plot(x, test_predict[i].reshape(-1), label="After Shift", color='#e74c3c', linestyle='--')
                        ax.legend()
                        plt.savefig('./sin_figure/iter_'+str(iteration)+'_batch_'+ str(i)+'_figure.png')

                    print('itr: ' + str(iteration) + ', test_loss: ' + str(test_loss))
                    # save checkpoint if validation is the best so far

                    print_and_log(logfile,
                                  'Iteration: {}, Loss: {:5.3f}, Loss_ce: {:5.3f}, Loss_ka: {:5.3f}, Loss_kl: {:5.3f},'.format(
                                      iteration, iteration_loss, iteration_loss_ce, iteration_loss_ka,
                                      iteration_loss_kl))
                    print_and_log(logfile, 'lmd_{}, gamma_{}, beta_{}'.format(lmd_, gamma_, beta_))


                iteration += 1
            # save the checkpoint from the final epoch
            saver.save(sess, save_path=checkpoint_path_final)
            np.save(checkpoint_path_final + '_state_fw.npy', state_fw)
            np.save(checkpoint_path_final + '_state_bw.npy', state_bw)
            print_and_log(logfile, 'Fully-trained model saved to: {}'.format(checkpoint_path_final))
            print_and_log(logfile, 'Best validation model saved to: {}'.format(checkpoint_path_validation))



if __name__ == "__main__":
    tf.app.run()

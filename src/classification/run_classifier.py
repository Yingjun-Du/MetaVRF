
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
from features import extract_features_omniglot, extract_features_mini_imagenet
from inference import infer_classifier, inference_block
from utilities import *  # sample_normal, multinoulli_log_density, print_and_log, get_log_files
from data import get_data
import os
"""
parse_command_line: command line parser
"""

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", choices=["Omniglot", "miniImageNet", 'tieredImageNet', 'cifarfs'],
                        default="miniImageNet", help="Dataset to use")
    parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                        help="Whether to run traing only, testing only, or both training and testing.")
    parser.add_argument("--seed", type=int, default=42,
                        help="dataset seeds")
    parser.add_argument("--d_theta", type=int, default=256,
                        help="Size of the feature extractor output.")
    parser.add_argument("--d_rn_f", type=int, default=512,
                        help="Size of the random feature base.")
    parser.add_argument("--shot", type=int, default=1,
                        help="Number of training examples.")
    parser.add_argument("--way", type=int, default=5,
                        help="Number of classes.")
    parser.add_argument("--test_shot", type=int, default=None,
                        help="Shot to be used at evaluation time. If not specified 'shot' will be used.")
    parser.add_argument("--test_way", type=int, default=None,
                        help="Way to be used at evaluation time. If not specified 'way' will be used.")
    parser.add_argument("--tasks_per_batch", type=int, default=2,
                        help="Number of tasks per batch.")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples from q.")
    parser.add_argument("--test_iterations", type=int, default=10,
                        help="test_iterations.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations.")
    parser.add_argument("--checkpoint_dir", "-c", default='./checkpoint',
                        help="Directory to save trained models.")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout keep probability.")
    parser.add_argument("--test_model_path", "-m", default='./checkpoint/best_validation',
                        help="Model to load and test.")
    parser.add_argument("--print_freq", type=int, default=10,
                        help="Frequency of summary results (in iterations).")
    parser.add_argument("--load_dir", "-lc", default='',
                        help="Directory to save trained models.")
    parser.add_argument("--aug", type=bool, default=False,
                        help="data augmentation")

    ## hyper_params
    parser.add_argument("--zeta", type=float, default=0.,
                        help="hyper param for kernel")
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

    # Load training and eval data
    data = get_data(args.dataset, seed=args.seed)

    # set the feature extractor based on the dataset
    if args.dataset == "miniImageNet" or args.dataset == 'tieredImageNet':
        feature_extractor_fn = extract_features_mini_imagenet
    else:
        feature_extractor_fn = extract_features_omniglot

    # evaluation samples
    eval_samples_train = 15
    eval_samples_test = args.shot



    # testing parameters
    test_iterations = args.test_iterations
    test_args_per_batch = 1  # always use a batch size of 1 for testing

    # tf placeholders
    train_images = tf.placeholder(tf.float32, [None,  # tasks per batch
                                               None,  # shot
                                               data.get_image_height(),
                                               data.get_image_width(),
                                               data.get_image_channels()],
                                  name='train_images')
    test_images = tf.placeholder(tf.float32, [None,  # tasks per batch
                                              None,  # num test images
                                              data.get_image_height(),
                                              data.get_image_width(),
                                              data.get_image_channels()],
                                 name='test_images')
    train_labels = tf.placeholder(tf.float32, [None,  # tasks per batch
                                               None,  # shot
                                               args.way],
                                  name='train_labels')
    test_labels = tf.placeholder(tf.float32, [None,  # tasks per batch
                                              None,  # num test images
                                              args.way],
                                 name='test_labels')
    dropout_keep_prob = tf.placeholder(tf.float32, [], name='dropout_keep_prob')
    # L = tf.constant(args.samples, dtype=tf.float32, name="num_samples")

    initial_state_fw_c = tf.placeholder(dtype=tf.float32, shape=[None, args.d_theta], name="initial_state_fw_c")
    initial_state_fw_h = tf.placeholder(dtype=tf.float32, shape=[None, args.d_theta], name="initial_state_fw_h")
    initial_state_bw_c = tf.placeholder(dtype=tf.float32, shape=[None, args.d_theta], name="initial_state_bw_c")
    initial_state_bw_h = tf.placeholder(dtype=tf.float32, shape=[None, args.d_theta], name="initial_state_bw_h")



    initial_state_fw = tf.nn.rnn_cell.LSTMStateTuple(initial_state_fw_c, initial_state_fw_h)
    initial_state_bw = tf.nn.rnn_cell.LSTMStateTuple(initial_state_bw_c, initial_state_bw_h)
    LSTM_cell = tf.nn.rnn_cell.LSTMCell(args.d_theta)
    zero_state = LSTM_cell.zero_state(batch_size=1, dtype=tf.float32)


    with tf.variable_scope('hyper_params'):
     

        lmd = init('lambda', None, tf.constant([args.lmd]))  # regularization
        lmd_abs = tf.abs(lmd)
        gamma = init('gamma', None, tf.constant([1.0]))  # calibration params
        beta = init('beta', None, tf.constant([.0]))  # calibration params
        # zeta = init('zeta', None, tf.constant([0.0])) # kernel_align_loss

        eps = np.random.normal(0.0, 1.0, [args.d_rn_f, args.d_theta])  # eps for bases
        bias = np.random.uniform(0.0, 2 * np.pi, [args.d_rn_f, 1])  # bias for bases

    def compute_base_distri(inputs):
        train_inputs, test_inputs, train_outputs, test_outputs = inputs

        with tf.variable_scope('shared_features'):
            # extract features from train and test data
            if args.aug and args.mode != 'test':
                train_inputs = data_aug(train_inputs, crop_ratio=0.8)
                test_inputs = data_aug(test_inputs, crop_ratio=0.8)

            features_train = feature_extractor_fn(images=train_inputs,
                                                  output_size=args.d_theta,
                                                  use_batch_norm=True,
                                                  dropout_keep_prob=dropout_keep_prob)
            features_test = feature_extractor_fn(images=test_inputs,
                                                 output_size=args.d_theta,
                                                 use_batch_norm=True,
                                                 dropout_keep_prob=dropout_keep_prob)
            features_train = normalize(features_train)
            features_test = normalize(features_test)
            support_mean_features_list = []
            for c in range(args.way):
                class_mask = tf.equal(tf.argmax(train_outputs, 1), c)
                class_features = tf.boolean_mask(features_train, class_mask)

                # Pool across dimensions
                nu = tf.expand_dims(tf.reduce_mean(class_features, axis=0), axis=0)
                support_mean_features_list.append(nu)
            support_mean_features = tf.concat(support_mean_features_list, axis=0)
            support_all_mean_features = tf.expand_dims(tf.reduce_mean(support_mean_features, axis=0), axis=0)
        return [support_mean_features, support_all_mean_features, features_train, features_test]

    batch_base_d_output = tf.map_fn(fn=compute_base_distri,
                                    elems=(train_images, test_images, train_labels, test_labels),
                                    dtype=[tf.float32, tf.float32, tf.float32, tf.float32],
                                    parallel_iterations=args.tasks_per_batch)

    batch_mean_features, batch_mean_all_features, batch_features_train, batch_features_test = batch_base_d_output


    outputs_l, output_state_fw, output_state_bw = bidirectionalLSTM(
        'bi-lstm',
        batch_mean_all_features,
        layer_sizes=args.d_theta,
        initial_state_fw=initial_state_fw,
        initial_state_bw=initial_state_bw)



    batch_representation = tf.stack(outputs_l, axis=0)


    # Relevant computations for a single task
    def evaluate_task(inputs):
        features_train, train_outputs, features_test, test_outputs, representation, mean_features = inputs


        q_representation = laplace_attention(features_test, mean_features, mean_features, True)
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

            # scale = tf.get_variable(name='scale', shape=logits.get_shape()[-1],
            #                         initializer=tf.constant_initializer(1.0),
            #                         trainable=True)
            # beta_new = tf.get_variable(name='beta', shape=logits.get_shape()[-1],
            #                        initializer=tf.constant_initializer(0.0),
            # #                        trainable=True)
            # logits = scale * logits + beta_new

            preds = tf.nn.softmax(logits)

            # acc
            task_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_outputs, axis=-1),
                                                            tf.argmax(preds, axis=-1)), tf.float32))

            # align loss
            target_kernel = dotp_kernel(train_outputs, tf.transpose(train_outputs))
            target_kernel = 0.99 * (target_kernel + 0.01)
            kernel_align_loss = cosine_dist(target_kernel, support_kernel)


            # kl loss
            kl_loss = KL_divergence(r_mu, r_log_var, p_mu, p_log_var)

            # cross entry loss
            cross_entry_loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=test_outputs)

            task_loss = cross_entry_loss + args.zeta * kernel_align_loss + args.tau * kl_loss

        return [task_loss, cross_entry_loss, kernel_align_loss, kl_loss, task_accuracy]

    # tf mapping of batch to evaluation function
    batch_output = tf.map_fn(fn=evaluate_task,
                             elems=(batch_features_train, train_labels, batch_features_test, test_labels,
                                    batch_representation, batch_mean_features),
                             dtype=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                             parallel_iterations=args.tasks_per_batch)

    # average all values across batch
    batch_losses, batch_cross_entry_losses, batch_align_losses, batch_kl_losses, batch_accuracies = batch_output
    loss = tf.reduce_mean(batch_losses)
    loss_ce = tf.reduce_mean(batch_cross_entry_losses)
    loss_ka = tf.reduce_mean(batch_align_losses)
    loss_kl = tf.reduce_mean(batch_kl_losses)
    accuracy = tf.reduce_mean(batch_accuracies)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print_and_log(logfile, "Options: %s\n" % args)
        saver = tf.train.Saver()

        state_fw, state_bw = sess.run(zero_state), sess.run(zero_state)

        if args.mode == 'train' or args.mode == 'train_test':

            optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
            train_step = optimizer.minimize(loss)  # , var_list=opt_list)

            validation_batches = 1000
            iteration = 0
            best_iteration = 0
            best_validation_accuracy = 0.0
            train_iteration_accuracy = []
            sess.run(tf.global_variables_initializer())
            if args.load_dir:
                saver.restore(sess, save_path=args.load_dir)
            # Main training loop
            while iteration < args.iterations:
                train_inputs, test_inputs, train_outputs, test_outputs = \
                    data.get_batch('train', args.tasks_per_batch, args.shot, args.way, eval_samples_train)

                feed_dict = {train_images: train_inputs, test_images: test_inputs,
                             train_labels: train_outputs, test_labels: test_outputs,
                             dropout_keep_prob: args.dropout,
                             initial_state_fw: state_fw,
                             initial_state_bw: state_bw}
                _, curr_states_fw, curr_states_bw, \
                iteration_loss, iteration_loss_ce, iteration_loss_ka, iteration_loss_kl, iteration_accuracy, \
                lmd_, gamma_, beta_ \
                    = sess.run([train_step, output_state_fw, output_state_bw,
                                loss, loss_ce, loss_ka, loss_kl, accuracy,
                                lmd_abs, gamma, beta], feed_dict)

                state_fw, state_bw = curr_states_fw, curr_states_fw  ################## share the same state ###########

                train_iteration_accuracy.append(iteration_accuracy)
                if (iteration > 0) and (iteration % args.print_freq == 0):
                    # compute accuracy on validation set
                    validation_iteration_accuracy = []
                    validation_iteration = 0
                    while validation_iteration < validation_batches:
                        train_inputs, test_inputs, train_outputs, test_outputs = \
                            data.get_batch('validation', test_args_per_batch, args.shot, args.way, eval_samples_test)
                        feed_dict = {train_images: train_inputs, test_images: test_inputs,
                                     train_labels: train_outputs, test_labels: test_outputs,
                                     dropout_keep_prob: 1.0,
                                     initial_state_fw: state_fw,
                                     initial_state_bw: state_bw}
                        iteration_accuracy = sess.run(accuracy, feed_dict)
                        validation_iteration_accuracy.append(iteration_accuracy)
                        validation_iteration += 1
                    validation_accuracy = np.array(validation_iteration_accuracy).mean()
                    train_accuracy = np.array(train_iteration_accuracy).mean()

                    # save checkpoint if validation is the best so far
                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        saver.save(sess=sess, save_path=checkpoint_path_validation)
                        np.save(checkpoint_path_validation + '_state_fw.npy', state_fw)
                        np.save(checkpoint_path_validation + '_state_bw.npy', state_bw)

                    print_and_log(logfile,
                                  'Iteration: {}, Loss: {:5.3f}, Loss_ce: {:5.3f}, Loss_ka: {:5.3f}, Loss_kl: {:5.3f}, '
                                  'Train-Acc: {:5.3f}, Val-Acc: {:5.3f}, best_iter: {}, Best-Acc: {:5.3f}'
                                  .format(iteration, iteration_loss, iteration_loss_ce, iteration_loss_ka,
                                          iteration_loss_kl, train_accuracy, validation_accuracy, best_iteration,
                                          best_validation_accuracy))
                    print_and_log(logfile, 'lmd_{}, gamma_{}, beta_{}'.format(lmd_, gamma_, beta_))

                    train_iteration_accuracy = []

                iteration += 1
            # save the checkpoint from the final epoch
            saver.save(sess, save_path=checkpoint_path_final)
            np.save(checkpoint_path_final + '_state_fw.npy', state_fw)
            np.save(checkpoint_path_final + '_state_bw.npy', state_bw)
            print_and_log(logfile, 'Fully-trained model saved to: {}'.format(checkpoint_path_final))
            print_and_log(logfile, 'Best validation accuracy: {:5.3f}'.format(best_validation_accuracy))
            print_and_log(logfile, 'Best validation model saved to: {}'.format(checkpoint_path_validation))

        def test_model(model_path, load=True):

            state_fw = np.load(model_path + '_state_fw.npy')
            state_bw = np.load(model_path + '_state_bw.npy')
            state_fw = (state_fw[0], state_fw[1])
            state_bw = (state_bw[0], state_bw[1])
            if load:
                saver.restore(sess, save_path=model_path)
            test_iteration = 0
            test_iteration_accuracy = []
            while test_iteration < test_iterations:
                train_inputs, test_inputs, train_outputs, test_outputs = \
                    data.get_batch('test', test_args_per_batch, args.test_shot, args.test_way,
                                   eval_samples_test)
                feedDict = {train_images: train_inputs, test_images: test_inputs,
                            train_labels: train_outputs, test_labels: test_outputs,
                            dropout_keep_prob: 1.0,
                            initial_state_fw: state_fw,
                            initial_state_bw: state_bw}
                iter_acc = sess.run(accuracy, feedDict)
                test_iteration_accuracy.append(iter_acc)
                test_iteration += 1
            test_accuracy = np.array(test_iteration_accuracy).mean() * 100.0
            confidence_interval_95 = \
                (196.0 * np.array(test_iteration_accuracy).std()) / np.sqrt(len(test_iteration_accuracy))
            print_and_log(logfile, 'Held out accuracy: {0:5.3f} +/- {1:5.3f} on {2:}'
                          .format(test_accuracy, confidence_interval_95, model_path))

        if args.mode == 'train_test':
            print_and_log(logfile, 'Train Shot: {0:d}, Train Way: {1:d}, Test Shot {2:d}, Test Way {3:d}'
                          .format(args.shot, args.way, args.test_shot, args.test_way))
            # test the model on the final trained model
            # no need to load the model, it was just trained
            # tf.reset_default_graph()
            test_model(checkpoint_path_final, load=False)

            # test the model on the best validation checkpoint so far
            test_model(checkpoint_path_validation)

        if args.mode == 'test':
            test_model(args.test_model_path)

    logfile.close()


if __name__ == "__main__":
    tf.app.run()

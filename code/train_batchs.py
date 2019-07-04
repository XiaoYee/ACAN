import argparse
import logging
import numpy as np
from time import time
import utils as U
from keras.layers import Input, Lambda
from keras.models import Sequential, Model
from my_layers import Discrepancy_loss, KL_loss, Entropy_loss, mmd_loss
import matplotlib.pyplot as plt


logging.basicConfig(
                    # filename='out.log',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
'''
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)
'''
##############################################################################################################################
# Parse arguments

parser = argparse.ArgumentParser()
# arguments related to datasets and data preprocessing
parser.add_argument("--dataset", dest="dataset", type=str, metavar='<str>', required=True, help="The name of the dataset (small_1|small_2|large|amazon)")
parser.add_argument("--source", dest="source", type=str, metavar='<str>', required=True, help="The name of the source domain")
parser.add_argument("--target", dest="target", type=str, metavar='<str>', required=True, help="The name of the source target")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=10000, help="Vocab size. '0' means no limit (default=0)")
parser.add_argument("--n-class", dest="n_class", type=int, metavar='<int>', default=2, help="The number of ouput classes")
parser.add_argument("-t", "--type", dest="model_type", type=str, metavar='<str>', default='DAS', help="Model type (default=DAS)")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file")


# hyper-parameters related to network training
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=15, help="Number of epochs (default=15)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50, help="Batch size (default=50)")

# hyper-parameters related to network structure
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=300, help="Embeddings dimension (default=300)")
parser.add_argument("-c", "--cnndim", dest="cnn_dim", type=int, metavar='<int>', default=300, help="CNN output dimension.(default=300)")
parser.add_argument("-w", "--cnnwin", dest="cnn_window_size", type=int, metavar='<int>', default=3, help="CNN window size. (default=3)")
parser.add_argument("--cnn-activation", dest="cnn_activation", type=str, metavar='<str>', default='relu', help="The activation of CNN")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5, help="The dropout probability. To disable, input 0 (default=0.5)")
parser.add_argument("--num-k", dest="num_k", type=int, metavar='<int>', default=2, help="hyper paremeter for generator update")
parser.add_argument("--lamda1", dest="lamda1", type=float, metavar='<float>', default=-0.1, help="stepB")
parser.add_argument("--lamda2", dest="lamda2", type=float, metavar='<float>', default=0.1, help="stepC")
parser.add_argument("--lamda3", dest="lamda3", type=float, metavar='<float>', default=5.0, help="stepC")
parser.add_argument("--lamda4", dest="lamda4", type=float, metavar='<float>', default=1.0, help="contra")
parser.add_argument("--hinge", dest="hinge", type=float, metavar='<float>', default=1.0, help="margin distance")
parser.add_argument("--plot", dest="plot", type=bool, metavar='<bool>', default=False, help="plot distribution")



# random seed that affects data splits and parameter intializations
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")

args = parser.parse_args()
U.print_args(args)

# small_1 and small_2 denote eperimenal setting 1 and setting 2 on the small-scale dataset respectively.
# large denotes the large-scale dataset. Table 1(b) in the paper
# amazon denotes the amazon benchmark dataset (Blitzer et al., 2007). See appendix A in the paper.
assert args.dataset in {'small_1', 'small_2', 'large', 'amazon'}
assert args.model_type == 'DAS'

# The domains contained in each dataset
if args.dataset in {'small_1', 'small_2'}:
    assert args.source in {'book', 'electronics', 'beauty', 'music'}
    assert args.target in {'book', 'electronics', 'beauty', 'music'}
elif args.dataset == 'large':
    assert args.source in {'imdb', 'yelp2014', 'cell_phone', 'baby'}
    assert args.target in {'imdb', 'yelp2014', 'cell_phone', 'baby'}
else:
    # note that the book and electronics domains of amazon benchmark are different from those in small_1 and small_2
    assert args.source in {'book', 'dvd', 'electronics', 'kitchen'} 
    assert args.target in {'book', 'dvd', 'electronics', 'kitchen'}

assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}


if args.seed > 0:
    np.random.seed(args.seed)

##############################################################################################################################
# Prepare data
if args.dataset == 'amazon':
    from read_amazon import get_data
else:
    from read import get_data

vocab, overall_maxlen, source_x, source_y, dev_x, dev_y, test_x, test_y, source_un, target_un = get_data(
    args.dataset, args.source, args.target, args.n_class, args.vocab_size)


print '------------ Traing Sets ------------'
print 'Number of labeled source examples: ', len(source_x)
print 'Number of total source examples (labeled+unlabeled): ', len(source_un)
print 'Number of unlabeled target examples: ', len(target_un)

print '------------ Development Set ------------'
print 'Size of development set: ', len(dev_x)

print '------------ Test Set -------------'
print 'Size of test set: ', len(test_x)


def batch_generator(data_list, batch_size):
    num = len(data_list[0])
    while True:
        excerpt = np.random.choice(num, batch_size)
        # print excerpt
        yield[data[excerpt] for data in data_list]

def batch_generator_large(data_list, batch_size):
    #######################################
    # Generate balanced labeled source examples.
    # Only used on large dataset as 
    # the training set is quite unbalanced.
    #######################################
    label_list = np.argmax(data_list[1], axis=-1)
    pos_inds = np.where(label_list==0)[0]
    neg_inds = np.where(label_list==1)[0]
    neu_inds = np.where(label_list==2)[0]

    while True:
        pos_sample = np.random.choice(pos_inds, batch_size/3)
        neg_sample = np.random.choice(neg_inds, batch_size/3)
        neu_sample = np.random.choice(neu_inds, batch_size/3+batch_size%3)
        excerpt = np.concatenate((pos_sample, neg_sample))
        excerpt = np.concatenate((excerpt, neu_sample))
        np.random.shuffle(excerpt)
        yield[data[excerpt] for data in data_list]

def return_ypred(y_true, y_pred):
    return y_pred
##############################################################################################################################
# Optimizer algorithm

from optimizers import get_optimizer
optimizer = get_optimizer(args)

##############################################################################################################################
# Building model

from models import build_generator, build_discriminator
import keras.backend as K

logger.info('  Building model')

generator = build_generator(args, overall_maxlen, vocab)
 
discriminator_C1 = build_discriminator(args, name='classifier1')
discriminator_C2 = build_discriminator(args, name='classifier2')

z1 = Input(shape=(overall_maxlen,))
feature_g1 = generator(z1)
prob1 = discriminator_C1(feature_g1)
combined_g_c1 = Model(z1, prob1)
combined_g_c1.compile(optimizer=optimizer,
            loss='categorical_crossentropy', metrics=['categorical_accuracy'])
combined_g_c1.summary()

z2 = Input(shape=(overall_maxlen,))
feature_g2 = generator(z2)
prob2 = discriminator_C2(feature_g2)
combined_g_c2 = Model(z2, prob2)
combined_g_c2.compile(optimizer=optimizer,
            loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# z_middle = Input(shape=(overall_maxlen,))
# feature_middle = generator(z_middle)
# output_class = middle_c1(feature_middle)
# combined_middle = Model(z_middle, output_class)

# step A
input_source = Input(shape=(overall_maxlen,))
input_un_source = Input(shape=(overall_maxlen,))
input_target = Input(shape=(overall_maxlen,))

feature_source = generator(input_source)

feature_source_un = generator(input_un_source)
feature_targrt = generator(input_target)
MMD_loss = KL_loss(args.batch_size, name='mmdy_loss')([feature_source_un, feature_targrt])

prob1 = discriminator_C1(feature_source) 
prob2 = discriminator_C2(feature_source)

combined_A = Model(inputs=[input_source, input_un_source, input_target], 
            outputs=[prob1, prob2, MMD_loss])
combined_A.compile(optimizer=optimizer,
            loss={'classifier1': 'categorical_crossentropy', 'classifier2': 'categorical_crossentropy', 'mmdy_loss': return_ypred},
            loss_weights={'classifier1': 1, 'classifier2': 1, 'mmdy_loss': args.lamda3},
            metrics={'classifier1': 'categorical_accuracy', 'classifier2': 'categorical_accuracy'})


# step B
source_input = Input(shape=(overall_maxlen,), dtype='int32', name='source_input')
target_un_input = Input(shape=(overall_maxlen,), dtype='int32', name='target_un_input')
generator.trainable = False
source_input_feature = generator(source_input)
target_un_input_feature = generator(target_un_input)
source_input_prob_C1 = discriminator_C1(source_input_feature)
source_input_prob_C2 = discriminator_C2(source_input_feature)
target_un_input_prob_C1 = discriminator_C1(target_un_input_feature)
target_un_input_prob_C2 = discriminator_C2(target_un_input_feature)
dis_loss = Discrepancy_loss(args.batch_size, name='discrepancy_loss')([target_un_input_prob_C1, target_un_input_prob_C2])


combined_fixG = Model(inputs=[source_input, target_un_input], 
            outputs=[source_input_prob_C1, source_input_prob_C2, dis_loss])
combined_fixG.compile(optimizer=optimizer,
            loss={'classifier1': 'categorical_crossentropy', 'classifier2': 'categorical_crossentropy', 'discrepancy_loss': return_ypred},
            loss_weights={'classifier1': 1, 'classifier2': 1, 'discrepancy_loss': args.lamda1},
            metrics={'classifier1': 'categorical_accuracy', 'classifier2': 'categorical_accuracy'})


def euclidean_distance(vects):
    x, y = vects
    x = x[:args.batch_size // 2]
    y = y[args.batch_size // 2:]
    # sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    sum_square = K.mean(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    y_true = y_true[:args.batch_size // 2]
    margin = args.hinge
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def rampup(epoch):
    max_rampup_epochs = 30.0
    if epoch == 0:
        return 0
    elif epoch < args.epochs:
        p = min(max_rampup_epochs, float(epoch)) / max_rampup_epochs
        p = 1.0 - p
        return np.exp(-p*p*5.0)*args.lamda4


# step C 
input_source = Input(shape=(overall_maxlen,), dtype='int32')
input_un_target = Input(shape=(overall_maxlen,), dtype='int32')
uns_weight = Input(shape=(1, ), dtype=K.floatx(), name='uns_weight')
generator.trainable = True
feature_input = generator(input_source)
discriminator_C1.trainable = False
discriminator_C2.trainable = False
feature_un_target = generator(input_un_target)

output_source_c1 = discriminator_C1(feature_input)
output_source_c2 = discriminator_C2(feature_input)

output_target_c1 = discriminator_C1(feature_un_target)
output_target_c2 = discriminator_C2(feature_un_target)

dis_loss_target = Discrepancy_loss(args.batch_size, name='discrepancy_loss_C')([output_target_c1, output_target_c2])
# SNTG for unlabeled data cluster
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name='con_loss')([feature_un_target, feature_un_target])

combined_fixC = Model(inputs=[input_source, input_un_target, uns_weight],
            outputs=[output_source_c1, output_source_c2, dis_loss_target, distance])
combined_fixC.compile(optimizer=optimizer,
            loss={'classifier1': 'categorical_crossentropy', 'classifier2': 'categorical_crossentropy','discrepancy_loss_C': return_ypred, 'con_loss': contrastive_loss},
            loss_weights={'classifier1': 1, 'classifier2': 1, 'discrepancy_loss_C': args.lamda2, 'con_loss': uns_weight},
            metrics={'classifier1': 'categorical_accuracy', 'classifier2': 'categorical_accuracy'})

###############################################################################################################################
# Training

from keras.utils.np_utils import to_categorical


from tqdm import tqdm
logger.info('----------------------------------------- Training Model ---------------------------------------------------------')

if args.dataset == 'large':
    source_gen = batch_generator_large([source_x, source_y], batch_size=args.batch_size)
else:
    source_gen = batch_generator([source_x, source_y], batch_size=args.batch_size)

source_un_gen = batch_generator([source_un], batch_size=args.batch_size)
target_un_gen = batch_generator([target_un], batch_size=args.batch_size)

samples_per_epoch = len(source_x)
batches_per_epoch = samples_per_epoch / args.batch_size

# Set the limit of batches_per_epoch to 500
# batches_per_epoch = min(batches_per_epoch, 500)
batches_per_epoch = min(192, 500)

# get_outputs = K.function([combined_middle.input, K.learning_phase()], [combined_middle.output])

best_valid_acc = 0
pred_probs = None

for ii in xrange(args.epochs):
    t0 = time()
    train_loss_c, train_metric, discre_loss, mmdd_loss, contt_loss = 0., 0., 0., 0., 0.
    
    print("ramp up: ",rampup(ii))

    for b in tqdm(xrange(batches_per_epoch)):

        batch_source_x, batch_source_y = source_gen.next()
        batch_source_un = source_un_gen.next()[0]
        batch_target_un = target_un_gen.next()[0]

        train_loss, train_loss_c1, train_loss_c2, mmd_loss, train_metric1, train_metric2 = combined_A.train_on_batch(
        [batch_source_x, batch_source_un, batch_target_un], {'classifier1': batch_source_y, 'classifier2': batch_source_y,
        'mmdy_loss': np.ones((args.batch_size, 1))})

        train_loss, train_loss_c1, train_loss_c2, dis_loss, train_metric1, train_metric2 = combined_fixG.train_on_batch(
        [batch_source_x, batch_target_un], {'classifier1': batch_source_y, 'classifier2': batch_source_y,
        'discrepancy_loss': np.ones((args.batch_size, 1))})


        pred_probs_c1 = combined_g_c1.predict(batch_target_un, batch_size=args.batch_size, verbose=0)
        pred_probs_c2 = combined_g_c2.predict(batch_target_un, batch_size=args.batch_size, verbose=0)
        pred_probs_c = pred_probs_c1+pred_probs_c2
        preds_c1 = np.argmax(pred_probs_c, axis=-1)
        batch_compare = np.zeros(args.batch_size)  # here for shape alignment
        for i in range(args.batch_size//2):
            if preds_c1[i] == preds_c1[i + args.batch_size // 2]:
                batch_compare[i] = 1

        for i in xrange(args.num_k):
            train_loss, train_loss_c1, train_loss_c2, dis_loss, cont_loss, train_metric1, train_metric2 = combined_fixC.train_on_batch([batch_source_x, \
            batch_target_un, np.full((args.batch_size, 1), rampup(ii))], {'classifier1': batch_source_y, 'classifier2': batch_source_y,
            'discrepancy_loss_C': np.ones((len(batch_target_un), 1)), 'con_loss': batch_compare})

        train_loss_c += train_loss_c1 / batches_per_epoch
        train_metric += train_metric1 / batches_per_epoch
        discre_loss += dis_loss / batches_per_epoch
        mmdd_loss += mmd_loss / batches_per_epoch
        contt_loss += cont_loss / batches_per_epoch

    tr_time = time() - t0

    valid_loss_c1, valid_metric1 = combined_g_c1.evaluate(dev_x, dev_y, batch_size=args.batch_size, verbose=1)
    valid_loss_c2, valid_metric2 = combined_g_c2.evaluate(dev_x, dev_y, batch_size=args.batch_size, verbose=1)
    
   
    logger.info('Epoch %d, train: %is' % (ii, tr_time))
    logger.info('[Train C1] loss: %.4f, [Train C1] metric: %.4f, [discrepancy loss]: %.4f, [mmd loss]: %.4f, [contra loss]: %.4f'\
                % (train_loss_c, train_metric1, discre_loss, mmdd_loss, contt_loss))
    logger.info('[Validation C1] loss: %.4f, [Validation C2] loss: %.4f, [Validation C1] metric: %.4f, [Validation C2] metric: %.4f' \
                % (valid_loss_c1, valid_loss_c2, valid_metric1, valid_metric2))
    
    # if valid_metric > best_valid_acc:
    #     best_valid_acc = valid_metric

    print("------------- Best performance on dev set so far ==> evaluating on test set -------------")
    logger.info("------------- Best performance on dev set so far ==> evaluating on test set -------------\n")

    if args.dataset == 'large':
        #pad test set so that its size is dividible by batch_size
        append = args.batch_size-(len(test_y)%args.batch_size)
        test_x_ = np.concatenate((test_x, np.zeros((append, test_x.shape[1]))))
        test_y_ = np.concatenate((test_y, np.zeros((append, test_y.shape[1]))))

        # pred_probs = combined.predict([test_x_, test_x_, test_x_, test_x_, 
        #     test_y_, np.ones((len(test_y_), 1))], batch_size=args.batch_size, verbose=1)[0]

        # pred_probs = pred_probs[:len(test_y)]

    else:
        pred_probs_c1 = combined_g_c1.predict(test_x, batch_size=args.batch_size, verbose=1)
        pred_probs_c2 = combined_g_c2.predict(test_x, batch_size=args.batch_size, verbose=1)
        pred_probs_c = pred_probs_c1+pred_probs_c2
        # print pred_probs

    from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
    preds_c1 = np.argmax(pred_probs_c1, axis=-1)
    preds_c2 = np.argmax(pred_probs_c2, axis=-1)
    pred_c = np.argmax(pred_probs_c, axis=-1)
    true = np.argmax(test_y, axis=-1)

    # Compute accuracy on test set
    logger.info("Classify C1 accuracy: " + str(accuracy_score(true, preds_c1)) + "\n")
    logger.info("Classify C2 accuracy: " + str(accuracy_score(true, preds_c2)) + "\n")
    logger.info("Classify C accuracy: "+ str(accuracy_score(true, pred_c)) + "\n")    

    # Compute macro-f1 on test set

    # p_macro, r_macro, f_macro, support_macro \
    #     = precision_recall_fscore_support(true, preds, average='macro')
    # f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    # logger.info("macro-f1: "+str(f_macro) + "\n\n")

    # if args.plot==True:
    #     batch_output_source = np.zeros((len(source_x), args.n_class))
    #     batch_output_target = np.zeros((len(target_un), args.n_class))

    #     fig, ax = plt.subplots()    
    #     colors = ['red', 'green', 'pink', 'blue']
        
    #     for ind in xrange(0, len(source_x), args.batch_size):
    #         if ind+args.batch_size > len(source_x):
    #             batch_inds = range(ind, len(source_x))
    #         else:
    #             batch_inds = range(ind, ind+args.batch_size)
    #         batch_ = source_x[batch_inds]
    #         batch_outputs = get_outputs([batch_,0])[0]
    #         for i, j in enumerate(batch_inds):
    #             batch_output_source[j] = batch_outputs[i]

        
    #     for ind in xrange(0, len(target_un), args.batch_size):
    #         if ind+args.batch_size > len(target_un):
    #             batch_inds = range(ind, len(target_un))
    #         else:
    #             batch_inds = range(ind, ind+args.batch_size)
    #         batch_ = target_un[batch_inds]
    #         batch_outputs = get_outputs([batch_,0])[0]
    #         for i, j in enumerate(batch_inds):
    #             batch_output_target[j] = batch_outputs[i]
        
    #     print("batch_output_source: ", batch_output_source.shape)
    #     print("batch_output_un_target: ",batch_output_target.shape )
        

    #     for i in range(batch_output_source.shape[0]):
    #         label_index = int(source_y[i][1])
    #         # label_index = 0
    #         ax.scatter(batch_output_source[i, 0], batch_output_source[i, 1], c=colors[label_index], s=10, label=label_index, alpha=0.7, edgecolors='none')

    #     for i in range(batch_output_target.shape[0]):
    #         # label_index = int(test_y[i][1])
    #         label_index = int(target_un_y[i][1])
    #         # label_index = 2        
    #         ax.scatter(batch_output_target[i, 0], batch_output_target[i, 1], c=colors[label_index+2], s=10, label=label_index+2, alpha=0.7, edgecolors='none')
            
    #     # plt.show()
    #     fig.savefig('source_un_target_%d.png' % (ii))
    #     plt.close(0)

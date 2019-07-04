import numpy as np
import logging
import codecs
from keras.layers import Dense, Dropout, Activation, Embedding, Input, Flatten
from keras.models import Sequential, Model
import keras.backend as K
from my_layers import Conv1DWithMasking, Max_over_time, KL_loss, Ensemble_pred_loss, mmd_loss
from keras.constraints import maxnorm


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def build_generator(args, overall_maxlen, vocab):

    ##############################################################################################################################
    # Custom CNN kernel initializer
    # Use the initialization from Kim et al. (2014) for CNN kernel initialization. 
    def my_init(shape, dtype=K.floatx()):
        return 0.01 * np.random.standard_normal(size=shape)

    ##############################################################################################################################
    # Funtion that loads word embeddings from Glove vectors
    def init_emb(emb_matrix, vocab, emb_file):
        print 'Loading word embeddings ...'
        counter = 0.
        pretrained_emb = open(emb_file)
        for line in pretrained_emb:
            tokens = line.split()
            if len(tokens) != 301:
                continue
            word = tokens[0]
            vec = tokens[1:]
            try:
                emb_matrix[0][vocab[word]] = vec
                counter += 1
            except KeyError:
                pass
               
        pretrained_emb.close()
        logger.info('%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100*counter/len(vocab)))
            
        return emb_matrix

    ##############################################################################################################################
    # Create Model
    
    cnn_padding='same'   
    vocab_size = len(vocab)

    texts = Input(shape=(overall_maxlen,), dtype='int32')
    word_emb = Embedding(vocab_size, args.emb_dim, mask_zero=True, name='word_emb')
    x = word_emb(texts)
    print 'use a cnn layer'
    conv = Conv1DWithMasking(filters=args.cnn_dim, kernel_size=args.cnn_window_size, \
        activation=args.cnn_activation, padding=cnn_padding, kernel_initializer=my_init, name='cnn')
    x = conv(x)
    print 'use max_over_time as aggregation function'
    features = Max_over_time()(x)     

    model_generate = Model(inputs=texts, outputs=features)

    if args.emb_path:
        # It takes around 3 mininutes to load pre-trained word embeddings.
        model_generate.get_layer('word_emb').set_weights(init_emb(model_generate.get_layer('word_emb').get_weights(), vocab, args.emb_path))

    return model_generate


def build_discriminator(args, name):

    latent = Input(shape=(args.cnn_dim,))
    x = Dropout(args.dropout_prob)(latent)
    # clf_initial = Dense(50, activation=K.relu, name='dense_initial' )
    clf = Dense(args.n_class, kernel_constraint=maxnorm(3), name='dense')
    # x = clf_initial(x)
    output_clf = clf(x)
    output_prob = Activation('softmax', name= 'softmax')(output_clf)
    model_discrim = Model(inputs=latent, outputs=output_prob, name=name)
    # model_middle = Model(inputs=latent, outputs=output_clf)    # use for plot
    
    return model_discrim
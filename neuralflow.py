import sys
if sys.version_info < (3,4):
    print('You are running an older version of Python!\n\n' \
          'You should consider updating to Python 3.4.0 or ' \
          'higher as the libraries built for this course ' \
          'have only been tested in Python 3.4 and higher.\n')
    print('Try installing the Python 3.5 version of anaconda '
          'and then restart `jupyter notebook`:\n' \
          'https://www.continuum.io/downloads\n\n')

# Now get necessary libraries
try:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize
except ImportError:
    print('You are missing some packages! ' \
          'We will try installing them before continuing!')
    !pip install "numpy>=1.11.0" "matplotlib>=1.5.1" "scikit-image>=0.11.3" "scikit-learn>=0.17" "scipy>=0.17.0"
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    from skimage import data
    from scipy.misc import imresize

    print('Done!')

# Import Tensorflow
try:
    import tensorflow as tf
except ImportError:
    print("You do not have tensorflow installed!")
    print("Follow the instructions on the following link")
    print("to install tensorflow before continuing:")
    print("")
    print("https://github.com/pkmital/CADL#installation-preliminaries")

# This cell includes the provided libraries from the zip file
# and a library for displaying images from ipython, which
# we will use to display the gif
try:
    from libs import utils, gif
    import IPython.display as ipyd
except ImportError:
    print("Make sure you have started notebook in the same directory" +
          " as the provided zip file which includes the 'libs' folder" +
          " and the file 'utils.py' inside of it.  You will NOT be able"
          " to complete this assignment unless you restart jupyter"
          " notebook inside the directory created by extracting"
          " the zip file or cloning the github repo.")
try:
    import pandas as pd
except ImportError:
    print("Install pandas package!!")

#try:
#    from libs import utils, gif, datasets, dataset_utils, vae, dft
#except ImportError:
#    print("Make sure you have started notebook in the same directory" +
#          " as the provided zip file which includes the 'libs' folder" +
#          " and the file 'utils.py' inside of it.  You will NOT be able"#
#          " to complete this assignment unless you restart jupyter"
#          " notebook inside the directory created by extracting"
#          " the zip file or cloning the github repo.")

# We'll tell matplotlib to inline any drawn figures like so:
#%matplotlib inline
plt.style.use('ggplot')




# Loading the data from our  dataset

#dirname = ...
dirname_1 = "../../neuralflow/data/1.0/data.csv"
dirname_2 = "../../neuralflow/data/newton.csv"
busd   = pd.read_csv(dirname_1)
solbus = pd.read_csv(dirname_2)
busd.fillna(0.0)
solbus.fillna(0.0)
print(busd.head())
print(solbus.head())
print(busd.shape)
print(solbus.shape)

busd["Bus2"] = busd['Bus']*busd['Bus']
busd["Type2"] = busd['Type']*busd['Type']
busd["Vsp2"] = busd['Vsp']*busd['Vsp']
busd["Pgi2"] = busd['Pgi']*busd['Pgi']
busd["Qgi2"] = busd['Qgi']*busd['Qgi']

#busd["Bus-Vsp"] = busd["Bus"]*busd["Vsp"]
#busd["Bus-Pgi"] = busd["Bus"]*busd["Pgi"]
#busd["Bus-Qgi"] = busd["Bus"]*busd["Qgi"]
#busd["Bus-Pli"] = busd["Bus"]*busd["Pli"]
#busd["Bus-Qli"] = busd["Bus"]*busd["Qli"]


#busd.plot.scatter(x='Pgi', y='Bus');
#busd["Pgi-Qgi"] = busd["Pgi"]*busd["Qgi"]
#busd["Pgi-Pli"] = busd["Pgi"]*busd["Pli"]
#busd["Pgi-Qli"] = busd["Pgi"]*busd["Qli"]

#busd["Pgi/Qgi"] = busd["Pgi"]/(busd["Qgi"]+1.0)
busd["Qgi/Pgi"] = busd["Qgi"]/(busd["Pgi"]+1.0)
#busd["Pli/Qli"] = busd["Pli"]/(busd["Qli"]+1.0)

busd["Vsp/Pgi"] = busd["Vsp"]/(busd["Pgi"]+1.0)
busd["Vsp/Qgi"] = busd["Vsp"]/(busd["Qgi"]+1.0)
busd["Vsp/Pli"] = busd["Vsp"]/busd["Pli"]
busd["Vsp/Qli"] = busd["Vsp"]/busd["Qli"]
#busd["Qmaxii/Qmini"] = busd["Qmax"]/busd["Qmin"]


print(busd.head())
#print(busd.shape)
#print(solbus.shape)

columns_busd = list(busd.columns.values)
columns_solbus = list(solbus.columns.values)
#for i in range(0, len(columns_busd)):
df1 = busd
df2 = solbus






# NORMALIZATION FUNCTIONs#

import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

# Function for Normalize the Datas and Split into Training and CV set...
def process_data(df1,df2,size,r_state):

    # Para normalizar los DataFrame, se extrae las columnas que no se quieren normalizar,
    # en nuestro caso para df1 es Index=[0,1] y para df2 es Index = [0]
    # Nombres de las columnas originales
    columns_features = list(df1.columns.values)
    columns_labels   = list(df2.columns.values)




    ####Normalizando Busdata
    x = df1.values
    x_scaled = preprocessing.scale(x)
    df1_norm_all = pd.DataFrame(x_scaled)
    #df1_norm_all.columns = columns_features
    df1_norm_all = df1_norm_all.drop(df1_norm_all.columns[[3]], axis=1)
    #df1_norm_all = df1_norm_all.drop(df1_norm_all.columns[[7]], axis=1)
    #df1_norm_all = df1_norm_all.drop(df1_norm_all.columns[[7]], axis=1)
    # For Solution data set
    y = df2.values
    y_scaled = preprocessing.scale(y)
    df2_norm_all = pd.DataFrame(y_scaled)
    #df2_norm_all.columns = columns_features
    #df2_norm_all = df2_norm_all.drop(df2_norm_all.columns[[7,8]], axis=1)
    #df2_norm_all.columns = columns_labels

    # Spliting

    #X_train, X_test, y_train, y_test = train_test_split(df1_norm, df2_norm, test_size= size, random_state = r_state)
    X_train, X_test, y_train, y_test = train_test_split(df1_norm_all, df2_norm_all, test_size= size, random_state = r_state)

    X_cv, X_test, y_cv, y_test = train_test_split(X_test,y_test, test_size = 0.50, random_state = r_state)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_cv = np.array(X_cv)
    y_cv = np.array(y_cv)

    return (X_train, y_train, X_test, y_test, X_cv, y_cv, df1_norm_all,df2_norm_all)

X_train, y_train, X_test, y_test, X_cv, y_cv, df1_norm_all, df2_norm_all  = process_data(df1,df2, 0.30, r_state = None )

def deprocess(X,Y,df1,df2):
    ####Normalizando Busdata

    columns_features = list(df1.columns.values)
    columns_labels   = list(df2.columns.values)

    x = df1.values
    x_mean = x.mean(axis=0)
    x_std =  x.std(axis=0)
    df1_X = x*x_std  + x_mean
    df1_X = pd.DataFrame(df1_X)
    #df1_X.columns = columns_features


    y = df2.values
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    df2_Y = y*y_std + y_mean
    df2_Y = pd.DataFrame(df2_Y)
    df2_Y.columns = columns_labels



    return (df1_X, df2_Y)

#Return original DATA
#df1_X, df2_Y = deprocess(df1_norm_all,df2_norm_all, df1,df2)

#print(df1_X.head())
#print(df2_Y.head())

print("NORMALIZED X_train")
print(X_train.shape)
print("....................")
print("NORMALIZED y_train")
print(y_train.shape)


# STARTING SESSION #
print("Starting session...")
sess = tf.InteractiveSession()


print("Building model...")

def build_model(xs, ys, n_neurons, n_layers, activation_fn,
                final_activation_fn, cost_type):

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if xs.ndim != 2:
        raise ValueError(
            'xs should be a n_observates x n_features, ' +
            'or a 2-dimensional array.')
    if ys.ndim != 2:
        raise ValueError(
            'ys should be a n_observates x n_features, ' +
            'or a 2-dimensional array.')

    # Extraring the number of features and targets
    # for the input and output of the NN


    n_xs = xs.shape[1]
    n_ys = ys.shape[1]

    X = tf.placeholder(name='X', shape=[None, n_xs],
                       dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=[None, n_ys],
                       dtype=tf.float32)

    current_input = X
    for layer_i in range(n_layers):
        current_input = utils.linear(
            current_input, n_neurons,
            activation=activation_fn,
            name='layer{}'.format(layer_i))[0]

    Y_pred = utils.linear(
        current_input, n_ys,
        activation=final_activation_fn,
        name='pred')[0]

    if cost_type == 'l1_norm':
        cost = tf.reduce_mean(tf.reduce_sum(
                tf.abs(Y - Y_pred), 1))
    elif cost_type == 'l2_norm':
        cost = tf.reduce_mean(tf.reduce_sum(
                tf.squared_difference(Y, Y_pred), 1))
    else:
        raise ValueError(
            'Unknown cost_type: {}.  '.format(
            cost_type) + 'Use only "l1_norm" or "l2_norm"')

    return {'X': X, 'Y': Y, 'Y_pred': Y_pred, 'cost': cost}


print("TRAINING....")
def train(X,y,X_cv,y_cv,X_test,y_test,
          learning_rate=0.0001,
          batch_size=50000,
          n_iterations=2000,
          gif_step=20,
          n_neurons=100,
          n_layers=10,
          #activation_fn=tf.nn.relu,
          activation_fn = tf.nn.sigmoid,
          #final_activation_fn = tf.nn.softmax,
          #final_activation_fn=tf.nn.tanh,
          final_activation_fn=None,
          cost_type='l2_norm'):

    #N, F = df1.shape

    #N, features = X_train.shape

    #all_xs, all_ys = [], []
    #xs, ys = df1, df2
    #all_xs.append(xs)
    #all_ys.append(ys)

    Xs = np.array(X).reshape(-1,X.shape[1])
    ys = np.array(y).reshape(-1,y.shape[1])

    #Xs_cv = np.array(X_cv).reshape(-1,9)
    #yx_cv = np.array(y_cv).reshape(-1,9)
    Xs_cv = np.array(X_cv).reshape(-1,X_cv.shape[1])
    ys_cv = np.array(y_cv).reshape(-1,y_cv.shape[1])

    Xs_test = np.array(X_test).reshape(-1,X_test.shape[1])
    ys_test = np.array(y_test).reshape(-1,y_test.shape[1])

    #xs = np.array(all_xs).reshape(-1, 10)
    #ys = np.array(all_ys).reshape(-1, 9)
    #ys = ys

    g = tf.Graph()
    #g_cv = tf.Graph()

    with tf.Session(graph=g) as sess:



        # Training

        model = build_model(Xs, ys, n_neurons, n_layers,
                            activation_fn, final_activation_fn, cost_type)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(model['cost'])

        predicted_y = tf.argmax(model['Y_pred'], 1)
        actual_y = tf.argmax(model['Y'], 1)

        correct_prediction = tf.equal(predicted_y, actual_y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(tf.all_variables())


        gifs = []
        costs_training = []
        costs_cv = []


        step_i = 0
        for it_i in range(n_iterations):
            # Get a random sampling of the dataset
            idxs = np.random.permutation(range(len(Xs)))

            # The number of batches we have to iterate over
            n_batches = len(idxs) // batch_size
            training_cost = 0




            idxs_cv = np.random.permutation(range(len(Xs_cv)))
            n_batches_cv = len(idxs_cv) // batch_size
            cv_cost = 0

            # Now iterate over our stochastic minibatches:
            for batch_i in range(n_batches):

                # Get just minibatch amount of data
                idxs_i = idxs[batch_i * batch_size:
                              (batch_i + 1) * batch_size]

                # And optimize, also returning the cost so we can monitor
                # how our optimization is doing.
                cost = sess.run(
                    [model['cost'], optimizer],
                    feed_dict={model['X']: Xs[idxs_i],
                               model['Y']: ys[idxs_i]})[0]
                training_cost += cost


            for batch_i in range(n_batches_cv):
                idxs_i_cv = idxs_cv[batch_i * batch_size:
                                    (batch_i + 1) * batch_size]
                cost_cv = sess.run(
                    [model['cost'], optimizer],
                    feed_dict={model['X']: Xs_cv[idxs_i_cv],
                               model['Y']: ys_cv[idxs_i_cv]})[0]

                cv_cost += cost_cv

            #control_list = []
            #control = training_cost/n_batches
            #control_list.append(control)


            print('iteration {}/{}: cost {},  cost_cv {} '.format(
                    it_i + 1, n_iterations, training_cost / n_batches, cv_cost/ n_batches))

            #valid = ds.valid
            print(sess.run(accuracy,feed_dict={
                        model['X']: X_cv,
                        model['Y']: y_cv}))

            # Also, every 20 iterations, we'll draw the prediction of our
            # input xs, which should try to recreate our image!
            #if (it_i + 1) % gif_step == 0:
             #   costs.append(training_cost / n_batches)
            #    ys_pred = model['Y_pred'].eval(
            #        feed_dict={model['X']: xs}, session=sess)
            #    img = ys_pred.reshape(imgs.shape)
            #    gifs.append(img)

            if (it_i + 1) % gif_step == 0:
                costs_training.append(training_cost / n_batches)

                costs_cv.append(cv_cost/n_batches)


                #plt.plot(costs_training,'r--',label='Iteration %d' %it_i)
                #plt.plot(costs_training,'r--',costs_cv,'b--',label='Iteration %d' %it_i)
                #plt.ylabel('Error')
                #plt.xlabel('Number of training examples')
                #plt.show()




        # Print final test accuracy:
        #test = ds.test
        print("final Accuracy",sess.run(accuracy,
                       feed_dict={model['X']: X_test,model['Y']: y_test}))


        save_path = saver.save(sess, "../../model.ckpt")
        print("Model saved in file: %s" % save_path)


                #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


powerplow_train_model = train(X_train,y_train,X_cv,y_cv,X_test,y_test,
          learning_rate=0.001,
          batch_size=57,
          n_iterations=100000,
          gif_step=20,
          n_neurons=50,
          n_layers=4,
          #activation_fn=tf.nn.tanh,
          activation_fn = tf.nn.sigmoid,
          #final_activation_fn = tf.nn.softmax,
          #final_activation_fn=tf.nn.tanh,
          final_activation_fn=None,
          cost_type='l2_norm')

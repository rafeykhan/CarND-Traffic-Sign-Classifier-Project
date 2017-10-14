# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "traffic-signs-data/train.p"
validation_file="traffic-signs-data/valid.p"
testing_file = "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']




### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = max(y_train)-min(y_train)+1

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)




### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline

##plt.figure();
##plt.title("Train label freq");plt.xlabel("Labels");plt.ylabel("Frequency")
##plt.xticks( range(0,n_classes,2) )
##plt.hist(y_train,bins=range(n_classes+1));
##
##plt.figure();
##plt.title("validation label freq");plt.xlabel("Labels");plt.ylabel("Frequency")
##plt.xticks( range(0,n_classes,2) )
##plt.hist(y_valid,bins=range(n_classes+1));






import numpy as np
import cv2
import scipy

from skimage import transform

def rotate(image):
    angle = np.random.randint(-15,15)
    while (angle<5 and angle>-5):
        angle = np.random.randint(-15,15)
    return transform.rotate(image, angle, resize=False, center=None,preserve_range=True).astype('uint8')

def translate_r(image):
    M = np.float32([[1,0,4],[0,1,4]])
    img = cv2.warpAffine(image,M,(32,32))
    n = img[4:,4:,:]
    return transform.resize(n,(32,32,3),mode='reflect', preserve_range=True).astype('uint8')

def translate_l(image):
    M = np.float32([[1,0,-4],[0,1,-4]])
    img = cv2.warpAffine(image,M,(32,32))
    n = img[:-4,:-4,:]
    return transform.resize(n,(32,32,3),mode='reflect', preserve_range=True).astype('uint8')

def distort(image):
    pts1 = np.float32([[0,0],[29,2],[2,29],[31,31]])
    pts2 = np.float32([[0,0],[31,0],[0,31],[31,31]])
    M = cv2.getPerspectiveTransform(pts1,pts2)

    return cv2.warpPerspective(image,M,(32,32))

def zoom(image):
    pts1 = np.float32([[2,2],[29,2],[2,29],[29,29]])
    pts2 = np.float32([[0,0],[31,0],[0,31],[31,31]])
    M = cv2.getPerspectiveTransform(pts1,pts2)

    return cv2.warpPerspective(image,M,(32,32))


def process_image(image):
    n = np.random.randint(1,6)
    if n ==1:
        return rotate(image)
    if n==2:
        return translate_r(image)
    if n==3:
        return translate_l(image)
    if n==4:
        return distort(image)
    if n==5:
        return zoom(image)




target = 500

hist = np.histogram(y_train,bins=range(n_classes+1))
features = hist[0]
labels = hist[1][:-1]
#print(features); print(labels)

assert(len(labels)==len(features))
X_train_add=[]
y_train_add=[]
for i in range(len(X_train)):
    label = y_train[i]
    if features[label] < target:
        im = np.copy(X_train[i])
        im2 = process_image(im)
        X_train_add.append(im2)
        y_train_add.append(label)

print ("More data generated")
print ("Shape before..")
print (X_train.shape)
print (y_train.shape)
print()
print ("Shape of new data to be added..")
print (np.array(X_train_add).shape)
print (np.array(y_train_add).shape)
print()

##plt.figure();
##plt.title("Train label freq before");plt.xlabel("Labels");plt.ylabel("Frequency")
##plt.xticks( range(0,n_classes,2) )
##plt.hist(y_train,bins=range(n_classes+1));

X_train = np.concatenate((X_train,np.array(X_train_add)),axis=0).astype('uint8')
y_train = np.concatenate((y_train,np.array(y_train_add)),axis=0).astype('uint8')

print ("Shape after..")
print (X_train.shape)
print (y_train.shape)

##plt.figure();
##plt.title("Train label freq after");plt.xlabel("Labels");plt.ylabel("Frequency")
##plt.xticks( range(0,n_classes,2) )
##plt.hist(y_train,bins=range(n_classes+1));






import random
index = random.randint(0, len(X_train))

#index= 40850
print('index = ',index)

image = X_train[index]

#plt.figure();plt.imshow(image)
print('label = ',y_train[index])





img0 = translate_r(image)
img1 = translate_l(image)                    
img2= rotate(image)
img3=zoom(image)
img4=distort(image)


##plt.figure();
##plt.subplot(121),plt.imshow(image),plt.title('Input')
##plt.subplot(122),plt.imshow(img0),plt.title('Translate R')
##plt.show()
##plt.subplot(121),plt.imshow(img1),plt.title('Translate L')
##plt.subplot(122),plt.imshow(img2),plt.title('rotate')
##plt.show()
##plt.subplot(121),plt.imshow(img3),plt.title('zoom')
##plt.subplot(122),plt.imshow(img4),plt.title('distort')
##plt.show()



### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.


def preprocess_images(images):
    # Convert to gray image, equalize it and add expand dimension
    imgs = np.ndarray((images.shape[0], 32, 32, 1), dtype=np.uint8)
    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
        img = cv2.equalizeHist(img)
        img = np.expand_dims(img, axis=2)
        imgs[i] = img
    return imgs


#plt.figure();plt.imshow(X_train[index]);plt.title('Original')

X_train = preprocess_images(X_train)
X_valid = preprocess_images(X_valid)
X_test = preprocess_images(X_test)

#plt.figure();plt.imshow(X_train[index].squeeze(), cmap="gray");plt.title('Gray')




def normalize(images,mean,std):
    return (images-mean)/std

mean = np.mean(X_train)
std = np.std(X_train)

X_train = normalize(X_train,mean,std)
X_valid = normalize(X_valid,mean,std)
X_test = normalize(X_test,mean,std)

print ("Images normalized")




### Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten

def LeNet(x, is_training,keep_prob, rgb=True):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    in_channels = 3
    if not rgb:
        in_channels=1
    
    '''
    conv2d(
        input,
        filter,
        strides,
        padding, ...
    )
    '''
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    fw_conv1 = tf.Variable(tf.truncated_normal((5,5,in_channels,6), mean=mu,stddev=sigma))  # 5x5 filter 
    fb_conv1 = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, fw_conv1, strides= [1,1,1,1], padding = 'VALID')
    conv1 = tf.nn.bias_add(conv1,fb_conv1)
    
    # batch normalization
#    conv1 = batch_norm(conv1, decay=0.9, is_training=is_training ,updates_collections=None)  #<<<<<<<<<<<<<<<<<<<<<
    
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)
    
    
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    '''
    tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')
    '''
    conv1 = tf.nn.max_pool(conv1, ksize=(1,2,2,1), strides=(1,2,2,1), padding= 'VALID')
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    
    fw_conv2 = tf.Variable(tf.truncated_normal((5,5,6,16), mean=mu, stddev=sigma))
    fb_conv2 = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, fw_conv2, strides=(1,1,1,1), padding='VALID')
    conv2 = tf.nn.bias_add(conv2,fb_conv2)
    
    # batch normalization
#    conv2 = batch_norm(conv2,decay=0.9, is_training=is_training ,updates_collections=None)     # <<<<<<<<<<<<<<<<<<<<<<<<<
    
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
    
    conv2 = tf.nn.dropout(conv2, keep_prob)
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    w_fc1 = tf.Variable(tf.truncated_normal((5*5*16,120), mean=mu, stddev=sigma))
    b_fc1 = tf.Variable(tf.zeros(120))
    fc1 = tf.add(tf.matmul(fc0, w_fc1), b_fc1)
    
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    w_fc2 = tf.Variable(tf.truncated_normal((120,84), mean=mu, stddev=sigma))
    b_fc2 = tf.Variable(tf.zeros(84))
    fc2 = tf.add(tf.matmul(fc1,w_fc2),b_fc2)
    
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = n_classes.
    w_out = tf.Variable(tf.truncated_normal((84,n_classes), mean=mu, stddev=sigma))
    b_out = tf.Variable(tf.zeros(n_classes))
    logits = tf.add(tf.matmul(fc2,w_out),b_out)
    

    l2_loss = tf.nn.l2_loss(fw_conv1) + tf.nn.l2_loss(fw_conv2) + tf.nn.l2_loss(w_fc1) + tf.nn.l2_loss(w_fc2) + tf.nn.l2_loss(w_out)
    l2_loss = 0.0001 * l2_loss
    
    return logits, l2_loss




### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

x = tf.placeholder(tf.float32, (None, 32, 32, None))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)
one_hot_y = tf.one_hot(y, n_classes)

EPOCHS = 100
BATCH_SIZE = 128

## training pipeline ##

#rate = tf.placeholder(tf.float32)
rate = 0.0001

logits, l2_loss = LeNet(x, is_training, keep_prob, rgb=False)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_op = tf.reduce_mean(cross_entropy)# + l2_loss)

loss_operation = tf.reduce_mean(loss_op + l2_loss)

optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


## model evaluation ##
correct_prediction = tf.equal(tf.argmax(logits,1) , tf.argmax(one_hot_y,1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset : offset+BATCH_SIZE], y_data[offset : offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x:batch_x , y:batch_y , keep_prob:1.0 , is_training:False})
        total_accuracy += (accuracy * len(batch_x))
    
    return total_accuracy/num_examples








## model training ##
from sklearn.utils import shuffle

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    
    
    # Measurements use for graphing loss and accuracy
    log_batch_step = 50
    batches = []
    loss_batch = []
    train_acc_batch = []
    valid_acc_batch = []
    
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
            
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_train[offset:offset+BATCH_SIZE] , y_train[offset:offset+BATCH_SIZE]
            
            #sess.run(training_operation, feed_dict={x:batch_x , y:batch_y , keep_prob:0.6, is_training:True})
            _, l_los = sess.run([training_operation, loss_operation],
                                feed_dict={x:batch_x , y:batch_y , keep_prob:0.5, is_training:True})
            print ("loss : ",l_los)
            # Log every 50 batches
            if not (offset/BATCH_SIZE) % log_batch_step:
                # Calculate Training and Validation accuracy

                
                training_accuracy = evaluate(X_train, y_train)
                validation_accuracy = evaluate(X_valid, y_valid)
                #print("train acc = ", training_accuracy)
                #print("valid acc = ", validation_accuracy)

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l_los)
                train_acc_batch.append(training_accuracy)
                valid_acc_batch.append(validation_accuracy)
        

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")


    loss_plot = plt.subplot(211)
    loss_plot.set_title('Loss')
    loss_plot.plot(batches, loss_batch, 'g')
    loss_plot.set_xlim([batches[0], batches[-1]])


    acc_plot = plt.subplot(212)
    acc_plot.set_title('Accuracy')
    acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
    acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
    acc_plot.set_ylim([0, 1.0])
    acc_plot.set_xlim([batches[0], batches[-1]])
    acc_plot.legend(loc=4)
    plt.tight_layout()
    plt.show()
    


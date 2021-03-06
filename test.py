from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model
from input_data import get_files
import os


def get_one_image(train):
   
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]   
 
    img = Image.open(img_dir)
    plt.imshow(img)
    imag = img.resize([64, 64])  
    image = np.array(imag)
    return image


def evaluate_one_image(image_array):
    with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 3
 
       image = tf.cast(image_array, tf.float32)
       image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, 64, 64, 3])
 
       #tf.initialize_all_variables().run()

       logit = model.inference(image, BATCH_SIZE, N_CLASSES)
 
       logit = tf.nn.softmax(logit)
 
       x = tf.placeholder(tf.float32, shape=[64, 64, 3])
 
       # you need to change the directories to yours.
       logs_train_dir = '/home/ece-student/Desktop/image/input_data/'
 
       saver = tf.train.Saver()

 
       with tf.Session() as sess:
 
           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')
 
           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)
           if max_index==0:
               print('This is a bus with possibility %.6f' %prediction[:, 0])
           elif max_index==1:
               print('This is a truck with possibility %.6f' %prediction[:, 1])
           else:
               print('This is a bicycle with possibility %.6f' %prediction[:, 2])
 

               
if __name__ == '__main__':
    
    train_dir = '/home/ece-student/Desktop/image/input_data'
    train, train_label, val, val_label = get_files(train_dir, 0.2)
    img =get_one_image(val)   
    evaluate_one_image(img)


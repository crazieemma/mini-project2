import os  
import tensorflow as tf  
from PIL import Image  

  
#the path of your images
orig_picture = '/home/ece-student/Desktop/image/train/'
 
#the path of your new images
gen_picture = '/home/ece-student/Desktop/image/input_data/'
 
#input the category of your images
classes = {'truck','bus','bicycle'} 
 
#the num of your samples
num_samples = 540 
   
#make TFRecords data  
def create_record():  
    writer = tf.python_io.TFRecordWriter("car_train.tfrecords")  
    for index, name in enumerate(classes):  
        class_path = orig_picture +"/"+ name+"/"  
        for img_name in os.listdir(class_path):  
            img_path = class_path + img_name  
            img = Image.open(img_path)  
            img = img.resize((64, 64))    #resize your images
            img_raw = img.tobytes()      #transfer images to bytes
            print (index,img_raw)  
            example = tf.train.Example(  
               features=tf.train.Features(feature={  
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),  
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
               }))  
            writer.write(example.SerializeToString())  
    writer.close()  
    
def read_and_decode(filename):  
    # create file_queue 
    filename_queue = tf.train.string_input_producer([filename])  
    # create a reader from file queue  
    reader = tf.TFRecordReader()  
    # read a example from the file_queue 
    _, serialized_example = reader.read(filename_queue)  
    # get feature from serialized example 
    features = tf.parse_single_example(  
        serialized_example,  
        features={  
            'label': tf.FixedLenFeature([], tf.int64),  
            'img_raw': tf.FixedLenFeature([], tf.string)  
        })  
    label = features['label']  
    img = features['img_raw']  
    img = tf.decode_raw(img, tf.uint8)  
    img = tf.reshape(img, [64, 64, 3])  
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  
    label = tf.cast(label, tf.int32)  
    return img, label  

if __name__ == '__main__':  
    create_record()  
    batch = read_and_decode('car_train.tfrecords')  
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())  
      
    with tf.Session() as sess: 
        sess.run(init_op)    
        coord=tf.train.Coordinator()    
        threads= tf.train.start_queue_runners(coord=coord)  
        
        for i in range(num_samples):    
            example, lab = sess.run(batch)   
            img=Image.fromarray(example, 'RGB')
            img.save(gen_picture+'/'+str(i)+'samples'+str(lab)+'.jpg')    
            print(example, lab)    
        coord.request_stop()    
        coord.join(threads)   
        sess.close()


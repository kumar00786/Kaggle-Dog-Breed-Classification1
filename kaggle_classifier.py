# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:02:09 2017

@author: shorav
"""
# Classfier for dog breed classification
import datetime
import os
import tensorflow as tf
counter=2
image_path2= []
count=0
chaabi=[]
v=[]

# Here you will give the path to your image which you want to classify.
listofimages= os.listdir('/home/shorav/kaggle_update/test/11')

for image in listofimages:
    image_path = '/home/shorav/kaggle/competition/test_images/test/{}'.format(image)
    image_path2.append(image_path)
    



# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("final_labels.txt")]
print 'inside label lines'
# Unpersists graph from file
with tf.gfile.FastGFile("./final_graph.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    print 'inside graph loop '

for hg in range(len(image_path2)):
        
    image_data = tf.gfile.FastGFile(image_path2[hg], 'rb').read()
    print 'Picture Number Below:'
    count+=1
    print count
    print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

    
    with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        score_results=[]
        final_label=[]
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
            score_results.append(score)
            final_label.append(human_string)
    dictionary = dict(zip(final_label, score_results))
    for key in sorted(dictionary.iterkeys()):
        chaabi.append(key)
        v.append(dictionary[key])
        with open('read.csv','w') as j:
            
            j.write('\n')
            j.write(';')
              
            for loopu in v:
                j.write('{}'.format(loopu))
                j.write(';')
            j.write('\n')  
    v.append('\n')   
    #    score_results.append('\n')
     #   final_label.append('\n')
'''
for k in chaabi:
            j.write(k)
            j.write(';')
        j.write('\n')
        j.write(';')  

'''        




    
    










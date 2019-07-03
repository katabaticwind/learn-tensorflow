import numpy as np
import tensorflow as tf
import os
import collections
import math
import random
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

# Download data
url = 'https://www.cs.cmu.edu/~spok/grimmtmp/'

dir_name = 'stories'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

def maybe_download(filename):
  """Download a file if not present"""
  print('Downloading file: ', dir_name+ os.sep+filename)

  if not os.path.exists(dir_name+os.sep+filename):
    filename, _ = urlretrieve(url + filename, dir_name+os.sep+filename)
  else:
    print('File ',filename, ' already exists.')

  return filename

num_files = 100
filenames = [format(i, '03d')+'.txt' for i in range(1,101)]

for fn in filenames:
    maybe_download(fn)


# Read data
def read_data(filename):

  with open(filename) as f:
    data = tf.compat.as_str(f.read())
    data = data.lower()
    data = list(data)
  return data

documents = []
for i in range(num_files):
    print('\nProcessing file %s'%os.path.join(dir_name,filenames[i]))
    chars = read_data(os.path.join(dir_name,filenames[i]))
    two_grams = [''.join(chars[ch_i:ch_i+2]) for ch_i in range(0,len(chars)-2,2)]
    documents.append(two_grams)
    print('Data size (Characters) (Document %d) %d' %(i,len(two_grams)))
    print('Sample string (Document %d) %s'%(i,two_grams[:50]))

# Build dataset
def build_dataset(documents):
    chars = []
    # This is going to be a list of lists
    # Where the outer list denote each document
    # and the inner lists denote words in a given document
    data_list = []

    for d in documents:
        chars.extend(d)
    print('%d Characters found.'%len(chars))
    count = []
    # Get the bigram sorted by their frequency (Highest comes first)
    count.extend(collections.Counter(chars).most_common())

    # Create an ID for each bigram by giving the current length of the dictionary
    # And adding that item to the dictionary
    # Start with 'UNK' that is assigned to too rare words
    dictionary = dict({'UNK':0})
    for char, c in count:
        # Only add a bigram to dictionary if its frequency is more than 10
        if c > 10:
            dictionary[char] = len(dictionary)

    unk_count = 0
    # Traverse through all the text we have
    # to replace each string word with the ID of the word
    for d in documents:
        data = list()
        for char in d:
            # If word is in the dictionary use the word ID,
            # else use the ID of the special token "UNK"
            if char in dictionary:
                index = dictionary[char]
            else:
                index = dictionary['UNK']
                unk_count += 1
            data.append(index)

        data_list.append(data)

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data_list, count, dictionary, reverse_dictionary

data_list, count, dictionary, reverse_dictionary = build_dataset(documents)
print('Most common words (+UNK)', count[:5])
print('Least common words (+UNK)', count[-15:])
print('Sample data', data_list[0][:10])
print('Sample data', data_list[1][:10])
print('Vocabulary: ',len(dictionary))
vocabulary_size = len(dictionary)


# Generate batches
class DataGeneratorOHE(object):

    def __init__(self,text,batch_size,num_unroll):
        # Text where a bigram is denoted by its ID
        self._text = text
        # Number of bigrams in the text
        self._text_size = len(self._text)
        # Number of datapoints in a batch of data
        self._batch_size = batch_size
        # Num unroll is the number of steps we unroll the RNN in a single training step
        # This relates to the truncated backpropagation we discuss in Chapter 6 text
        self._num_unroll = num_unroll
        # We break the text in to several segments and the batch of data is sampled by
        # sampling a single item from a single segment
        self._segments = self._text_size//self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):
        '''
        Generates a single batch of data
        '''
        # Train inputs (one-hot-encoded) and train outputs (one-hot-encoded)
        batch_data = np.zeros((self._batch_size,vocabulary_size),dtype=np.float32)
        batch_labels = np.zeros((self._batch_size,vocabulary_size),dtype=np.float32)

        # Fill in the batch datapoint by datapoint
        for b in range(self._batch_size):
            # If the cursor of a given segment exceeds the segment length
            # we reset the cursor back to the beginning of that segment
            if self._cursor[b]+1>=self._text_size:
                self._cursor[b] = b * self._segments

            # Add the text at the cursor as the input
            batch_data[b,self._text[self._cursor[b]]] = 1.0
            # Add the preceding bigram as the label to be predicted
            batch_labels[b,self._text[self._cursor[b]+1]]= 1.0
            # Update the cursor
            self._cursor[b] = (self._cursor[b]+1)%self._text_size

        return batch_data,batch_labels

    def unroll_batches(self):
        '''
        This produces a list of num_unroll batches
        as required by a single step of training of the RNN
        '''
        unroll_data,unroll_labels = [],[]
        for ui in range(self._num_unroll):
            data, labels = self.next_batch()
            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        '''
        Used to reset all the cursors if needed
        '''
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

# Running a tiny set to see if things are correct
dg = DataGeneratorOHE(data_list[0][25:50],5,5)
u_data, u_labels = dg.unroll_batches()

# Iterate through each data batch in the unrolled set of batches
for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):
    print('\n\nUnrolled index %d'%ui)
    dat_ind = np.argmax(dat,axis=1)
    lbl_ind = np.argmax(lbl,axis=1)
    print('\tInputs:')
    for sing_dat in dat_ind:
        print('\t%s (%d)'%(reverse_dictionary[sing_dat],sing_dat),end=", ")
    print('\n\tOutput:')
    for sing_lbl in lbl_ind:
        print('\t%s (%d)'%(reverse_dictionary[sing_lbl],sing_lbl),end=", ")

# Hyperparameters
tf.reset_default_graph()

# Number of steps to unroll
num_unroll = 50

batch_size = 64 # At train time
test_batch_size = 1 # At test time

# Number of hidden neurons in the state
hidden = 64

# Input size and output Size
in_size,out_size = vocabulary_size,vocabulary_size


# Inputs and Outputs
# Train dataset
# We use unrolling over time
train_dataset, train_labels = [],[]
for ui in range(num_unroll):
    train_dataset.append(tf.placeholder(tf.float32, shape=[batch_size,in_size],name='train_dataset_%d'%ui))
    train_labels.append(tf.placeholder(tf.float32, shape=[batch_size,out_size],name='train_labels_%d'%ui))

# Validation dataset
valid_dataset = tf.placeholder(tf.float32, shape=[1,in_size],name='valid_dataset')
valid_labels = tf.placeholder(tf.float32, shape=[1,out_size],name='valid_labels')

# Test dataset
test_dataset = tf.placeholder(tf.float32, shape=[test_batch_size,in_size],name='test_dataset')

## Model Parameters & Variables
# Weights between inputs and h
W_xh = tf.Variable(tf.truncated_normal([in_size,hidden],stddev=0.02,dtype=tf.float32),name='W_xh')

# Weights between h and h
W_hh = tf.Variable(tf.truncated_normal([hidden,hidden],stddev=0.02,dtype=tf.float32),name='W_hh')

# Weights between h and y
W_hy = tf.Variable(tf.truncated_normal([hidden,out_size],stddev=0.02,dtype=tf.float32),name='W_hy')

# Maintain the previous state of hidden nodes in an un-trainable variable (Training data)
prev_train_h = tf.Variable(tf.zeros([batch_size,hidden],dtype=tf.float32),name='train_h',trainable=False)

# Maintain the previous state of hidden nodes in an un-trainable variable (Validation data)
prev_valid_h = tf.Variable(tf.zeros([1,hidden],dtype=tf.float32),name='valid_h',trainable=False)

# Maintain the previous state of hidden nodes in testing phase
prev_test_h = tf.Variable(tf.zeros([test_batch_size,hidden],dtype=tf.float32),name='test_h')


# ===============================================================================
# Train score (unnormalized) values and predictions (normalized)
y_scores, y_predictions = [],[]

# Appending the calculated output of RNN for each step in the num_unroll steps
outputs = list()

# This will be iteratively used within num_unroll steps of calculation
output_h = prev_train_h

# Calculating the output of the RNN for num_unroll steps
# (as required by the truncated BPTT)
for ui in range(num_unroll):
        output_h = tf.nn.tanh(
            tf.matmul(tf.concat([train_dataset[ui],output_h],1),
                      tf.concat([W_xh,W_hh],0))
        )
        outputs.append(output_h)

# Get the scores and predictions for all the RNN outputs we produced for num_unroll steps
y_scores = [tf.matmul(outputs[ui],W_hy) for ui in range(num_unroll)]
y_predictions = [tf.nn.softmax(y_scores[ui]) for ui in range(num_unroll)]

# We calculate train perplexity with the predictions made by the RNN
train_perplexity_without_exp = tf.reduce_sum(tf.concat(train_labels,0)*-tf.log(tf.concat(y_predictions,0)+1e-10))/(num_unroll*batch_size)

# ===============================================================================
# Validation data related inference logic
# (very similar to the training inference logic)

# Compute the next valid state (only for 1 step)
next_valid_state = tf.nn.tanh(tf.matmul(valid_dataset,W_xh)  +
                                tf.matmul(prev_valid_h,W_hh))

# Calculate the prediction using the state output of the RNN
# But before that, assign the latest state output of the RNN
# to the state variable of the validation phase
# So you need to make sure you execute valid_predictions operation
# To update the validation state
with tf.control_dependencies([tf.assign(prev_valid_h,next_valid_state)]):
    valid_scores = tf.matmul(next_valid_state,W_hy)
    valid_predictions = tf.nn.softmax(valid_scores)

# Validation data related perplexity
valid_perplexity_without_exp = tf.reduce_sum(valid_labels*-tf.log(valid_predictions+1e-10))

# ===============================================================================
# Test data realted inference logic

# Calculating hidden output for test data
next_test_state = tf.nn.tanh(tf.matmul(test_dataset,W_xh) +
                          tf.matmul(prev_test_h,W_hh)
                         )

# Making sure that the test hidden state is updated
# every time we make a prediction
with tf.control_dependencies([tf.assign(prev_test_h,next_test_state)]):
    test_prediction = tf.nn.softmax(tf.matmul(next_test_state,W_hy))

# Here we make sure that before calculating the loss, the state variable is updated
# with the last RNN output state we obtained
with tf.control_dependencies([tf.assign(prev_train_h,output_h)]):
    # We calculate the softmax cross entropy for all the predictions we obtained
    # in all num_unroll steps at once.
    rnn_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=tf.concat(y_scores,0), labels=tf.concat(train_labels,0)
    ))

# Validation RNN loss
rnn_valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
   logits=valid_scores, labels=valid_labels))


# Be very careful with the learning rate when using Adam
rnn_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

# Optimization with graident clipping
gradients, v = zip(*rnn_optimizer.compute_gradients(rnn_loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
rnn_optimizer = rnn_optimizer.apply_gradients(zip(gradients, v))

# Reset the hidden states
reset_train_h_op = tf.assign(prev_train_h,tf.zeros([batch_size,hidden],dtype=tf.float32))
reset_valid_h_op = tf.assign(prev_valid_h,tf.zeros([1,hidden],dtype=tf.float32))

# Note that we are using small imputations when resetting the test state
# As this helps to add more variation to the generated text
reset_test_h_op = tf.assign(prev_test_h,tf.truncated_normal([test_batch_size,hidden],stddev=0.01,dtype=tf.float32))

def sample(distribution):
    '''
    Sample a word from the prediction distribution
    '''
    best_idx = np.argmax(distribution)
    return best_idx


def train():

    num_steps = 26 # Number of steps we run the algorithm for
# How many training steps are performed for each document in a single step
steps_per_document = 100

# How often we run validation
valid_summary = 1

# In the book we run tests with this set to both 20 and 100
train_doc_count = 20
# Number of docs we use in a single step
# When train_doc_count = 20 => train_docs_to_use = 5
# # When train_doc_count = 100 => train_docs_to_use = 10
train_docs_to_use =5

# Store the training and validation perplexity at each step
valid_perplexity_ot = []
train_perplexity_ot = []

session = tf.InteractiveSession()
# Initializing variables
tf.global_variables_initializer().run()

print('Initialized')
average_loss = 0

# We use the first 10 documents that has
# more than (num_steps+1)*steps_per_document bigrams for creating the validation dataset

# Identify the first 10 documents following the above condition
long_doc_ids = []
for di in range(num_files):
  if len(data_list[di])>(num_steps+1)*steps_per_document:
    long_doc_ids.append(di)
  if len(long_doc_ids)==10:
    break

# Generating validation data
data_gens = []
valid_gens = []
for fi in range(num_files):
  # Get all the bigrams if the document id is not in the validation document ids
  if fi not in long_doc_ids:
    data_gens.append(DataGeneratorOHE(data_list[fi],batch_size,num_unroll))
  # if the document is in the validation doc ids, only get up to the
  # last steps_per_document bigrams and use the last steps_per_document bigrams as validation data
  else:
    data_gens.append(DataGeneratorOHE(data_list[fi][:-steps_per_document],batch_size,num_unroll))
    # Defining the validation data generator
    valid_gens.append(DataGeneratorOHE(data_list[fi][-steps_per_document:],1,1))


feed_dict = {}
for step in range(num_steps):
    print('\n')
    for di in np.random.permutation(train_doc_count)[:train_docs_to_use]:
        doc_perplexity = 0
        for doc_step_id in range(steps_per_document):

            # Get a set of unrolled batches
            u_data, u_labels = data_gens[di].unroll_batches()

            # Populate the feed dict by using each of the data batches
            # present in the unrolled data
            for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):
                feed_dict[train_dataset[ui]]=dat
                feed_dict[train_labels[ui]] = lbl

            # Running the TensorFlow operation
            _, l, step_predictions, _, step_labels, step_perplexity = \
            session.run([rnn_optimizer, rnn_loss, y_predictions,
                         train_dataset,train_labels,train_perplexity_without_exp],
                        feed_dict=feed_dict)

            # Update doc perplexity variable
            doc_perplexity += step_perplexity
            # Update average step perplexity
            average_loss += step_perplexity

        print('Document %d Step %d processed (Perplexity: %.2f).'
              %(di,step+1,np.exp(doc_perplexity/steps_per_document))
             )

        # resetting hidden state after processing a single document
        # It's still questionable if this adds value in terms of learning
        # One one hand it's intuitive to reset the state when learning a new document
        # On the other hand this approach creates a bias for the state to be zero
        # We encourage the reader to investigate further the effect of resetting the state
        session.run(reset_train_h_op)

    # Validation phase
    if step % valid_summary == 0:

      # Compute average loss
      average_loss = average_loss / (train_docs_to_use*steps_per_document*valid_summary)

      print('Average loss at step %d: %f' % (step+1, average_loss))
      print('\tPerplexity at step %d: %f' %(step+1, np.exp(average_loss)))
      train_perplexity_ot.append(np.exp(average_loss))

      average_loss = 0 # reset loss

      valid_loss = 0 # reset loss

      # calculate valid perplexity
      for v_doc_id in range(10):
          # Remember we process things as bigrams
          # So need to divide by 2
          for v_step in range(steps_per_document//2):
            uvalid_data,uvalid_labels = valid_gens[v_doc_id].unroll_batches()

            # Run validation phase related TensorFlow operations
            v_perp = session.run(
                valid_perplexity_without_exp,
                feed_dict = {valid_dataset:uvalid_data[0],valid_labels: uvalid_labels[0]}
            )

            valid_loss += v_perp

          session.run(reset_valid_h_op)
          # Reset validation data generator cursor
          valid_gens[v_doc_id].reset_indices()

      print()
      v_perplexity = np.exp(valid_loss/(steps_per_document*10.0//2))
      print("Valid Perplexity: %.2f\n"%v_perplexity)
      valid_perplexity_ot.append(v_perplexity)

      # Generating new text ...
      # We will be generating one segment having 1000 bigrams
      # Feel free to generate several segments by changing
      # the value of segments_to_generate
      print('Generated Text after epoch %d ... '%step)
      segments_to_generate = 1
      chars_in_segment = 1000

      for _ in range(segments_to_generate):
        print('======================== New text Segment ==========================')
        # Start with a random word
        test_word = np.zeros((1,in_size),dtype=np.float32)
        test_word[0,data_list[np.random.randint(0,num_files)][np.random.randint(0,100)]] = 1.0
        print("\t",reverse_dictionary[np.argmax(test_word[0])],end='')

        # Generating words within a segment by feeding in the previous prediction
        # as the current input in a recursive manner
        for _ in range(chars_in_segment):
          test_pred = session.run(test_prediction, feed_dict = {test_dataset:test_word})
          next_ind = sample(test_pred.ravel())
          test_word = np.zeros((1,in_size),dtype=np.float32)
          test_word[0,next_ind] = 1.0
          print(reverse_dictionary[next_ind],end='')

        print("")
        # Reset test state
        session.run(reset_test_h_op)
        print('====================================================================')
      print("") 

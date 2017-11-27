import csv
import string
import os
import argparse
import math
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#=======================================================================================================================
# Parse Titanic Dataset
#=======================================================================================================================

CSV_DELIM=','

# create a list where each element is a dictionary reference of a passenger
#
def get_passenger_data (data_csv):

    passenger = []

    with open(data_csv) as f:
        reader = csv.DictReader(f, delimiter=CSV_DELIM)
        header = reader.fieldnames

        print(header)
    
        for row in reader:
            passenger.append( row )

    return passenger


# get the titles/lastname 
#
def parse_name( passenger ):
    lastname_list = []
    titles_list   = []

    for p in passenger:
        name = p['Name']

        end_lastname = name.index(', ')
        lastname = name[0:end_lastname]

        end_title = name.index(' ', len(lastname)+2)
        title = name[len(lastname)+2:end_title]
    
        lastname_list.append(lastname)
        titles_list.append(title)

    return lastname_list, titles_list


# get the deck
#
def parse_cabin( passenger ):
    deck = []
    room_no = []

    for p in passenger:
        cabin = p['Cabin']

        # remove '<char> ...'
        #
        cabin = re.sub(r'^. ', '', cabin)
        cabin = re.sub(r'^.$', '', cabin)

        # if no cabin, insert NO_DECK info 
        #
        if cabin == '':
            deck.append('NONE')
            room_no.append('0')
            continue
        
        # may have multiple cabins, choose first deck that appears
        #
        cabin = cabin.split()

        deck_letter = cabin[0][0:1]
        deck.append(deck_letter)

        # average the room number
        #
        accum = 0

        for c in cabin:
            cno = int(c[1:])
            accum += cno

        room_no.append( str(accum/len(cabin)) )

    return deck, room_no


# Get Fare Per Person
#
def parse_fare( passenger ):
    far_per_person = []
    family_no = []
    
    for p in passenger:
        fno = int(p['SibSp']) + int(p['Parch']) + 1
        fpp = float(p['Fare']) / fno

        family_no.append(str(fno))
        far_per_person.append(str(fpp))

    return family_no, far_per_person


# Parse Raw Data for new Features
#
def engineer_features( passenger ):
    lastname, title = parse_name(passenger)
    deck, room_no   = parse_cabin(passenger)
    fam_no, fare_pp = parse_fare(passenger)

    N = len(passenger)

    # Add new features to dictionary
    #
    for n in range(N):
        passenger[n].update( {'Lastname':lastname[n]} )
        passenger[n].update( {'Title':title[n]} )
        passenger[n].update( {'Deck':deck[n]} )
        passenger[n].update( {'Room':room_no[n]} )
        passenger[n].update( {'Family_size':fam_no[n]} )
        passenger[n].update( {'Fare_per_person':fare_pp[n]} )
   
        #print( passenger[n]['Name']+" -- "+lastname[n]+" -- "+title[n]+ \
        #      " -- "+deck[n]+" -- "+room_no[n]+" -- "+fam_no[n]+" -- "+fare_pp[n] )

        #print( passenger[n] )

    return passenger


#=======================================================================================================================
# Convert Passenger Data to NN input data
#=======================================================================================================================

# Fields
#
# PassengerId
# Survived          - Train only
# Pclass            - 1, 2, 3
# Name              - DROP
# Sex               - M/F
# Age               - FLOAT
# SibSp             - INT
# Parch             - INT
# Ticket            - DROP
# Fare              - FLOAT
# Cabin             - DROP
# Embarked          - C/Q/S
# 
# Lastname          - STRING
# Title             - STRING
# Deck              - CHAR
# Room              - INT
# Family_size       - INT
# Fare_pre_person   - FLOAT

SEX_ENUM      = { 'male':1, 'female':2 }
EMBARKED_ENUM = { 'C':1, 'Q':2, 'S':3, '':0 }

# Add lastname mappings by...
#   if lastname not in LASTNAME_ENUM.keys():
#       LASTNAME_ENUM[lastname] = LASTNAME_ENUM_IND
#       LASTNAME_ENUM_IND += 1
#
LNAME_ENUM          = {}
LNAME_ENUM_IND      = 1

TITLE_ENUM          = {}
TITLE_ENUM_IND      = 1

DECK_ENUM           = {}
DECK_ENUM_IND       = 1

# Number of Features
#
D = 13

# Check if key is in enum - if not, add to enum
#
def add2enum( enum, ind, key ):
    if key not in enum.keys():
        enum[key] = ind
        ind += 1
    return enum, ind

# Convert to Numpy Array for NN input
#
# Drop Name Field, Ticket Field, Cabin Field
#
def data_convert2numpy( passenger, training=False ):
    global LNAME_ENUM        
    global LNAME_ENUM_IND    
    global TITLE_ENUM        
    global TITLE_ENUM_IND    
    global DECK_ENUM         
    global DECK_ENUM_IND     
    
    N = len(passenger)

    pid = np.zeros( (N) ) 
    survived = np.zeros( (N) ) 
    data = np.zeros( (N,D) ) 

    age_accum = 0
    known_age_cnt = 0
    

    # FIRST PASS - initial load of data
    #
    for n in range(N):
        p = passenger[n]

        pid[n] = int( p['PassengerId'] )
    
        if training:
            survived[n] = int( p['Survived'] )

        # Update Enums if necessary
        #
        LNAME_ENUM, LNAME_ENUM_IND = add2enum( LNAME_ENUM, LNAME_ENUM_IND, p['Lastname'] )
        TITLE_ENUM, TITLE_ENUM_IND = add2enum( TITLE_ENUM, TITLE_ENUM_IND, p['Title'] )
        DECK_ENUM,  DECK_ENUM_IND  = add2enum( DECK_ENUM,  DECK_ENUM_IND,  p['Deck'] )

        # update the data
        #
        Pclass          = float( p['Pclass'] )
        Sex             = float( SEX_ENUM[ p['Sex'] ] )

        if p['Age'] == '':
            Age         = float( -1 )
        else:
            Age         = float( p['Age'] )
            age_accum += Age
            known_age_cnt += 1


        SibSp           = float( p['SibSp'] )
        Parch           = float( p['Parch'] )
        Fare            = float( p['Fare'] )

        Embarked        = float( EMBARKED_ENUM[ p['Embarked'] ] )

        Lastname        = float( LNAME_ENUM[ p['Lastname'] ] )
        Title           = float( TITLE_ENUM[ p['Title'] ] )
        Deck            = float( DECK_ENUM[ p['Deck'] ] )
        Room            = float( p['Room'] )
        Family_size     = float( p['Family_size'] )
        Fare_per_person = float( p['Fare_per_person'] )

        data[n,:] = [ Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, \
                      Lastname, Title, Deck, Room, Family_size, Fare_per_person ]



    avg_age = age_accum/known_age_cnt

    data[:,2]


    for n in range(N):

        # fill in missing ages with avg age
        #
        if data[n,2] == -1 : 
            data[n,2] = avg_age



    if training:
        return pid, data, survived
    else:
        return pid, data




#=======================================================================================================================
# TensorFlow Neural Net
#=======================================================================================================================

lr = 0.05        # Learning Rate

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, D])
y = tf.placeholder(tf.int64,   [None])
is_training = tf.placeholder(tf.bool)

def titanic_model( X, is_training ):
    with tf.variable_scope("titanic_nn"):
        #X = tf.layers.dense( X, units=100, activation=tf.nn.relu )
        #X = tf.layers.dense( X, units=25,  activation=tf.nn.relu )
        X = tf.layers.dense( X, units=2,   activation=tf.tanh)

        return X



# run_model function from cs231n a2/TensorFlow.ipynb, with corrections
#
def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):

    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    

    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [loss_val,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, acc = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
                #print(np.sum(corr))
                #print(actual_batch_size)


            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))

        print(correct)
        print(Xd.shape[0])


        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct






def train_model(pid, survived, data):
    
    # Construct model, loss/opt, train_step
    #
    y_out = titanic_model(X, is_training)

    mean_loss  = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits=y_out, labels=tf.one_hot(y,2) ) )
    optimizer  = tf.train.AdamOptimizer(learning_rate=lr)
    train_step = optimizer.minimize(mean_loss)

    # Tensorflow Session
    # 
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    X_train = data
    y_train = survived.astype(int)

    print('Training')
    run_model( sess, y_out, mean_loss, X_train, y_train, 10, 64, 100, train_step, False)






#=======================================================================================================================
# Main and Argument Parse
#=======================================================================================================================

def main():
    descr = 'Titanic Dataset Parser'
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument('--train', '-t', help='Data to Parse')

    args = parser.parse_args()

    if (args.train == None):
        print('ERROR! PROPER MINIMUM FORMAT: python parse_titanic.py --train <CSV FILE>')
        exit(1)

    passenger = get_passenger_data(args.train)
    passenger = engineer_features(passenger) 

    pid, data, survived = data_convert2numpy(passenger, training=True)

    #print(pid)
    #print(survived)
    #print(data[200])
    
    train_model(pid, survived, data)




if __name__ == '__main__':
    main()



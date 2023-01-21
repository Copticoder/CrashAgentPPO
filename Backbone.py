import tensorflow as tf

"""This is the backbone for feature extraction of both the Actor and The critic networks. They share the same
 hidden layers but differ in the final head.
"""

class ActorCriticBackbone(tf.keras.Model):
    def __init__(self):
        super(ActorCriticBackbone,self).__init__()
        self.backbone=tf.keras.Sequential()
        self.backbone.add(tf.keras.layers.Conv2D(filters = 32,kernel_size=(3,3),strides=4,activation='relu'))
        self.backbone.add(tf.keras.layers.BatchNormalization())
        self.backbone.add(tf.keras.layers.Conv2D(filters = 64,kernel_size=(4,4),strides=2,activation='relu'))
        self.backbone.add(tf.keras.layers.BatchNormalization())
        self.backbone.add(tf.keras.layers.Conv2D(filters = 64 ,kernel_size=(4,4),strides=1,activation='relu'))
        self.backbone.add(tf.keras.layers.Flatten())
        self.backbone.add(tf.keras.layers.Dense(512,activation='relu'))

    def __call__(self,state):
        return self.backbone(state)

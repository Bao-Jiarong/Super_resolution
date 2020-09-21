'''
  Author       : Bao Jiarong
  Creation Date: 2020-09-21
  email        : bao.salirong@gmail.com
  Task         : Super Resolution
  Dataset      :Flowers(daisy, dandelion, rose, sunflower, tulip...)
'''
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import random
import cv2
import loader
import conv_ae

# np.random.seed(7)
# tf.random.set_seed(7)
# np.set_printoptions(threshold=np.inf)

# Input/Ouptut Parameters
width      = 48 << 2
height     = 48 << 2
channel    = 3
model_name = "models/super_res/flowers_120"
data_path  = "../data_img/flowers/train/"

# Step 0: Global Parameters
epochs     = 400
lr_rate    = 0.0001
batch_size = 8

# Step 1: Create Model
model = conv_ae.Conv_AE((None,height, width, channel), latent = 200, units=128)

if sys.argv[1] == "train":

    print(model.summary())
    # sys.exit()

    # Load weights:
    try:
        model.load_weights(model_name)
        print("weights were loaded successfully, we will continue training!")
    except:
        print("no pre-trained weights were found, training from scratch")

    # Step 3: Load data
    X_train, Y_train, X_valid, Y_valid = loader.load_light(data_path,width,height,True,0.8,False)
    # Define The Optimizer
    optimizer= tf.keras.optimizers.Adam(learning_rate=lr_rate)
    # Define The Loss
    #---------------------
    @tf.function
    def my_loss(y_true, y_pred):
        return tf.keras.losses.MSE(y_true=y_true, y_pred=y_pred)

    # Define The Metrics
    tr_loss = tf.keras.metrics.MeanSquaredError(name = 'tr_loss')
    va_loss = tf.keras.metrics.MeanSquaredError(name = 'va_loss')

    #---------------------
    @tf.function
    def train_step(X, Y_true):
        with tf.GradientTape() as tape:
            Y_pred = model(X, training=True)
            loss   = my_loss(y_true=Y_true, y_pred=Y_pred )
        gradients= tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        tr_loss.update_state(y_true = Y_true, y_pred = Y_pred )

    #---------------------
    @tf.function
    def valid_step(X, Y_true):
        Y_pred= model(X, training=False)
        loss  = my_loss(y_true=Y_true, y_pred=Y_pred)
        va_loss.update_state(y_true = Y_true, y_pred = Y_pred)

    #---------------------
    # start training
    L = len(X_train)
    M = len(X_valid)
    steps  = int(L/batch_size)
    steps1 = int(M/batch_size)

    for epoch in range(epochs):
        # Run on training data + Update weights
        for step in range(steps):
            images, labels = loader.get_batch_light(X_train, X_train, batch_size, width, height)
            train_step(images,labels)

            print(epoch,"/",epochs,step,steps,
                  "loss:",tr_loss.result().numpy(),end="\r")

        # Run on validation data without updating weights
        for step in range(steps1):
            images, labels = loader.get_batch_light(X_valid, X_valid, batch_size, width, height)
            valid_step(images, labels)

        print(epoch,"/",epochs,step,steps,
              "loss:",tr_loss.result().numpy(),
              "val_loss:",va_loss.result().numpy())

        # Save the model for each epoch
        model.save_weights(filepath=model_name, save_format='tf')

elif sys.argv[1] == "predict":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 4: Prepare the input
    img   = cv2.imread(sys.argv[2])
    image = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA)
    image = loader.scaling_tech(image,method="normalization")
    images = []
    origin = []
    for _ in range(5):
        k     = np.random.choice([3,5,7,9,11,13,15,17])
        j     = np.random.randint(1,90)
        res   = cv2.GaussianBlur(image,(k,k),j)
        images.append(res)
        origin.append(image)

    # True images
    images = np.array(images)
    origin = np.array(origin)

    # Step 5: Predict the class
    preds = my_model.predict(images)
    # preds = (preds[0] - preds.min())/(preds.max() - preds.min())
    true_images = np.hstack(images)
    true_origin = np.hstack(origin)
    pred_images = np.hstack(preds)
    kernel = np.ones((3,3),np.float32)/9
    pred_images = cv2.filter2D(pred_images,-1,kernel)


    images = np.vstack((true_images, true_origin, pred_images))

    cv2.imshow("imgs",images)
    cv2.waitKey(0)

elif sys.argv[1] == "predict_all":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # imgs_filenames = [os.path.abspath(d) for d in os.listdir("../../data/classification/flowers/test/daisy/")[10:12]]
    paths = ["../data_img/flowers/test/daisy/",
             "../data_img/flowers/test/dandelion/",
             "../data_img/flowers/test/rose/",
             "../data_img/flowers/test/sunflower/",
             "../data_img/flowers/test/tulip/"]
    imgs_filenames = []
    for path in paths:
        imgs_filenames.extend([os.path.join(path, file) for file in os.listdir(path)])
    # np.random.shuffle(imgs_filenames)
    imgs_filenames = imgs_filenames[:10]

    images = []
    origin = []
    for filename in imgs_filenames:
        img = cv2.imread(filename)
        img = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA)
        img = loader.scaling_tech(img,method="normalization")
        k   = np.random.choice([3,5,7,9,11,13,15,17])
        j   = np.random.randint(1,90)
        res = cv2.GaussianBlur(img,(k,k),j)
        images.append(res)
        origin.append(img)

    # True images
    images = np.array(images)
    origin = np.array(origin)

    # Predicted images
    preds = my_model.predict(images)
    # preds = (preds - preds.min())/(preds.max() - preds.min())
    true_images = np.hstack(images)
    # true_origin = np.hstack(origin)
    pred_images = np.hstack(preds)

    images = np.vstack((true_images, pred_images))
    # h = images.shape[0]
    # w = images.shape[1]
    # images = cv2.resize(images,(w << 0, h << 0))
    # preds = cv2.bilateralFilter(preds,5,2,2)

    cv2.imshow("imgs",images)
    cv2.waitKey(0)

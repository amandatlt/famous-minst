### heavily referenced links ###

# https://blog.rstudio.com/2017/09/05/keras-for-r/

### packages/paths ###

library(keras)
library(ggplot2)

path <- "/Users/amanda/Desktop/Digit/Data"

### load data ###
df_test <- read.csv(file.path(path, "test.csv"))
head(df_test,10)

df_train <- read.csv(file.path(path, "train.csv"))

### exploration ###

table(df_train$label)

names(df_train) <- gsub("pixel", "pixel.", names(df_train))

df_train$instance <- 1 : NROW(df_train) #create identifier
df_train_viz <- reshape(df_train, 
                     direction = "long",
                     idvar = "instance", 
                     varying = grep("pixel", names(df_train)),
                     v.names = "pixel",
                     timevar = "pixel_num",
                     times = 1:NROW(grep("pixel", names(df_train)))
                    )

df_train_viz <- df_train_viz[with(df_train_viz, order(instance, pixel_num)),]

df_train_viz$x <-rep(1:28, max(df_train$instance))
df_train_viz$y <-unlist(lapply(28:1, function(x) rep(x, 28)))

temp <- df_train_viz[df_train_viz$instance < 12,] #show first 12 images
ggplot(data = temp, aes(x = x, y = y, fill = pixel)) + geom_tile() + facet_wrap(~ instance + label)

### NN ###

#one-hot encode 
y_train <- to_categorical(df_train$label, 10)

#create input
x_train <- df_train[,!(names(df_train) %in% c('label', 'instance'))]
x_train <- sapply(x_train, function(x) x/255)

#create test set


model <- keras_model_sequential()
model %>% 
layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
  )

history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

plot(history)

### Predict ###

x_test <- as.matrix(df_test)/255

y_test <- model %>% predict_classes(x_test)
head(y_test)

y_to_submit <- data.frame(ImageId = rep(1:nrow(y_test)), Label = y_test)
write.csv(y_to_submit, file = file.path(path,"prediction"), col.names = TRUE, row.names = FALSE)
  

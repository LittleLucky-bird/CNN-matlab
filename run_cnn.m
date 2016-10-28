clear

model = Network();
model.add(Convolution('conv1', 7, 1, 9, 3, 0.01));
model.add(Relu('relu1'));
model.add(Pooling('pool1', 2, 0));  % output shape: 14 x 14 x  x N
model.add(Convolution('conv2', 7, 7, 7, 3, 0.01));
model.add(Convolution('conv1', 5, 1, 4, 2, 0.01));
model.add(Relu('relu1'));
model.add(Pooling('pool1', 2, 0));  % output shape: 14 x 14 x 4 x N
model.add(Convolution('conv2', 5, 4, 4, 2, 0.01));
model.add(Relu('relu2'));
model.add(Pooling('pool2', 2, 0)); % output shape: 7 x 7 x 7 x N
model.add(Linear('linear1',343,10,0.01));
% model.add(Relu('relu3'));
% model.add(Linear('linear1',100,10,0.01));
loss = SoftmaxLoss('loss');

% load data
if ~exist('data/mnist.mat', 'file')
	get_mnist('data')
end
load('data/mnist.mat');
train_data = mnist.train_data / 255;
test_data = mnist.test_data / 255;
train_label = mnist.train_label;
test_label = mnist.test_label;

update.learning_rate = 0.0002;
update.weight_decay = 0.001;
update.learning_rate = 0.1;
update.weight_decay = 0;
update.momentum = 0.9;

solver.update = update;
solver.shuffle = true;
solver.batch_size = 96;
solver.display_freq = 50;
solver.max_iter = 10000;
solver.test_freq = 500;

solve_cnn(model, loss, train_data, train_label, ...
      test_data, test_label, solver);

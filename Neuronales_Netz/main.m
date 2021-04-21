function [] = main()
%
%load('data1.mat','X');
trainingdata = load('mnist_train.csv');
y = trainingdata(:,1);              % Y
X = trainingdata(:,2:785);          % X

testdata = load('mnist_test.csv');
testlabels1 = testdata(:,1);
testX = testdata(:,2:785);

%m = size(X,1);

rand = randperm(size(testX,1)); % picks 100 random examples of data1.mat and
rand = rand(1:100);             % displays them
displayData(testX(rand,:), 28);

% imagesc(reshape(X(l,:),28,28)');



N = 784;                    % number of input neurons
N_ = 50;                    % number of hidden neurons
num_labels = 10;            % number of output neurons

% init weights
init_omega1 = init_weights(N, N_);
init_omega2 = init_weights(N_, num_labels);
init_params = [init_omega1(:) ; init_omega2(:)];

% init regularization factor
lambda = 0;

% set up max iterations for gradient descent 
i = 50;
options = optimset('MaxIter', i);

% set cost function as function handle C(p) with p being the weights
costfunc = @(x) costfunction(x, N, N_, num_labels, X, y, lambda);

% minimize cost function and returns weights where C is minimal
[params, fu, ~] = fmincg(costfunc, init_params, options);

x = 1:i;
figure
subplot(2,1,1)
plot(x,fu);
title('Cost function C without regularization');
xlabel('iterations i');
ylabel('cost function C');

% reshape optimized weights back to matrices
omega1 = reshape(params(1 : N_ * (N + 1)), N_, (N + 1));
omega2 = reshape(params(1 + (N_ * (N + 1)):end), num_labels, (N_ + 1));

% return prediction with given input and optimized weights
prediction = predict(omega1, omega2, X);
me1 = mean(double(prediction == y)) * 100;                 % accuracy in %


% again with regularization
lambda2 = 1;
costfunc2 = @(x) costfunction(x, N, N_, num_labels, X, y, lambda2);
[params2, fu2, ~] = fmincg(costfunc2, init_params, options);

subplot(2,1,2)
plot(x,fu2);
title('Cost function C with regularization');
xlabel('iterations i');
ylabel('cost function C');

omega1_ = reshape(params2(1 : N_ * (N + 1)), N_, (N + 1));
omega2_ = reshape(params2(1 + (N_ * (N + 1)):end), num_labels, (N_ + 1));

prediction2 = predict(omega1_, omega2_, X);
me2 = mean(double(prediction2 == y)) * 100;

fprintf('\n Training Set Accuracy without regularization: %f \n', me1);
fprintf('\n Training Set Accuracy with regularization: %f \n', me2);

predictionTest1 = predict(omega1, omega2, testX);
meTest1 = mean(double(predictionTest1 == testlabels1)) * 100;

predictionTest2 = predict(omega1_, omega2_, testX);
meTest2 = mean(double(predictionTest2 == testlabels1)) * 100;

fprintf('\n Test Set Accuracy without regularization: %f \n', meTest1);
fprintf('\n Test Set Accuracy with regularization: %f \n', meTest2);

hold off

end


function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
K = num_labels;

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Reshaping Y
Y = zeros(length(y),K);
for z = 1:length(y);
  Y(z,y(z)) = 1;
end

% add ones to the X matrix
X = [ones(m, 1) X];

% Forward propagation
z1 = sigmoid(Theta1 * X');
a2 = [ones(1, size(z1, 2)); z1];
a3 = sigmoid(Theta2 * a2);
h = a3;

% Computing J
J = (1/m) * (sum( sum( -1*Y'.*log(h) - (1 - Y').*log(1-h) ) ));

% Implementing regularization
% First, we toss the first columns of each Theta(i) matrix.

Theta1Reg = Theta1(:,2:size(Theta1,2));
Theta2Reg = Theta2(:,2:size(Theta2,2));

% Now implement the regularization formula described on page 6 of ex4.

Reg = (lambda/(2*m)) * (sum(sum( Theta1Reg.^2 )) + sum( sum( Theta2Reg.^2 ) ));

% Now just add the regularization term to the previously calculated J

J = J + Reg;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

% initialize for loop across training set
for t = 1:m

  % First, propogate forward on X (a1) which already contains bias node

  a1 = X(t,:);
  z2 = Theta1 * a1';

  a2 = sigmoid(z2);
  a2 = [1 ; a2];

  % Now we have our final activation layer, a3 == h(theta)

  z3 = Theta2 * a2;
  a3 = sigmoid(z3);

  % Now we can backpropagate
  % For k3 from the output layer
  d3 = a3 - Y'(:,t);

  % For k2 from the hidden layer
  % Readding a bias node Also
  z2 = [1; z2];
  d2 = (Theta2' * d3) .* sigmoidGradient(z2);

  % Strip out the bias node
  d2 = d2(2:end);

  % Accumulate gradiants
  Theta2_grad = (Theta2_grad + d3 * a2');
  Theta1_grad = (Theta1_grad + d2 * a1);

end

% Obtain unreqularized grad by dividing by m

Theta2_grad = Theta2_grad ./ m;
Theta1_grad = Theta1_grad ./ m;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% implement for Theta1 and Theta 2 when l = 0
Theta1_grad = Theta1_grad + (lambda/m).*[zeros(size(Theta1,1),1), Theta1(:,2:end)];
Theta2_grad = Theta2_grad + (lambda/m).*[zeros(size(Theta2,1),1), Theta2(:,2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

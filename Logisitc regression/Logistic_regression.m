%Amr Shehab Amin
% 37-20200
% Communication
% Logistic regression with Multiple variables 

clear;
close all;
clc;

% reading the Data Set  

Heart_data_set = readtable('heart_DD.csv');

Heart_data_set_training = table2array(Heart_data_set(1:175,1:14));
Heart_data_set_testing = table2array(Heart_data_set(175:250,1:14));


% Initializing Input(4features) and output ( target)

X1 = Heart_data_set_training(:, [1 2 3 4 ] );   
X2 = Heart_data_set_training(:, [5 6 7 8 ] ); 
X3 = Heart_data_set_training(:, [2 5 9 3 ] ); 
Feature_2=Heart_data_set_training(:, [2] ).^2;
X4 = Heart_data_set_training(:, [ 9 10 11 ] ); 
x4_final=[Feature_2 X4];

y = Heart_data_set_training(:, 14);     % Results matrix (target)

% Normalizing the features for better learning rate and not to overfit


x1_normalized=var_Normalise(X1);
x2_normalized=var_Normalise(X2);
x3_normalized=var_Normalise(X3);
x4_normalized=var_Normalise(x4_final);
 



m = length(y);    % number of training examples

% initial weights
theta1 = zeros(5, 1);
theta2 = zeros(5, 1);
theta4 = zeros(5, 1);
theta3 = zeros(5, 1);    

%randomizing the weights 
theta1(:,1)=rand;
theta2(:,1)=rand;
theta4(:,1)=rand;
theta3(:,1)=rand;


iterations = 3000;      % Iterations needed for Gradient Descent
alpha = 0.01;          % Learning Rate

%Normal Method (Stochastic Filter)
theta_normal1= normalEq(x1_normalized,y);
theta_normal2= normalEq(x2_normalized,y);
theta_normal3= normalEq(x3_normalized,y);
theta_normal4= normalEq(x4_normalized,y);

% Compute the initial Cost 
Xa = [ones(m, 1), x1_normalized];
Xb = [ones(m, 1), x2_normalized];
Xc = [ones(m, 1), x3_normalized];
Xd = [ones(m, 1), x4_normalized];
J1 =cost_logistic(Xa,y, theta1);
J2 =cost_logistic(Xb,y, theta2);
J3 =cost_logistic(Xc,y, theta3);
J4 =cost_logistic(Xd,y, theta4);

%  Gradient Descent Graph of cost vs ietrations is included in the function
[theta_1, Js1] = GradientDescent_multi_var_logisitc(Xa,y, theta1 , alpha, iterations);
figure();
[theta_2, Js2] = GradientDescent_multi_var_logisitc(Xb,y, theta2 , alpha, iterations);
figure();
[theta_3, Js3] = GradientDescent_multi_var_logisitc(Xc,y, theta3 , alpha, iterations);
 figure();
[theta_4, Js4] = GradientDescent_multi_var_logisitc(Xd,y, theta4 , alpha, iterations);




%Predicting the target

k1=Heart_data_set_testing(:, [1 2 3 4 ] );
k2=Heart_data_set_testing(:, [5 6 7 8 ] );
k3=Heart_data_set_testing(:, [2 5 9 3] );
k4=Heart_data_set_testing(:, [9 10 11] );

 Feature_2_final=Heart_data_set_testing(:, [2] ).^2;
 k4_final=[ Feature_2_final k4];
 y2 = Heart_data_set_testing(:, 14);  

 %Normalizing Testing data
 input_feautures_k1= var_Normalise(k1);
 input_feautures_k2= var_Normalise(k2);
 input_feautures_k3= var_Normalise(k3);
 input_feautures_k4= var_Normalise(k4_final);
 
 v1 = [ones(length(y2), 1), input_feautures_k1];
 v2 = [ones(length(y2), 1), input_feautures_k2];
 v3 = [ones(length(y2), 1), input_feautures_k3];
 v4 = [ones(length(y2), 1), input_feautures_k4];
 
   
%  Predecting the data by multiplying the testing features by the thetas 
  predection_features_1st_array=v1* theta_1;
  predection_features_2nd_array=v2* theta_2;
  predection_features_3rd_array=v3* theta_3;
  predection_features_4th_array=v4* theta_4;
 
%calculating error using the MSE of the cost function
 error1=cost_logistic( predection_features_1st_array,y2,transpose(theta1));
 error2=cost_logistic(predection_features_2nd_array,y2,transpose(theta1));
 error3=cost_logistic(predection_features_3rd_array,y2,transpose(theta1));
 error4=cost_logistic(predection_features_4th_array,y2,transpose(theta1));

  



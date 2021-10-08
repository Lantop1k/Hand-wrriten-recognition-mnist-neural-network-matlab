clc
clear 
close

rng(0) %reproductivity

iter = 1e-1;    %learning rate

%vector input
X=[0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0;
   0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,0;
   0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0;
   0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0;
   0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0;
   0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0;
   
   
   0,1,1,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0;
   0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,1,0,0;
   0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0;
   0,0,1,1,0,0,1,0,0,1,0,1,0,0,1,0,1,0,1,0,0,1,1,0,0;
   1,1,1,1,1,1,0,0,0,1,1,0,1,0,1,1,0,0,0,1,1,1,1,1,1;
   0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0;
  ];

%Target
T=[1,1,1,1,1,1,0,0,0,0,0,0]';

hid=1;                 %hidden neurons (single perceptron)

%select random 4 patterns for training 
idx1=datasample(1:6,4, 'replace',false);
idx0=datasample(7:12,4, 'replace',false);
idx=[idx1 idx0];

%extract the testing set
idxtest=[];
for i=1:12
   
    if isempty(find(idx==i, 1))
        idxtest=[idxtest i];
    end
end

Xtrain=X(idx,:); %training input
ytrain=T(idx);   %training target

Xtest=X(idxtest,:); %testing input 
ytest=T(idxtest);   %testing target

W =iter* randn(size(X,2),hid); %initial weight

b =iter* rand(1,hid);               %bias

J=[];  %empty array to store error at every epochs 


%Neural network training
while true
    
    for j = 1:size(Xtrain,1)
         
        z = Xtrain(j,:)*W+b;  %network output 
        pred_class = z>0;    %network output (sign)
        pred_error =  ytrain(j)-pred_class; %compute the error 
        
        W = W + iter* Xtrain(j,:)'*pred_error; %weight updating
        b=  b + iter*  pred_error;              %bias updating
    end
    
     %error computing for all the training class
     z = Xtrain*W+b; 
     pred_class = z>0;    %network output (sign)
     
     J = [J;mse(pred_class-ytrain)]; %mse error
     
     %check for stoping criteria
     if J(end)<=0.001
        break 
     end
end

%plotting training process
figure(1)
plot(1:length(J),J)
ylabel('Mean Square error')
xlabel('Epochs')
title('Training Process for Single perceptron')

%Predicting the class labels for testing data
z = Xtest*W+b;   %target
pred_class = double(z>0); %sign of target

C=confusionmat(pred_class,ytest);

acc=100*sum(diag(C))/sum(C(:))  %accuracy

miscla=sum(C(:))-sum(diag(C))   %misclassification



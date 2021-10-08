clc
clear 
close

rng(1) %reproductivity

Maxepoch = 1000; %epochs (number of iterations)
iter = 1e-3;    %learning rate

load('mnist.mat')       %load dataset

%extract 600 images  from each class 
idx=[];
for i=0:9
    idx=horzcat(idx,datasample(find(trainY==i),600, 'replace',false));
end

trainX=double(trainX(idx,:))/255;  %extract training input data
trainY=double(trainY(idx)'); %extract training target data
testX=double(testX)/255;    %extract testing  input data 
testY=double(testY);    %extract testing  target data

hid=10;                 %hidden neurons

W =iter* randn(size(trainX,2),hid); %initial weight

b =iter* rand(1,hid);               %bias

J=[];  %empty array to store error at every epochs 

%convert the training target for 10 perceptions
ytrain=zeros(size(trainX,1),10);
for i=1:size(trainX,1)
    d=trainY(i);
    ytrain(i,d+1)=1;
end

%Neural network training
epoch=0;
while true
    
    for j = 1:size(trainX,1)
         
        z = trainX(j,:)*W+b;  %network output 
        pred_class = 1./(1+exp(-z));    %network output
        pred_error =  ytrain(j,:)-pred_class; %compute the error 
        
        W = W + iter* trainX(j,:)'*pred_error; %weight updating
        b=  b + iter*  pred_error;              %bias updating
    end
    
     %error computing for all the training class
     z = trainX*W+b; 
     pred_class = 1./(1+exp(-z));    %network output
     
     J = [J;mse(pred_class-ytrain)]; %mse error
     
     if J(end)<=0.001 || epoch>=Maxepoch
        break 
     end
     
     epoch=epoch+1;
end

%plotting training process
figure(1)
plot(1:length(J),J)
xlim([0 1000])
ylabel('Mean Square error')
xlabel('Epochs')
title('Training Process for Single layer perceptron')


 %Predicting the class labels for testing data
 z = testX*W+b;               %target
 pred_class = 1./(1+exp(-z)); %network output
 
 %convert the output to single target 
 predY=zeros(size(pred_class,1),1);
 for i=1:size(pred_class,1)
     
     [~,x]=max(pred_class(i,:));
     predY(i)=x-1;
     
 end
 
%display the confusion matrix
C=confusionmat(predY,testY);
acc=100*sum(diag(C))/sum(C(:))

miscla=sum(C(:))-sum(diag(C))

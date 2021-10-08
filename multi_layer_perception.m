clc
clear 
close

rng(1) %reproductivity
Maxepoch = 1000;  %Maximum epoch
lr = 1e-4;    %learning rate

load('mnist.mat')  %load dataset

%extract 600 images  from each class 
idx=[];
for i=0:9
    idx=horzcat(idx,datasample(find(trainY==i),600, 'replace',false));
end

trainX=double(trainX(idx,:))/255;  %extract training input data
trainY=double(trainY(idx)'); %extract training target data
testX=double(testX)/255;    %extract testing  input data 
testY=double(testY);    %extract testing  target data

iterations = size(trainX,1);
hid = 15;            % No. of hidden neurons
out=10;              % No. of output neurons 

Wi = lr*randn(hid, size(trainX,2));  %Input weights
Wo = lr*randn(out,hid);  % Output weights

bi=lr*randn(hid,1);   %input bias
bo=lr*randn(out,1);   %output bias

%convert the training target for 10 perceptions
ytrain=zeros(size(trainX,1),10);
for i=1:size(trainX,1)
    d=trainY(i);
    ytrain(i,d+1)=1;
end

J=[]; %empty array to store error at every epochs 

epoch=0;
while true
    
        sumerr=0;   
    for j = 1:size(trainX,1)

        inputi=trainX(j,:)';
        A = 1./(1+exp(-Wi*inputi+bi));                       % Hidden output
        Yo = Wo*A + bo;                                      % Predicted output
         
        ca = find(ytrain(j,:)==1);                            % actual class
        [~,cp] = max(Yo);                                     % Predicted class
        
         er = zeros(out,1);                                    % For MLS error 
         for ac = 1 : out
             if Yo(ac) * ytrain(j,ac) <1 
                 er(ac) = ytrain(j,ac) - Yo(ac);
             end    
         end
         
       %Back propagation  
       Wo = Wo + lr * (er * A');                          %update output weight
       Wi = Wi + lr  * ((Wo'*er).*A.*(1-A))*inputi';     % update  input weight
           
       bo=bo+lr*er;                              %update output bias
       bi=bi+lr*((bo'*er).*A.*(1-A));          %update input bias
      
       sumerr = sumerr + sum(er.^2);
    end
    
    %compute error
    A = 1./(1+exp(-Wi*trainX'+bi));                    % Hidden output
    Yo = (Wo*A)'+bo';
    
    J = [J;mse(Yo-ytrain)]; %mse error  
    
    epoch=epoch+1;
       
     if J(end)<=0.001 || epoch>=Maxepoch
        break 
     end
     
   
end

%plotting training process
figure(1)
plot(1:epoch,J)
ylabel('Mean Square error')
xlabel('Epochs')
xlim([0 1000])
title('Training Process for Multi layer perceptron')

 %Predicting the class labels for testing data
A = 1./(1+exp(-Wi*testX'+bi));                    % Hidden output
Yo = (Wo*A)'+bo';

%predictions
ypred=zeros(length(Yo),1);
for i=1:length(Yo)
    [~,cp] = max(Yo(i,:));                      % Predicted class 
    
    ypred(i)=cp-1;
end

%compute accuracy of predictions
C=confusionmat(ypred,testY)
acc=100*sum(diag(C))/sum(C(:))

miscla=sum(C(:))-sum(diag(C))

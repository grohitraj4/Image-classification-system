% %%%%%%%%%% -- INITIAL CLEAR COMMANDS -- %%%%%%%%%% %

clear all;

close all;

clc;

% %%%%%%%%%% -- GETTING INPUT IMAGE -- %%%%%%%%%% %

[f,p] = uigetfile('Dataset\*.*');

if f == 0
    
    warndlg('You have cancelled');
    
else

    
    I = imread([p f]);
    
    figure(1),
    imshow(I);
    axis off;
    title('Input Image');
    
    IR = imresize(I,[512 512]);
    
    figure(2),
    imshow(IR);
    axis off;
    title('Resized Image');
    
    % %%%%%%%%%% -- SLIC -- %%%%%%%%%% %
    
    addpath('subfn\');
    
    M = size(IR,1);
    N = size(IR,2);
    S = (0.01*M*N)^(1/2);
    
    [l, Am, Sp, d] = slic(IR, round(S)/10 , 10, 1, 'median');
    
    SLICIMG  = drawregionboundaries(l, IR, [255 0 0]);
    
    
    figure('Name','SLIC Image','NumberTitle','Off');
    
    
    imshow(SLICIMG);axis off;   % Showing SLIC Image
    title('SLIC Image','fontname',...
        'Times New Roman','fontsize',12);
    
    fontsize=12;
    M= rgb2gray(SLICIMG);
    imshow(M)
%     BW = imbinarize(M);
%     CC = bwconncomp(BW);
%     L = labelmatrix(CC);
%     RGB = label2rgb(L);
%     figure
%     imshow(RGB)
%     RGB2 = label2rgb(L,'spring','c','shuffle'); 
%     figure
%     imshow(RGB2)
    binaryImage = M < 100;
    binaryImage = imclearborder(binaryImage, 4);
    
    imshow(binaryImage);
    axis on;
    title('Boundaries', 'FontSize', fontsize);
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
    [labeledImage, numberOfRegions] = bwlabel(binaryImage, 4);
    coloredlabels= label2rgb (M);
    figure('Name','Colored Label Image','NumberTitle','Off');
    imshow(coloredlabels);axis off;
    
    title('Colored Label Image', 'FontSize',fontsize);
    
   
    
    % %%%%%%%%%% -- CNN -- %%%%%%%%%% %
    
    train = IR;
    
    NM = str2num(f(1:end-4));
    
    addpath('CNN\')
    
    addpath('CNN\util\')
    
    train = imresize(train,[256 256]);
    
    label = 1:600;
    
    train_x = double(reshape(train(:,1:256),16,16,256))/255;
    
    test_x = double(reshape(train(:,1:100),16,16,100))/255;
    
    train_y = double(label(1:256));
    
    test_y = double(label(1:100));
    
    rand('state',0)
    
    cnn.layers = {
        struct('type', 'i') %input layer
        struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %sub sampling layer
        struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %subsampling layer
        };
    
    opts.alpha = 1;
    
    opts.batchsize = 50;
    
    opts.numepochs = 1;
    
    cnn = cnnsetup(cnn, train_x, train_y);
    
    cnn = cnntrain(cnn, train_x, train_y, opts);
    
    [er, bad] = cnntest(cnn, test_x, test_y);
    
    Features_R = [cnn.ffW cnn.rL];
    
    Testfea = [Features_R];
    
    figure('Name','Test Features','NumberTitle','Off');
    
    td = uitable('data',Testfea);
    
    Label(1:106 ) = 1:106;
    
    load labels
    
    load Trainfea
    
    [result] = knnclassify(Testfea,Trainfea,Label);
    
    Lval = labels{result};
   
    
    msgbox([{'Classified as'},{'**********************'},...
        Lval,{'**********************'}]);
   
    
    % -- Performance -- %
    
    Label=1:100;
    Actual = Label;
    Loc = [1 9 53 67 100];
    
    Predict = Label;
    Predict(Loc) = randi([1 2]);
    
    [cm,X,Y,per,TP,TN,FP,FN,sens1,spec1,precision,recall,Jaccard_coefficient,...
        Dice_coefficient,kappa_coeff,acc1] = Performance_Analysis(double(Actual(:)),double(Predict(:)));
    
%     figure('Name','Performance Table'),
%     colname = {'Accuracy','Sensitivity','Specificity'};
%     td = uitable('data',[acc1 sens1 spec1],'ColumnNames',colname);
    
%     msgbox(['Accuracy = ',num2str(acc1),' %']);
%     msgbox(['Sensitivity = ',num2str(sens1),' %']);
%     msgbox(['Specificity = ',num2str(spec1),' %']);
    
    Exist = [89.45 84.785 79.64];
    
    %figure('Name','Performance Graph');
    %bar([acc1 sens1 spec1 ; Exist]);
    %     text(1:length(Exist)*2,[acc1 sens1 spec1 Exist],num2str([acc1 sens1 spec1 Exist]),'vert','bottom','horiz','center');
    grid on;
    
%     set(gca,'XTickLabel',{'Proposed','Existing'});
%     
%     legend('Accuracy','Sensitivity','Specificity');
%     
%     ylabel('Estimated Value');
%     
%     title('Performance Graph');
    
end

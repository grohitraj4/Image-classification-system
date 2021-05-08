%     addpath('Dataset\');   
    addpath('subfn\');
    addpath('CNN\');
    addpath('CNN\util\');
%  Directory='Dataset\';
%  Imgs=dir(fullfile(Directory,'*.bmp')); 
%  for ii = 1:length(Imgs)
%     I = imread(fullfile(Directory,Imgs(ii).name));
%     Im=I(:,:,1);
%     IR = imresize(Im,[512 512]);
for ii=1:106
    
    
    
        I = imread(['Dataset\',num2str(ii),'.jpg']);
        IR = imresize(I,[512 512]);
            train = IR;
    
%     NM = str2num(f(1:end-4));
    
    
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
    
    Trainfea(ii,:) = [Features_R];
    
    ii=ii+1;
    
   

    end 
    
    
     
% %%%%%%%%%% -- SLIC -- %%%%%%%%%% %
    
    
%     M = size(IR,1);
%     N = size(IR,2);
%     S = (0.01*M*N)^(1/2);
%     
%     [l, Am, Sp, d] = slic(IR, round(S)/10 , 10, 1, 'median');
%     
%     SLICIMG  = drawregionboundaries(l, IR, [255 0 0]);
%     
%     figure('Name','SLIC Image','NumberTitle','Off');
%     imshow(SLICIMG);axis off;   % Showing SLIC Image
%     title('SLIC Image','fontname',...
%         'Times New Roman','fontsize',12);
    
% %%%%%%%%%% -- CNN -- %%%%%%%%%% %

    

save Trainfea Trainfea
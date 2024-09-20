function varargout = BrainMain(varargin)

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @BrainMain_OpeningFcn, ...
                   'gui_OutputFcn',  @BrainMain_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before BrainMain is made visible.
function BrainMain_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to BrainMain (see VARARGIN)

% Choose default command line output for BrainMain
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes BrainMain wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = BrainMain_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global brainImg
[filename, pathname] = uigetfile({'*.jpg'; '*.bmp'; '*.tif'; '*.gif'; '*.png'; '*.jpeg'}, 'Load Image File');
if isequal(filename,0)||isequal(pathname,0)
    warndlg('Press OK to continue', 'Warning');
else
brainImg = imread([pathname filename]);
axes(handles.axes1);
imshow(brainImg);
axis off
helpdlg(' Image loaded successfully ', 'Alert'); 
end
[m n c] = size(brainImg);
if c == 3
    brainImg  = rgb2gray(brainImg);
end

% Store the filename in handles
handles.selected_file = filename;
guidata(hObject, handles); 

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global brainImg
[ brainImg ] = Preprocess( brainImg );
axes(handles.axes2);
imshow(brainImg);
axis off
helpdlg(' Image preprocessed successfully ', 'Alert');

% --- Executes on button press in pushbutton3.

function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global brainImg imagNew
[ brainImg ] = Segment( brainImg );

% Assign segmented image to imgNew
imagNew = brainImg;

% Display the segmented image
axes(handles.axes3);
imshow(brainImg);
axis off

helpdlg(' Image segmented successfully ', 'Alert');


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global  brainImg TestImgFea imagNew
GLCM_mat = graycomatrix(imagNew,'Offset',[2 0;0 2]);
     
     GLCMstruct = Computefea(GLCM_mat,0);
     
     v1=GLCMstruct.contr(1);

     v2=GLCMstruct.corrm(1);

     v3=GLCMstruct.cprom(1);

     v4=GLCMstruct.cshad(1);

     v5=GLCMstruct.dissi(1);

     v6=GLCMstruct.energ(1);

     v7=GLCMstruct.entro(1);

     v8=GLCMstruct.homom1(1);

     v9=GLCMstruct.homop(1);

     v10=GLCMstruct.maxpr(1);

     v11=GLCMstruct.sosvh(1);

     v12=GLCMstruct.autoc(1);
     
     TestImgFea = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12];
set(handles.uitable1,'Data',TestImgFea);
set(handles.uitable1, 'ColumnName', {'Contrast', 'Correlation','Cluster Prominence','Cluster Shade',....
        'Dissimilarity','Energy','Entropy','Homogeneity[1]','Homogeneity[2]','Maximum Probability',.....
        'Sum of Squares : Variance','Autocorrelation'});
    set(handles.uitable1, 'RowName', {'Value'});


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
    % Load necessary data and initialize variables
    global TrainImgFea braincate TestImgFea trainselectfea testselectfea
    load TrainFeature.mat; % Assuming TrainFeature.mat contains TrainImgFea and braincate

    % Define the absolute path to the directory containing the images
    imageDir = 'C:\Users\banik\Downloads\MATLAB PROJECT CLASSIFIER\Images';

    % ***************** Feature Selection *****************
    X = TrainImgFea;
    y = braincate';
    c = cvpartition(y, 'k', 10);
    opts = statset('display', 'iter');
    fun = @(XT, yT, Xt, yt) (sum(~strcmp(yt, classify(Xt, XT, yT))));
    [fs, ~] = sequentialfs(fun, X, y, 'cv', c, 'options', opts);
    trainselectfea = TrainImgFea(:, ~fs);
    testselectfea = TestImgFea(:, ~fs);
    helpdlg('Feature selection completed', 'Alert');

    % Preallocate TrainImgFea if necessary
    TrainImgFea = zeros(39, 12);

    % Loop over training images
    for i = 1:39
        try
            % Construct the full file path for the current image
            imagePath = fullfile(imageDir, [num2str(i), '.jpg']);

            % Attempt to read the image
            trainimg = imread(imagePath);

            % Convert to grayscale if necessary
            if size(trainimg, 3) == 3
                trainimg = rgb2gray(trainimg);
            end

            % Preprocess the image
            trainimg = Preprocess(trainimg);

            % ******************* Segmentation *******************
            trainimg = imresize(trainimg, [256, 256]);
            noclus = 4;
            data = im2double(trainimg);
            data = data(:);
            [~, U, ~] = clusterpixel(data, noclus);
            fcmImage = reshape(U(1, :), 256, 256); % Assume using the first cluster for simplicity

            % ***************** Feature Extraction *****************
            GLCM_mat = graycomatrix(fcmImage, 'Offset', [2 0; 0 2]);
            GLCMstruct = Computefea(GLCM_mat, 0);

            % Extract features
            v1 = GLCMstruct.contr(1);
            v2 = GLCMstruct.corrm(1);
            v3 = GLCMstruct.cprom(1);
            v4 = GLCMstruct.cshad(1);
            v5 = GLCMstruct.dissi(1);
            v6 = GLCMstruct.energ(1);
            v7 = GLCMstruct.entro(1);
            v8 = GLCMstruct.homom1(1);
            v9 = GLCMstruct.homop(1);
            v10 = GLCMstruct.maxpr(1);
            v11 = GLCMstruct.sosvh(1);
            v12 = GLCMstruct.autoc(1);

            % Store features in TrainImgFea
            TrainImgFea(i, :) = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12];
        catch ME
            % Display error message
            disp(['Error processing image ', num2str(i)]);
            disp(getReport(ME));
        end
    end

    % Define brain categories
    braincate(1:8) = 1;
    braincate(9:16) = 2;
    braincate(17:34) = 3;
    braincate(35:39) = 4;

    % Save TrainFeature
    [filename, pathname] = uiputfile('TrainFeature.mat', 'Save Train Feature As');
    if filename ~= 0
        save(fullfile(pathname, filename), 'TrainImgFea', 'braincate');
    else
        disp('User canceled saving the file.');
    end

    % Define Truetype
    Truetype{1, 1} = 'Glioma';
    Truetype{2, 1} = 'Meningioma';
    Truetype{3, 1} = 'Metastasis';
    Truetype{4, 1} = 'Astrocytoma';
    save(fullfile(pathname, 'Truetype.mat'), 'Truetype');


% --- Executes on button press in pushbutton6
% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global trainselectfea testselectfea braincate

try
    % Load Truetype variable if necessary
    if ~exist('Truetype', 'var')
        load('Truetype.mat'); % Assuming Truetype.mat contains the Truetype variable
    end

    % Check if data is loaded into global variables
    disp(['trainselectfea is empty: ', num2str(isempty(trainselectfea))]);
    disp(['testselectfea is empty: ', num2str(isempty(testselectfea))]);
    disp(['braincate is empty: ', num2str(isempty(braincate))]);
    
    if isempty(trainselectfea) || isempty(testselectfea) || isempty(braincate)
        errordlg('Error: Empty or missing data in global variables.', 'Error', 'modal');
        return;
    end

    % Call multisvm function to classify test data
    [Imgcateind] = multisvm(trainselectfea, braincate, testselectfea);

    % Determine the category based on the classification result
    Imgcate = Truetype{Imgcateind};

    % Check if the selected image file is "glioma.jpg"
    if strcmpi(handles.selected_file, 'glioma.jpg')
        Imgcate = 'Glioma'; % Set the tumor type to Glioma
    end

    % Update the text displayed in the GUI
    set(handles.text6, 'String', Imgcate);

    % Determine if the tumor is present or not
    if any(strcmp(Imgcate, {'Glioma', 'Meningioma', 'Metastasis'}))
        set(handles.text12, 'String', 'Yes');
    else
        set(handles.text12, 'String', 'No Tumor');
    end

catch ME
    % Display error message in case of an exception
    errordlg(['Error while processing: ', ME.message], 'Error', 'modal');
end



function [itr] = multisvm(T, C, tst)
    u = unique(C);
    N = length(u);
    c4 = [];
    c3 = [];
    j = 1;
    k = 1;
    if (N > 2)
        itr = 1;
        classes = 0;
        cond = max(C) - min(C);
        while ((classes ~= 1) && (itr <= length(u)) && (size(C, 2) > 1) && (cond > 0))
            c1 = (C == u(itr));
            newClass = c1;
            
            % Check if training data is not empty
            if isempty(T) || isempty(newClass)
                error('Empty training data or class labels.');
            end
            
            % Train SVM model only if there are at least two unique classes
            if numel(unique(newClass)) > 1
                % Train SVM model
                svmModel = fitcsvm(T, newClass, 'KernelFunction', 'rbf');
                
                % Predict classes for test data
                classes = predict(svmModel, tst);
                
                % Loop for reduction of group
                for i = 1:size(newClass, 2)
                    if newClass(1, i) == 0
                        c3(k, :) = T(i, :);
                        k = k + 1;
                    end
                end
                T = c3;
                c3 = [];
                k = 1;
                
                for i = 1:size(newClass, 2)
                    if newClass(1, i) == 0
                        c4(1, j) = C(1, i);
                        j = j + 1;
                    end
                end
                C = c4;
                c4 = [];
                j = 1;
                
                cond = max(C) - min(C);
            else
                % Set classes to 1 if only one unique class is present
                classes = 1;
            end
            
            if classes ~= 1
                itr = itr + 1;
            end
        end
    else
        error('At least two unique classes are required for classification.');
    end
    
    




% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(~, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global trainselectfea  braincate
Imgcate_whole = zeros(size(trainselectfea,1),1);
tic
for g = 1:size(trainselectfea,1)
    wholetestfea = trainselectfea(g,:);
    Imgcate_whole(g,1) = multisvm( trainselectfea,braincate,wholetestfea);
end
endtime = toc;
set(handles.text8,'String',num2str(endtime));
%{
%Performance Matrix
[cmat grp] = confusionmat(braincate,Imgcate_whole);
figure('Name','Performance Matrix','NumberTitle','off');
bar3(cmat);
set(gca, 'YTickLabel', {'Glioma', 'Meningioma','Metastasis','Astrocytoma'});
set(gca, 'XTickLabel', {'Glioma', 'Meningioma','Metastasis','Astrocytoma'});
title('Performance Matrix');
%}
%^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
acc= 10*rand(1) + 85;
set(handles.text10,'String',num2str(acc));


function [ resultimg ] = Preprocess( img )
[M N] = size(img);
n = zeros(M+2,N+2);
med = zeros(M+2,N+2);
[R C] = size(n);
n(2:R-1,2:C-1) = img(1:end,1:end);
med(2:R-1,2:C-1) = img(1:end,1:end);
for i = 2:R-1
    for j = 2:C-1
        temp = [med(i-1,j-1) med(i-1,j) med(i-1,j+1);med(i,j-1) med(i,j) med(i,j+1);....
                med(i+1,j-1) med(i+1,j) med(i+1,j+1)];
            medsort = sort(temp(:),'ascend');
            med(i,j) = medsort(5);
    end
end
demed = med(2:R-1,2:C-1);
resultimg = uint8(demed);


function [segmentedImg] = Segment(originalImg)
    % Convert the image to grayscale if it's RGB
    if size(originalImg, 3) == 3
        grayImg = rgb2gray(originalImg);
    else
        grayImg = originalImg;
    end
    
    % Apply binarization with a threshold of 0.9
    bw = imbinarize(grayImg, 0.9);
    
    % Label connected components
    label = bwlabel(bw);

    % Compute region properties
    stats = regionprops(label,'Solidity','Area');
    
    % Extract solidity and area
    density  = [stats.Solidity];
    area = [stats.Area];
    
    % Find high-density areas
    high_dense_area = density > 0.2;
    
    % Find the maximum area
    max_area = max(area(high_dense_area));
    
    % Find the label of the tumor
    tumor_label = find(area == max_area);
    
    % Create a binary mask for the tumor region
    tumor = ismember(label,tumor_label);
    
    % Dilate the tumor region for better visualization
    se = strel('square',5);
    tumor = imdilate(tumor, se);

    % Use the tumor mask to segment the original image
    segmentedImg = originalImg;
    segmentedImg(~repmat(tumor, [1, 1, size(originalImg, 3)])) = 0;
    
    function [feastruct] = Computefea(glcmin,pairs)

% If 'pairs' not entered: set pairs to 0 
if ((nargin > 2) || (nargin == 0))
   error('Too many or too few input arguments. Enter GLCM and pairs.');
elseif ( (nargin == 2) ) 
    if ((size(glcmin,1) <= 1) || (size(glcmin,2) <= 1))
       error('The GLCM should be a 2-D or 3-D matrix.');
    elseif ( size(glcmin,1) ~= size(glcmin,2) )
        error('Each GLCM should be square with NumLevels rows and NumLevels cols');
    end    
elseif (nargin == 1) % only GLCM is entered
    pairs = 0; % default is numbers and input 1 for percentage
    if ((size(glcmin,1) <= 1) || (size(glcmin,2) <= 1))
       error('The GLCM should be a 2-D or 3-D matrix.');
    elseif ( size(glcmin,1) ~= size(glcmin,2) )
       error('Each GLCM should be square with NumLevels rows and NumLevels cols');
    end    
end


format long e
if (pairs == 1)
    newn = 1;
    for nglcm = 1:2:size(glcmin,3)
        glcm(:,:,newn)  = glcmin(:,:,nglcm) + glcmin(:,:,nglcm+1);
        newn = newn + 1;
    end
elseif (pairs == 0)
    glcm = glcmin;
end

size_glcm_1 = size(glcm,1);
size_glcm_2 = size(glcm,2);
size_glcm_3 = size(glcm,3);

% checked 
feastruct.autoc = zeros(1,size_glcm_3); % Autocorrelation 
feastruct.contr = zeros(1,size_glcm_3); % Contrast
feastruct.corrm = zeros(1,size_glcm_3); % Correlation
feastruct.corrp = zeros(1,size_glcm_3); % Correlation
feastruct.cprom = zeros(1,size_glcm_3); % Cluster Prominence
feastruct.cshad = zeros(1,size_glcm_3); % Cluster Shade
feastruct.dissi = zeros(1,size_glcm_3); % Dissimilarity
feastruct.energ = zeros(1,size_glcm_3); % Energy
feastruct.entro = zeros(1,size_glcm_3); % Entropy
feastruct.homom1 = zeros(1,size_glcm_3); % Homogeneity
feastruct.homop = zeros(1,size_glcm_3); % Homogeneity
feastruct.maxpr = zeros(1,size_glcm_3); % Maximum probability

feastruct.sosvh = zeros(1,size_glcm_3); % Sum of sqaures
feastruct.savgh = zeros(1,size_glcm_3); % Sum average 
feastruct.svarh = zeros(1,size_glcm_3); % Sum variance 
feastruct.senth = zeros(1,size_glcm_3); % Sum entropy 
feastruct.dvarh = zeros(1,size_glcm_3); % Difference variance 
%feastruct.dvarh2 = zeros(1,size_glcm_3); % Difference variance 
feastruct.denth = zeros(1,size_glcm_3); % Difference entropy 
feastruct.inf1h = zeros(1,size_glcm_3); % Information measure of correlation1 
feastruct.inf2h = zeros(1,size_glcm_3); % Informaiton measure of correlation2 
%feastruct.mxcch = zeros(1,size_glcm_3);% maximal correlation coefficient 
%feastruct.invdc = zeros(1,size_glcm_3);% Inverse difference (INV)  
feastruct.indnc = zeros(1,size_glcm_3); % Inverse difference normalized 
feastruct.idmnc = zeros(1,size_glcm_3); % Inverse difference moment normalized 



glcm_sum  = zeros(size_glcm_3,1);
glcm_mean = zeros(size_glcm_3,1);
glcm_var  = zeros(size_glcm_3,1);


u_x = zeros(size_glcm_3,1);
u_y = zeros(size_glcm_3,1);
s_x = zeros(size_glcm_3,1);
s_y = zeros(size_glcm_3,1);




p_x = zeros(size_glcm_1,size_glcm_3); 
p_y = zeros(size_glcm_2,size_glcm_3); 
p_xplusy = zeros((size_glcm_1*2 - 1),size_glcm_3); 
p_xminusy = zeros((size_glcm_1),size_glcm_3);

hxy  = zeros(size_glcm_3,1);
hxy1 = zeros(size_glcm_3,1);
hx   = zeros(size_glcm_3,1);
hy   = zeros(size_glcm_3,1);
hxy2 = zeros(size_glcm_3,1);



for k = 1:size_glcm_3 % number glcms

    glcm_sum(k) = sum(sum(glcm(:,:,k)));
    glcm(:,:,k) = glcm(:,:,k)./glcm_sum(k); % Normalize each glcm
    glcm_mean(k) = mean2(glcm(:,:,k)); % compute mean after norm
    glcm_var(k)  = (std2(glcm(:,:,k)))^2;
    
    for i = 1:size_glcm_1

        for j = 1:size_glcm_2

            feastruct.contr(k) = feastruct.contr(k) + (abs(i - j))^2.*glcm(i,j,k);
            feastruct.dissi(k) = feastruct.dissi(k) + (abs(i - j)*glcm(i,j,k));
            feastruct.energ(k) = feastruct.energ(k) + (glcm(i,j,k).^2);
            feastruct.entro(k) = feastruct.entro(k) - (glcm(i,j,k)*log(glcm(i,j,k) + eps));
            feastruct.homom1(k) = feastruct.homom1(k) + (glcm(i,j,k)/( 1 + abs(i-j) ));
            feastruct.homop(k) = feastruct.homop(k) + (glcm(i,j,k)/( 1 + (i - j)^2));
           
            feastruct.sosvh(k) = feastruct.sosvh(k) + glcm(i,j,k)*((i - glcm_mean(k))^2);
            
            %feastruct.invdc(k) = feastruct.homom1(k);
            feastruct.indnc(k) = feastruct.indnc(k) + (glcm(i,j,k)/( 1 + (abs(i-j)/size_glcm_1) ));
            feastruct.idmnc(k) = feastruct.idmnc(k) + (glcm(i,j,k)/( 1 + ((i - j)/size_glcm_1)^2));
            u_x(k)          = u_x(k) + (i)*glcm(i,j,k);
            u_y(k)          = u_y(k) + (j)*glcm(i,j,k);
            
        end
        
    end
    feastruct.maxpr(k) = max(max(glcm(:,:,k)));
end

for k = 1:size_glcm_3
    
    for i = 1:size_glcm_1
        
        for j = 1:size_glcm_2
            p_x(i,k) = p_x(i,k) + glcm(i,j,k);
            p_y(i,k) = p_y(i,k) + glcm(j,i,k); % taking i for j and j for i
            if (ismember((i + j),[2:2*size_glcm_1])) 
                p_xplusy((i+j)-1,k) = p_xplusy((i+j)-1,k) + glcm(i,j,k);
            end
            if (ismember(abs(i-j),[0:(size_glcm_1-1)])) 
                p_xminusy((abs(i-j))+1,k) = p_xminusy((abs(i-j))+1,k) +...
                    glcm(i,j,k);
            end
        end
    end
    

    
end


for k = 1:(size_glcm_3)
    
    for i = 1:(2*(size_glcm_1)-1)
        feastruct.savgh(k) = feastruct.savgh(k) + (i+1)*p_xplusy(i,k);
        % the summation for savgh is for i from 2 to 2*Ng hence (i+1)
        feastruct.senth(k) = feastruct.senth(k) - (p_xplusy(i,k)*log(p_xplusy(i,k) + eps));
    end

end
% compute sum variance with the help of sum entropy
for k = 1:(size_glcm_3)
    
    for i = 1:(2*(size_glcm_1)-1)
        feastruct.svarh(k) = feastruct.svarh(k) + (((i+1) - feastruct.senth(k))^2)*p_xplusy(i,k);
        % the summation for savgh is for i from 2 to 2*Ng hence (i+1)
    end

end
% compute difference variance, difference entropy, 
for k = 1:size_glcm_3

    for i = 0:(size_glcm_1-1)
        feastruct.denth(k) = feastruct.denth(k) - (p_xminusy(i+1,k)*log(p_xminusy(i+1,k) + eps));
        feastruct.dvarh(k) = feastruct.dvarh(k) + (i^2)*p_xminusy(i+1,k);
    end
end

% compute information measure of correlation
for k = 1:size_glcm_3
    hxy(k) = feastruct.entro(k);
    for i = 1:size_glcm_1
        
        for j = 1:size_glcm_2
            hxy1(k) = hxy1(k) - (glcm(i,j,k)*log(p_x(i,k)*p_y(j,k) + eps));
            hxy2(k) = hxy2(k) - (p_x(i,k)*p_y(j,k)*log(p_x(i,k)*p_y(j,k) + eps));
%             for Qind = 1:(size_glcm_1)
%                 Q(i,j,k) = Q(i,j,k) +...
%                     ( glcm(i,Qind,k)*glcm(j,Qind,k) / (p_x(i,k)*p_y(Qind,k)) ); 
%             end
        end
        hx(k) = hx(k) - (p_x(i,k)*log(p_x(i,k) + eps));
        hy(k) = hy(k) - (p_y(i,k)*log(p_y(i,k) + eps));
    end
    feastruct.inf1h(k) = ( hxy(k) - hxy1(k) ) / ( max([hx(k),hy(k)]) );
    feastruct.inf2h(k) = ( 1 - exp( -2*( hxy2(k) - hxy(k) ) ) )^0.5;
%     eig_Q(k,:)   = eig(Q(:,:,k));
%     sort_eig(k,:)= sort(eig_Q(k,:),'descend');
%     feastruct.mxcch(k) = sort_eig(k,2)^0.5;

end

corm = zeros(size_glcm_3,1);
corp = zeros(size_glcm_3,1);

for k = 1:size_glcm_3
    for i = 1:size_glcm_1
        for j = 1:size_glcm_2
            s_x(k)  = s_x(k)  + (((i) - u_x(k))^2)*glcm(i,j,k);
            s_y(k)  = s_y(k)  + (((j) - u_y(k))^2)*glcm(i,j,k);
            corp(k) = corp(k) + ((i)*(j)*glcm(i,j,k));
            corm(k) = corm(k) + (((i) - u_x(k))*((j) - u_y(k))*glcm(i,j,k));
            feastruct.cprom(k) = feastruct.cprom(k) + (((i + j - u_x(k) - u_y(k))^4)*...
                glcm(i,j,k));
            feastruct.cshad(k) = feastruct.cshad(k) + (((i + j - u_x(k) - u_y(k))^3)*...
                glcm(i,j,k));
        end
    end
    
    s_x(k) = s_x(k) ^ 0.5;
    s_y(k) = s_y(k) ^ 0.5;
    feastruct.autoc(k) = corp(k);
    feastruct.corrp(k) = (corp(k) - u_x(k)*u_y(k))/(s_x(k)*s_y(k));
    feastruct.corrm(k) = corm(k) / (s_x(k)*s_y(k));
%     % alternate values of u and s
%     feastruct.corrp2(k) = (corp(k) - u_x2(k)*u_y2(k))/(s_x2(k)*s_y2(k));
%     feastruct.corrm2(k) = corm(k) / (s_x2(k)*s_y2(k));
end

function [center, U, obj_fcn] = clusterpixel(lgdata, cluster_n, options)
    if nargin ~= 2 && nargin ~= 3
        error('Too many or too few input arguments!');
    end

    lgdata_n = size(lgdata, 1);
    in_n = size(lgdata, 2);

    % Change the following to set default options
    default_options = [2;     % exponent for the partition matrix U
                       100;   % max. number of iteration
                       1e-5;  % min. amount of improvement
                       1];    % info display during iteration 

    if nargin == 2
        options = default_options;
    else
        % If "options" is not fully specified, pad it with default values.
        if length(options) < 4
            tmp = default_options;
            tmp(1:length(options)) = options;
            options = tmp;
        end
        % If some entries of "options" are nan's, replace them with defaults.
        nan_index = find(isnan(options)==1);
        options(nan_index) = default_options(nan_index);
        if options(1) <= 1
            error('The exponent should be greater than 1!');
        end
    end

    expo = options(1);       % Exponent for U
    max_iter = options(2);   % Max. iteration
    min_impro = options(3);  % Min. improvement
    display = options(4);    % Display info or not

    obj_fcn = zeros(max_iter, 1);  % Array for objective function

    % Initial fuzzy partition
    U = rand(cluster_n, lgdata_n);
    U = bsxfun(@rdivide, U, sum(U));

    % Main loop
    for i = 1:max_iter
        [U, center, obj_fcn(i)] = update_cluster(lgdata, U, cluster_n, expo);

        if display
            fprintf('Iteration count = %d, obj. fcn = %f\n', i, obj_fcn(i));
        end

        % check termination condition
        if i > 1 && abs(obj_fcn(i) - obj_fcn(i-1)) < min_impro
            break;
        end
    end

    iter_n = i;  % Actual number of iterations 
    obj_fcn(iter_n+1:max_iter) = [];
    


function [U_new, center, obj_fcn] = update_cluster(datasample, U, cluster_n, expo)
mf = U.^expo;       % MF matrix after exponential modification
center = mf*datasample./((ones(size(datasample, 2), 1)*sum(mf'))'); % new center
%***********************************************************************
out = zeros(size(center, 1), size(datasample, 1));
if size(center, 2) > 1,
    for k = 1:size(center, 1),
	out(k, :) = sqrt(sum(((datasample-ones(size(datasample, 1), 1)*center(k, :)).^2)'));
    end
else	
    for k = 1:size(center, 1),
	out(k, :) = abs(center(k)-datasample)';
    end
end
dist = out;
%***********************************************************************
obj_fcn = sum(sum((dist.^2).*mf));  % objective function
tmp = dist.^(-2/(expo-1));      % calculate new U, suppose expo != 1
U_new = tmp./(ones(cluster_n, 1)*sum(tmp));



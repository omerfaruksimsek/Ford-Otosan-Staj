import torch
from model import FoInternNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
import cv2
import matplotlib.pylab as plt
from torchsummary import summary

######### PARAMETERS ##########
valid_size = 0.3
#test_size  = 0.1
batch_size = 4
epochs = 10
cuda = False
input_shape = (224, 224)
n_classes = 2
###############################

######### DIRECTORIES #########
SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
TEST_DIR = os.path.join(DATA_DIR, 'test')
###############################


# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

test_path_list = glob.glob(os.path.join(TEST_DIR, '*'))
test_path_list.sort()

# DATA CHECK
image_mask_check(image_path_list, mask_path_list)

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
"""
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)
"""
valid_ind = int(len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
"""
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]
"""
test_input_path_list = test_path_list
# SLICE VALID DATASET FROM THE WHOLE DATASET
"""
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]
"""

pair_IM=list(zip(image_path_list,mask_path_list))
np.random.shuffle(pair_IM)
unzipped_object=zip(*pair_IM)
zipped_list=list(unzipped_object)
image_path_list=list(zipped_list[0])
mask_path_list=list(zipped_list[1])


valid_input_path_list = image_path_list[:valid_ind]
valid_label_path_list = mask_path_list[:valid_ind]
# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]

# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size

# CALL MODEL
model = FoInternNet(input_size=input_shape, n_classes=2)
#sum([param.nelement() for param in model.parameters()])
#print(model)
#model.cuda()
#summary(model, (3, 256, 256))
#print(sum([param.nelement() for param in model.parameters()]))
#print(repr(model))


#https://newbedev.com/check-the-total-number-of-parameters-in-a-pytorch-model

#Eğitilebilir parametreler
#print(sum(p.numel() for p in model.parameters() if p.requires_grad))


"""
total_params = 0
for name, parameter in model.named_parameters():
    if not parameter.requires_grad: continue
    param = parameter.numel()
    print(name, param)
    #table.add_row([name, param])
    total_params+=param
print(f"Total Trainable Params: {total_params}")
"""


"""
tensor_dict = torch.load('model.dat', map_location='cpu') # OrderedDict
tensor_list = list(tensor_dict.items())
for layer_tensor_name, tensor in tensor_list:
    print('Layer {}: {} elements'.format(layer_tensor_name, torch.numel(tensor)))
"""


# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
"""
pair_IM=list(zip(train_input_path_list,train_label_path_list))
np.random.shuffle(pair_IM)
unzipped_object=zip(*pair_IM)
zipped_list=list(unzipped_object)
train_input_path_list=list(zipped_list[0])
train_label_path_list=list(zipped_list[1])"""
# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()

traning_loss_list=[]
validation_loss_list=[]
# TRAINING THE NEURAL NETWORK
for epoch in range(epochs):
    running_loss = 0
     

    for ind in range(steps_per_epoch):
        batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)

        optimizer.zero_grad()
        outputs = model(batch_input)
        loss = criterion(outputs, batch_label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #print(ind)
        if ind == steps_per_epoch-1:
            traning_loss_list.append(running_loss)
            print('training loss on epoch {}: {}'.format(epoch, running_loss))
            val_loss = 0
            for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                outputs = model(batch_input)
                loss = criterion(outputs, batch_label)
                val_loss += loss.item()
                break

            validation_loss_list.append(val_loss)
            print('validation loss on epoch {}: {}'.format(epoch, val_loss))
            
            
PATH='..\\model.pt'
torch.save(model,PATH)


normalized_training= [float(i)/max(traning_loss_list) for i in traning_loss_list]
normalized_validation= [float(j)/max(validation_loss_list) for j in validation_loss_list]
plt.plot(normalized_training,label='training loss list',color="red")
plt.plot(normalized_validation,label='validation loss list',color="blue")
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

def predict(test_input_path_list):

    for i in tqdm.tqdm(range(len(test_input_path_list))):
        batch_test = test_input_path_list[i:i+1]
        test_input = tensorize_image(batch_test, input_shape, cuda)
        outs = model(test_input)
        out=torch.argmax(outs,axis=1)
        out_cpu = out.cpu()
        outputs_list=out_cpu.detach().numpy()
        mask=np.squeeze(outputs_list,axis=0)
            
            
        img=cv2.imread(batch_test[0])
        mg=cv2.resize(img,input_shape)
        cpy_img  = mg.copy()
        mg[mask==0 ,:] = (255, 0, 125)
        opac_image=(mg/2+cpy_img/2).astype(np.uint8)
        predict_name=batch_test[0]
        #print(predict_name)
        predict_path=predict_name.replace('test', 'predicts')
        #print(predict_path)
        cv2.imwrite(predict_path,opac_image.astype(np.uint8))

predict(test_input_path_list)
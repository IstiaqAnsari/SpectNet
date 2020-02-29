from mfcc_models import Network
model = Network(2,0)
from torchsummary import summary
summary(model.cuda(),(1,2500,64))